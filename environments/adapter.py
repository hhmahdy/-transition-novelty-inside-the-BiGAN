"""Environment API adapter for the Adventurer training pipeline.

The adapter isolates environment API details from PPO and BiGAN code. It
supports modern Gymnasium five-value steps and legacy four-value steps while
preserving the distinction between true termination and time-limit
truncation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from utils.seed import seed_gym_environment


@dataclass(frozen=True)
class EnvironmentStep:
    """Normalized result of one environment transition."""

    observation: Tensor
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    @property
    def done(self) -> bool:
        """Return whether the episode boundary requires trainer handling.

        Input: This normalized transition.
        Output: ``terminated or truncated``.
        Mathematical meaning: Identifies whether the trajectory boundary must
            be represented by a reset before collecting the next transition.
        """
        return self.terminated or self.truncated


class SingleEnvironmentAdapter:
    """Wrap one Gymnasium-compatible environment for tensor-based training.

    Args:
        environment: Object implementing ``reset``, ``step``, and ``close``.
        device: Device for returned observation tensors.
        observation_dtype: Tensor dtype for observations.
        action_dtype: Tensor dtype used when converting tensor actions back to
            the environment.
    """

    def __init__(
        self,
        environment: Any,
        device: Union[torch.device, str] = "cpu",
        observation_dtype: torch.dtype = torch.float32,
        action_dtype: torch.dtype = torch.int64,
    ) -> None:
        """Initialize and validate the wrapped environment.

        Input: Environment object and tensor conversion settings.
        Output: An initialized environment adapter.
        Mathematical meaning: Establishes the state/action interface through
            which PPO observes transitions from the MDP.
        """
        for method_name in ("reset", "step"):
            if not callable(getattr(environment, method_name, None)):
                raise TypeError(f"environment must provide callable {method_name}")
        self.environment = environment
        self.device = torch.device(device)
        self.observation_dtype = observation_dtype
        self.action_dtype = action_dtype
        self._observation_keys = self._resolve_observation_keys()
        self._needs_reset = True

    def _resolve_observation_keys(self) -> Optional[Tuple[str, ...]]:
        """Resolve deterministic keys for a dictionary observation space.

        Input: Wrapped environment observation space.
        Output: Canonical tuple ``(observation, achieved_goal, desired_goal)``
            when available, otherwise ``None`` for array spaces.
        Mathematical meaning: Defines the fixed state vector presented to the
            policy for goal-conditioned environments.
        """
        observation_space = getattr(self.environment, "observation_space", None)
        spaces = getattr(observation_space, "spaces", None)
        if spaces is None:
            return None
        preferred = ("observation", "achieved_goal", "desired_goal")
        keys = tuple(key for key in preferred if key in spaces)
        if not keys:
            keys = tuple(sorted(spaces.keys()))
        return keys

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        """Return the flattened state shape consumed by neural networks.

        Input: This adapter.
        Output: Fixed tuple shape after dictionary observations are flattened.
        Mathematical meaning: Defines the state domain of PPO and BiGAN.
        """
        if self._observation_keys is not None:
            spaces = self.environment.observation_space.spaces
            size = sum(int(np.prod(spaces[key].shape)) for key in self._observation_keys)
            return (size,)
        shape = getattr(self.environment.observation_space, "shape", None)
        if shape is None:
            raise ValueError("observation space must expose shape or dictionary spaces")
        return tuple(int(dimension) for dimension in shape)

    @staticmethod
    def _unwrap_reset(result: Any) -> tuple[Any, dict[str, Any]]:
        """Normalize modern and legacy reset return values.

        Input: Raw environment reset result, either ``observation`` or
            ``(observation, info)``.
        Output: ``(observation, info)`` with an always-present info dictionary.
        Mathematical meaning: Extracts the initial MDP state and metadata.
        """
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            return result[0], result[1]
        return result, {}

    def _observation_tensor(self, observation: Any) -> Tensor:
        """Convert one raw environment observation to a tensor.

        Input: NumPy array, Python sequence, or tensor observation.
        Output: Tensor on the adapter device using ``observation_dtype``.
        Mathematical meaning: Converts state ``s_t`` into the representation
            consumed by the actor, critic, encoder, and novelty estimator.
        """
        if self._observation_keys is not None:
            if not isinstance(observation, dict):
                raise TypeError("dictionary observation space requires a dictionary observation")
            components = []
            for key in self._observation_keys:
                component = observation[key]
                if isinstance(component, Tensor):
                    component_tensor = component.detach().float().reshape(-1)
                else:
                    component_tensor = torch.as_tensor(np.asarray(component)).float().reshape(-1)
                components.append(component_tensor)
            tensor = torch.cat(components, dim=0)
        elif isinstance(observation, Tensor):
            tensor = observation.detach()
        else:
            tensor = torch.as_tensor(np.asarray(observation))
        return tensor.to(device=self.device, dtype=self.observation_dtype)

    def reset(self, seed: Optional[int] = None) -> Tuple[Tensor, Dict[str, Any]]:
        """Reset the environment and return its initial tensor observation.

        Input: Optional non-negative episode seed.
        Output: ``(observation, info)`` with observation as a device tensor.
        Mathematical meaning: Samples an initial state ``s_0`` from the seeded
            environment initial-state distribution.
        """
        if seed is not None and seed < 0:
            raise ValueError("seed must be non-negative")
        raw_result = self.environment.reset(seed=seed) if seed is not None else self.environment.reset()
        raw_observation, info = self._unwrap_reset(raw_result)
        self._needs_reset = False
        return self._observation_tensor(raw_observation), info

    def seed(self, seed: int) -> None:
        """Seed the wrapped environment and action space.

        Input: Non-negative environment seed.
        Output: No value; supported environment RNGs are seeded.
        Mathematical meaning: Fixes the MDP transition and initial-state random
            process for reproducible trajectory collection.
        """
        seed_gym_environment(self.environment, seed)
        self._needs_reset = True

    def _environment_action(self, action: Union[Tensor, int, float, np.ndarray]) -> Any:
        """Convert a tensor action to the environment's native action type.

        Input: Scalar, NumPy action, or tensor action.
        Output: Python scalar or NumPy array suitable for ``environment.step``.
        Mathematical meaning: Converts policy action ``a_t`` without changing
            its value or categorical/continuous structure.
        """
        if isinstance(action, Tensor):
            action = action.detach().to("cpu")
            if action.numel() == 1:
                return action.item()
            return action.numpy().astype(np.float32 if action.is_floating_point() else np.int64)
        return action

    def step(self, action: Union[Tensor, int, float, np.ndarray]) -> EnvironmentStep:
        """Execute one environment transition using a policy action.

        Input: Action produced by the actor for the current state.
        Output: Normalized ``EnvironmentStep`` containing next state, reward,
            terminal/truncation flags, and info.
        Mathematical meaning: Executes the MDP transition
            ``s_(t+1), r_t, terminated_t, truncated_t = P(s_t,a_t)``.
        """
        if self._needs_reset:
            raise RuntimeError("environment must be reset before step")
        result = self.environment.step(self._environment_action(action))
        if not isinstance(result, tuple) or len(result) not in (4, 5):
            raise ValueError("environment.step must return four or five values")
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
        else:
            observation, reward, done, info = result
            terminated = bool(done)
            truncated = bool(info.get("TimeLimit.truncated", False)) if isinstance(info, dict) else False
            terminated = terminated and not truncated
        if not isinstance(info, dict):
            info = {"raw_info": info}
        self._needs_reset = bool(terminated or truncated)
        return EnvironmentStep(
            observation=self._observation_tensor(observation),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
        )

    def _state_backend(self) -> Any:
        """Return the most likely backend exposing simulator-state methods.

        Input: This wrapped environment.
        Output: Unwrapped environment or nested ALE backend object.
        Mathematical meaning: Identifies the state-transition implementation
            whose state defines the resettable MDP configuration.
        """
        unwrapped = getattr(self.environment, "unwrapped", self.environment)
        ale_backend = getattr(unwrapped, "ale", None)
        return ale_backend if ale_backend is not None else unwrapped

    def supports_state_restore(self) -> bool:
        """Return whether simulator snapshots can be cloned and restored.

        Input: This adapter.
        Output: ``True`` only when both clone and restore methods are available.
        Mathematical meaning: Determines whether Algorithm 2's resettable
            premise is valid for this environment.
        """
        backend = self._state_backend()
        clone_available = callable(getattr(backend, "clone_state", None)) or callable(
            getattr(backend, "cloneState", None)
        )
        restore_available = callable(getattr(backend, "restore_state", None)) or callable(
            getattr(backend, "restoreState", None)
        )
        return clone_available and restore_available

    def clone_state(self) -> Any:
        """Capture a simulator state for later episodic-memory restoration.

        Input: Adapter wrapping an environment with clone-state support.
        Output: Opaque simulator snapshot.
        Mathematical meaning: Stores the state associated with a high-novelty
            observation for later sampling from episodic memory.
        """
        backend = self._state_backend()
        clone_method = getattr(backend, "clone_state", None) or getattr(backend, "cloneState", None)
        if not callable(clone_method):
            raise RuntimeError("environment does not support simulator-state cloning")
        return clone_method()

    def restore_state(self, snapshot: Any) -> Tensor:
        """Restore a simulator snapshot and recover its current observation.

        Input: Opaque snapshot previously returned by ``clone_state``.
        Output: Observation tensor corresponding to the restored simulator state.
        Mathematical meaning: Implements Algorithm 2's return to a novel state
            sampled from the previous epoch episodic memory.
        """
        backend = self._state_backend()
        restore_method = getattr(backend, "restore_state", None) or getattr(backend, "restoreState", None)
        if not callable(restore_method):
            raise RuntimeError("environment does not support simulator-state restoration")
        restore_method(snapshot)
        unwrapped = getattr(self.environment, "unwrapped", self.environment)
        observation_method = getattr(unwrapped, "_get_obs", None) or getattr(unwrapped, "get_obs", None)
        if callable(observation_method):
            observation = observation_method()
        else:
            ale_backend = getattr(unwrapped, "ale", None)
            screen_method = getattr(ale_backend, "getScreenRGB", None)
            if not callable(screen_method):
                raise RuntimeError(
                    "restored environment does not expose a method to recover its observation"
                )
            observation = np.asarray(screen_method())
        self._needs_reset = False
        return self._observation_tensor(observation)

    def close(self) -> None:
        """Close the wrapped environment if it exposes a close method.

        Input: This adapter.
        Output: No value; environment resources are released when supported.
        Mathematical meaning: Ends interaction with the sampled MDP instance.
        """
        close_method = getattr(self.environment, "close", None)
        if callable(close_method):
            close_method()

    def __enter__(self) -> "SingleEnvironmentAdapter":
        """Enter a context manager and return this adapter.

        Input: This adapter.
        Output: The same adapter instance.
        Mathematical meaning: Establishes a bounded environment lifetime.
        """
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Close the environment when leaving a context-manager block.

        Input: Standard context-manager exception information.
        Output: No value; wrapped environment is closed.
        Mathematical meaning: Guarantees environment resources are released.
        """
        self.close()


def make_gymnasium_environment(
    environment_id: str,
    **kwargs: Any,
) -> SingleEnvironmentAdapter:
    """Construct a Gymnasium environment through an explicit factory.

    Input: Registered Gymnasium environment ID and keyword arguments accepted
        by ``gymnasium.make``.
    Output: ``SingleEnvironmentAdapter`` around the created environment.
    Mathematical meaning: Instantiates the MDP whose transition samples are
        used to optimize PPO and collect BiGAN observations.

    The repository's own Noisy-TV environments are registered here, so
    ``--env NoisyTVMaze-v0`` (the pure-python reconstruction of the maze) and
    ``--env NoisyTVUnity-v0`` (the original Unity build, when its executable and
    the upstream ``unityagents`` client are available) work without an extra
    import step. Registration is a no-op if a different environment already
    claims an ID.
    """
    try:
        import gymnasium as gym
        from environments.noisy_tv_unity import register_environment

        register_environment()
        if environment_id.startswith("ALE/"):
            import ale_py  # noqa: F401
        if environment_id.startswith(("Fetch", "Hand", "Adroit", "PointMaze", "AntMaze")):
            import gymnasium_robotics
            gym.register_envs(gymnasium_robotics)
    except ImportError as error:
        raise ImportError(
            "required Gymnasium environment package is not installed"
        ) from error
    return SingleEnvironmentAdapter(gym.make(environment_id, **kwargs))
