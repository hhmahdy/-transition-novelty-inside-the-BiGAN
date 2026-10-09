"""On-policy rollout storage for separate extrinsic and intrinsic streams.

The paper's Adventurer objective estimates extrinsic and intrinsic advantages
independently. This buffer therefore never requires a pre-combined reward. It
stores the two reward streams and both value predictions, while minibatches
carry the combined policy advantage and separate critic targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Union

import torch
from torch import Tensor


@dataclass(frozen=True)
class RolloutBatch:
    """Flattened transition data consumed by two-stream PPO."""

    observations: Tensor
    actions: Tensor
    old_log_probabilities: Tensor
    extrinsic_rewards: Tensor
    intrinsic_rewards: Tensor
    terminated: Tensor
    truncated: Tensor
    extrinsic_values: Tensor
    intrinsic_values: Tensor
    advantages: Tensor
    extrinsic_advantages: Tensor
    intrinsic_advantages: Tensor
    extrinsic_returns: Tensor
    intrinsic_returns: Tensor

    def __len__(self) -> int:
        """Return the number of flattened transitions.

        Input: This rollout minibatch.
        Output: Number of samples represented by the first tensor dimension.
        Mathematical meaning: Gives the empirical sample count for the PPO
            expectation over the combined advantage.
        """
        return int(self.extrinsic_rewards.shape[0])


class RolloutBuffer:
    """Store one fixed-length rollout with separate reward/value streams.

    Args:
        num_steps: Number of transitions collected before one PPO update.
        num_envs: Number of parallel environments.
        observation_shape: Shape of one environment observation.
        action_shape: Shape of one action excluding environment dimension.
        device: Device on which storage tensors are allocated.
        observation_dtype: Observation storage dtype.
        action_dtype: Action storage dtype.
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        observation_shape: tuple[int, ...],
        action_shape: tuple[int, ...] = (),
        device: Union[torch.device, str] = "cpu",
        observation_dtype: torch.dtype = torch.float32,
        action_dtype: torch.dtype = torch.long,
    ) -> None:
        """Allocate validated two-stream rollout storage.

        Input: Rollout capacity, environment count, tensor shapes, device, and
            storage dtypes.
        Output: Empty storage for ``num_steps`` transitions.
        Mathematical meaning: Allocates samples for
            ``(r_t^e,r_t^i,V_t^e,V_t^i)`` without constructing a mixed reward.
        """
        if num_steps <= 0 or num_envs <= 0:
            raise ValueError("num_steps and num_envs must be positive")
        if not observation_shape or any(d <= 0 for d in observation_shape):
            raise ValueError("observation_shape must contain positive dimensions")
        if any(d <= 0 for d in action_shape):
            raise ValueError("action_shape must contain positive dimensions")
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.device = torch.device(device)
        self.observations = torch.empty(
            (num_steps + 1, num_envs, *observation_shape),
            dtype=observation_dtype,
            device=self.device,
        )
        self.actions = torch.empty(
            (num_steps, num_envs, *action_shape),
            dtype=action_dtype,
            device=self.device,
        )
        scalar_shape = (num_steps, num_envs)
        self.log_probabilities = torch.empty(scalar_shape, dtype=torch.float32, device=self.device)
        self.extrinsic_rewards = torch.empty(scalar_shape, dtype=torch.float32, device=self.device)
        self.intrinsic_rewards = torch.empty(scalar_shape, dtype=torch.float32, device=self.device)
        self.terminated = torch.empty(scalar_shape, dtype=torch.bool, device=self.device)
        self.truncated = torch.empty(scalar_shape, dtype=torch.bool, device=self.device)
        self.extrinsic_values = torch.empty(scalar_shape, dtype=torch.float32, device=self.device)
        self.intrinsic_values = torch.empty(scalar_shape, dtype=torch.float32, device=self.device)
        self._position = 0
        self._has_final_observation = False

    @property
    def position(self) -> int:
        """Return the number of transitions currently stored.

        Input: This buffer.
        Output: Integer in ``[0,num_steps]``.
        Mathematical meaning: Identifies the current trajectory time index.
        """
        return self._position

    @property
    def is_full(self) -> bool:
        """Return whether the configured rollout horizon is full.

        Input: This buffer.
        Output: Boolean indicating whether PPO may consume the rollout.
        Mathematical meaning: Ensures one complete on-policy sample set is
            available before policy optimization.
        """
        return self._position == self.num_steps

    def reset(self) -> None:
        """Reset the logical insertion state without reallocating tensors.

        Input: This buffer.
        Output: No value; starts a fresh on-policy trajectory collection.
        Mathematical meaning: Prevents samples from different behavior policies
            from being mixed in one PPO update.
        """
        self._position = 0
        self._has_final_observation = False

    def add(
        self,
        observation: Tensor,
        action: Tensor,
        extrinsic_reward: Tensor,
        intrinsic_reward: Tensor,
        terminated: Tensor,
        truncated: Tensor,
        extrinsic_value: Tensor,
        intrinsic_value: Tensor,
        log_probability: Tensor,
        next_observation: Tensor,
    ) -> None:
        """Append one transition for each parallel environment.

        Input: ``s_t``, ``a_t``, separate rewards ``r_t^e``/``r_t^i``, masks,
            separate values ``V_t^e``/``V_t^i``, old policy log probability,
            and ``s_(t+1)``.
        Output: No value; writes one time slice and advances the cursor.
        Mathematical meaning: Stores all terms needed for two independent TD
            residuals and later advantage combination.
        """
        if self.is_full:
            raise RuntimeError("cannot add to a full rollout buffer")
        index = self._position
        observation = observation.to(self.device)
        next_observation = next_observation.to(self.device)
        action = action.to(self.device)
        extrinsic_reward = extrinsic_reward.to(self.device).float().reshape(self.num_envs)
        intrinsic_reward = intrinsic_reward.to(self.device).float().reshape(self.num_envs)
        terminated = terminated.to(self.device).bool().reshape(self.num_envs)
        truncated = truncated.to(self.device).bool().reshape(self.num_envs)
        extrinsic_value = extrinsic_value.to(self.device).float().reshape(self.num_envs)
        intrinsic_value = intrinsic_value.to(self.device).float().reshape(self.num_envs)
        log_probability = log_probability.to(self.device).float().reshape(self.num_envs)
        if observation.shape != (self.num_envs, *self.observation_shape):
            raise ValueError("observation has the wrong shape")
        if next_observation.shape != (self.num_envs, *self.observation_shape):
            raise ValueError("next_observation has the wrong shape")
        if action.shape != (self.num_envs, *self.action_shape):
            raise ValueError("action has the wrong shape")
        self.observations[index].copy_(observation)
        self.observations[index + 1].copy_(next_observation)
        self.actions[index].copy_(action)
        self.extrinsic_rewards[index].copy_(extrinsic_reward)
        self.intrinsic_rewards[index].copy_(intrinsic_reward)
        self.terminated[index].copy_(terminated)
        self.truncated[index].copy_(truncated)
        self.extrinsic_values[index].copy_(extrinsic_value)
        self.intrinsic_values[index].copy_(intrinsic_value)
        self.log_probabilities[index].copy_(log_probability)
        self._position += 1

    def set_final_observation(self, observation: Tensor) -> None:
        """Store the final bootstrap observation after a full rollout.

        Input: ``s_T`` for every parallel environment.
        Output: No value; writes the final observation slot.
        Mathematical meaning: Supplies the state needed to calculate terminal
            TD bootstrap values for both reward streams.
        """
        if not self.is_full:
            raise RuntimeError("final observation requires a full rollout")
        observation = observation.to(self.device)
        expected_shape = (self.num_envs, *self.observation_shape)
        if observation.shape != expected_shape:
            raise ValueError("final observation has the wrong shape")
        self.observations[self.num_steps].copy_(observation)
        self._has_final_observation = True

    def tensors(self) -> dict[str, Tensor]:
        """Return time-major transition tensors for two-stream GAE.

        Input: Full buffer with a final observation.
        Output: Dictionary containing both rewards, both value streams, masks,
            observations, actions, and old log probabilities.
        Mathematical meaning: Exposes the separate trajectories required for
            ``A^e`` and ``A^i`` estimation.
        """
        if not self.is_full or not self._has_final_observation:
            raise RuntimeError("a full rollout with a final observation is required")
        return {
            "observations": self.observations,
            "actions": self.actions,
            "extrinsic_rewards": self.extrinsic_rewards,
            "intrinsic_rewards": self.intrinsic_rewards,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "extrinsic_values": self.extrinsic_values,
            "intrinsic_values": self.intrinsic_values,
            "log_probabilities": self.log_probabilities,
        }

    def iter_minibatches(
        self,
        advantages: Tensor,
        extrinsic_advantages: Tensor,
        intrinsic_advantages: Tensor,
        extrinsic_returns: Tensor,
        intrinsic_returns: Tensor,
        minibatch_size: int,
        beta: float,
        generator: Optional[torch.Generator] = None,
    ) -> Iterator[RolloutBatch]:
        """Yield shuffled PPO minibatches with separate critic targets.

        Input: Combined policy advantages, both uncombined advantage streams,
            both return targets, minibatch size, intrinsic advantage coefficient,
            and optional random generator.
        Output: Iterator yielding every rollout sample exactly once per epoch.
        Mathematical meaning: Combines only advantages as
            ``A=A^e+beta*A^i``; rewards and value targets remain separate.
        """
        if not self.is_full or not self._has_final_observation:
            raise RuntimeError("a full rollout with a final observation is required")
        expected = (self.num_steps, self.num_envs)
        tensors = (advantages, extrinsic_advantages, intrinsic_advantages, extrinsic_returns, intrinsic_returns)
        if any(tensor.shape != expected for tensor in tensors):
            raise ValueError("all advantage and return tensors must have [num_steps,num_envs] shape")
        if minibatch_size <= 0 or (self.num_steps * self.num_envs) % minibatch_size != 0:
            raise ValueError("rollout size must be divisible by minibatch_size")
        if beta < 0.0:
            raise ValueError("beta must be non-negative")
        total = self.num_steps * self.num_envs

        def flatten(tensor: Tensor) -> Tensor:
            """Flatten time and environment dimensions.

            Input: Time-major tensor with first dimensions ``[T,N]``.
            Output: Tensor with sample dimension ``[T*N]``.
            Mathematical meaning: Converts trajectory samples into PPO
                minibatch rows without changing per-sample quantities.
            """
            return tensor.reshape(total, *tensor.shape[2:])

        indices = torch.randperm(total, generator=generator, device=self.device)
        batch_tensors = {
            "observations": flatten(self.observations[:-1]),
            "actions": flatten(self.actions),
            "old_log_probabilities": flatten(self.log_probabilities),
            "extrinsic_rewards": flatten(self.extrinsic_rewards),
            "intrinsic_rewards": flatten(self.intrinsic_rewards),
            "terminated": flatten(self.terminated),
            "truncated": flatten(self.truncated),
            "extrinsic_values": flatten(self.extrinsic_values),
            "intrinsic_values": flatten(self.intrinsic_values),
        }
        flat_advantages = flatten(advantages.to(self.device))
        flat_extrinsic_advantages = flatten(extrinsic_advantages.to(self.device))
        flat_intrinsic_advantages = flatten(intrinsic_advantages.to(self.device))
        flat_extrinsic_returns = flatten(extrinsic_returns.to(self.device))
        flat_intrinsic_returns = flatten(intrinsic_returns.to(self.device))
        for start in range(0, total, minibatch_size):
            selected = indices[start : start + minibatch_size]
            yield RolloutBatch(
                observations=batch_tensors["observations"][selected],
                actions=batch_tensors["actions"][selected],
                old_log_probabilities=batch_tensors["old_log_probabilities"][selected],
                extrinsic_rewards=batch_tensors["extrinsic_rewards"][selected],
                intrinsic_rewards=batch_tensors["intrinsic_rewards"][selected],
                terminated=batch_tensors["terminated"][selected],
                truncated=batch_tensors["truncated"][selected],
                extrinsic_values=batch_tensors["extrinsic_values"][selected],
                intrinsic_values=batch_tensors["intrinsic_values"][selected],
                advantages=flat_advantages[selected],
                extrinsic_advantages=flat_extrinsic_advantages[selected],
                intrinsic_advantages=flat_intrinsic_advantages[selected],
                extrinsic_returns=flat_extrinsic_returns[selected],
                intrinsic_returns=flat_intrinsic_returns[selected],
            )
