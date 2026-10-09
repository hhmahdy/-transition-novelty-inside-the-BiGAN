"""Simple from-scratch vectorized environment adapter.

This module manages independent environment instances without relying on an
external vectorized RL implementation. It is intended for the paper's Atari
protocol with 96 parallel environments and a 128-step rollout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

from environments.adapter import EnvironmentStep, SingleEnvironmentAdapter


@dataclass(frozen=True)
class VectorEnvironmentStep:
    """Batched result of one transition in every environment."""

    observations: Tensor
    rewards: Tensor
    terminated: Tensor
    truncated: Tensor
    infos: List[Dict[str, Any]]

    @property
    def done(self) -> Tensor:
        """Return one episode-boundary flag per environment.

        Input: This vector transition.
        Output: Boolean tensor ``terminated | truncated`` of shape ``[N]``.
        Mathematical meaning: Identifies which MDP trajectories require a
            reset before their next action.
        """
        return self.terminated | self.truncated


class VectorEnvironmentAdapter:
    """Batch independent environment adapters for PPO rollouts."""

    def __init__(self, environments: List[SingleEnvironmentAdapter]) -> None:
        """Initialize a vector adapter from non-empty compatible environments.

        Input: List of single-environment adapters.
        Output: Vector adapter with ``num_envs=len(environments)``.
        Mathematical meaning: Constructs the product transition process
            required for batched PPO sampling.
        """
        if not environments:
            raise ValueError("environments must not be empty")
        first_shape = environments[0].observation_shape
        if any(environment.observation_shape != first_shape for environment in environments):
            raise ValueError("all environments must have the same flattened observation shape")
        self.environments = environments
        self.num_envs = len(environments)
        self.device = environments[0].device
        self.observation_dtype = environments[0].observation_dtype
        self._needs_reset = [True] * self.num_envs

    @property
    def observation_shape(self) -> tuple:
        """Return the shared unbatched observation shape.

        Input: This vector adapter.
        Output: Tuple describing one environment observation.
        Mathematical meaning: Defines the per-environment state domain.
        """
        return self.environments[0].observation_shape

    def seed(self, seed: int) -> None:
        """Seed every child environment with a deterministic derived seed.

        Input: Non-negative base seed.
        Output: No value; child environment RNGs and action spaces are seeded.
        Mathematical meaning: Creates reproducible independent transition
            streams in the product MDP.
        """
        if seed < 0:
            raise ValueError("seed must be non-negative")
        for index, environment in enumerate(self.environments):
            environment.seed(seed + index)
        self._needs_reset = [True] * self.num_envs

    def reset(self, seeds: Optional[List[int]] = None) -> Tuple[Tensor, List[Dict[str, Any]]]:
        """Reset all environments and return a stacked initial-state batch.

        Input: Optional list of one seed per environment.
        Output: ``(observations, infos)`` with observations shaped ``[N,...]``.
        Mathematical meaning: Samples the product MDP's initial state batch.
        """
        if seeds is not None and len(seeds) != self.num_envs:
            raise ValueError("seeds must have one entry per environment")
        observations: List[Tensor] = []
        infos: List[Dict[str, Any]] = []
        for index, environment in enumerate(self.environments):
            seed = seeds[index] if seeds is not None else None
            observation, info = environment.reset(seed=seed)
            observations.append(observation)
            infos.append(info)
            self._needs_reset[index] = False
        return torch.stack(observations, dim=0).to(self.device), infos

    def reset_one(self, index: int, seed: Optional[int] = None) -> Tensor:
        """Reset one child environment and return its initial observation.

        Input: Valid child index and optional seed.
        Output: Reset observation tensor for that child.
        Mathematical meaning: Starts one new trajectory in the product MDP.
        """
        if not 0 <= index < self.num_envs:
            raise IndexError("environment index is out of range")
        observation, _ = self.environments[index].reset(seed=seed)
        self._needs_reset[index] = False
        return observation.to(self.device)

    def reset_done(self, seeds: Optional[List[int]] = None) -> Tensor:
        """Reset only environments that ended and return their new observations.

        Input: Optional list of seeds for all environments; entries for active
            environments are ignored.
        Output: Tensor shaped ``[N,...]`` containing reset observations for done
            environments and zeros elsewhere.
        Mathematical meaning: Starts fresh trajectories in completed product-MDP
            components while preserving active trajectories.
        """
        if seeds is not None and len(seeds) != self.num_envs:
            raise ValueError("seeds must have one entry per environment")
        observations: List[Tensor] = []
        for index, environment in enumerate(self.environments):
            if self._needs_reset[index]:
                seed = seeds[index] if seeds is not None else None
                observation, _ = environment.reset(seed=seed)
                self._needs_reset[index] = False
            else:
                observation = torch.zeros(
                    self.observation_shape,
                    dtype=self.observation_dtype,
                    device=self.device,
                )
            observations.append(observation)
        return torch.stack(observations, dim=0).to(self.device)

    def step(self, actions: Tensor) -> VectorEnvironmentStep:
        """Step all environments using one action per environment.

        Input: Actions shaped ``[N]`` for discrete spaces or ``[N,A]`` for
            continuous spaces.
        Output: Batched next observations, rewards, masks, and info records.
        Mathematical meaning: Executes N independent transitions in the
            product MDP under the batched policy action.
        """
        if actions.shape[0] != self.num_envs:
            raise ValueError("actions must have one entry per environment")
        observations: List[Tensor] = []
        rewards: List[float] = []
        terminated: List[bool] = []
        truncated: List[bool] = []
        infos: List[Dict[str, Any]] = []
        for index, environment in enumerate(self.environments):
            if self._needs_reset[index]:
                raise RuntimeError(f"environment {index} must be reset before stepping")
            action = actions[index]
            result: EnvironmentStep = environment.step(action)
            observations.append(result.observation)
            rewards.append(result.reward)
            terminated.append(result.terminated)
            truncated.append(result.truncated)
            infos.append(result.info)
            self._needs_reset[index] = result.done
        return VectorEnvironmentStep(
            observations=torch.stack(observations, dim=0).to(self.device),
            rewards=torch.tensor(rewards, dtype=torch.float32, device=self.device),
            terminated=torch.tensor(terminated, dtype=torch.bool, device=self.device),
            truncated=torch.tensor(truncated, dtype=torch.bool, device=self.device),
            infos=infos,
        )

    def supports_state_restore(self) -> bool:
        """Return whether every environment supports simulator restoration.

        Input: This vector adapter.
        Output: ``True`` only if every child supports clone and restore.
        Mathematical meaning: Algorithm 2 requires resettable semantics for
            every parallel trajectory component.
        """
        return all(environment.supports_state_restore() for environment in self.environments)

    def clone_states(self) -> List[Any]:
        """Capture one simulator snapshot per environment.

        Input: This vector adapter.
        Output: List of N opaque simulator snapshots.
        Mathematical meaning: Stores the product-MDP state needed for episodic
            resettable exploration.
        """
        return [environment.clone_state() for environment in self.environments]

    def restore_state(self, index: int, snapshot: Any) -> Tensor:
        """Restore one child environment and return its observation.

        Input: Environment index and snapshot from that environment.
        Output: Restored observation tensor for the selected environment.
        Mathematical meaning: Restarts one trajectory from a sampled novel state.
        """
        if not 0 <= index < self.num_envs:
            raise IndexError("environment index is out of range")
        self._needs_reset[index] = False
        return self.environments[index].restore_state(snapshot)

    def close(self) -> None:
        """Close all child environments.

        Input: This vector adapter.
        Output: No value; all environment resources are released.
        Mathematical meaning: Ends all product-MDP interactions.
        """
        for environment in self.environments:
            environment.close()
