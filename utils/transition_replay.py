"""Replay storage for latent transition-model training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import Tensor


@dataclass(frozen=True)
class TransitionBatch:
    """A minibatch of state-action-next-state transitions."""

    observations: Tensor
    actions: Tensor
    next_observations: Tensor


class TransitionReplayBuffer:
    """Fixed-capacity circular replay buffer for transition novelty."""

    def __init__(
        self,
        capacity: int,
        observation_shape: tuple,
        action_shape: tuple = (),
        device: Union[torch.device, str] = "cpu",
        observation_dtype: torch.dtype = torch.float32,
        action_dtype: torch.dtype = torch.long,
    ) -> None:
        """Allocate transition replay storage.

        Input: Capacity, observation/action shapes, device, and dtypes.
        Output: Empty transition replay buffer.
        Mathematical meaning: Defines the empirical transition distribution
            used to fit ``f_phi(z,a)``.
        """
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.observation_shape = tuple(observation_shape)
        self.action_shape = tuple(action_shape)
        self.device = torch.device(device)
        self.observations = torch.empty((capacity, *self.observation_shape), dtype=observation_dtype, device=self.device)
        self.next_observations = torch.empty_like(self.observations)
        self.actions = torch.empty((capacity, *self.action_shape), dtype=action_dtype, device=self.device)
        self._size = 0
        self._next_index = 0

    def __len__(self) -> int:
        """Return the number of valid transitions.

        Input: This buffer.
        Output: Integer in ``[0,capacity]``.
        Mathematical meaning: Gives the empirical transition support size.
        """
        return self._size

    def add_batch(self, observations: Tensor, actions: Tensor, next_observations: Tensor) -> None:
        """Append a batch of transitions in collection order.

        Input: Batched ``s_t``, ``a_t``, and ``s_(t+1)`` tensors.
        Output: No value; transitions are copied into circular storage.
        Mathematical meaning: Adds samples from ``p(s,a,s')`` for forward-model
            maximum-likelihood-style regression.
        """
        if observations.shape[0] != actions.shape[0] or observations.shape[0] != next_observations.shape[0]:
            raise ValueError("transition batch dimensions must match")
        if tuple(observations.shape[1:]) != self.observation_shape or tuple(next_observations.shape[1:]) != self.observation_shape:
            raise ValueError("observation shapes do not match replay configuration")
        if tuple(actions.shape[1:]) != self.action_shape:
            raise ValueError("action shape does not match replay configuration")
        for index in range(observations.shape[0]):
            slot = self._next_index
            self.observations[slot].copy_(observations[index].detach().to(self.device))
            self.actions[slot].copy_(actions[index].detach().to(self.device))
            self.next_observations[slot].copy_(next_observations[index].detach().to(self.device))
            self._next_index = (self._next_index + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, generator: Optional[torch.Generator] = None) -> TransitionBatch:
        """Uniformly sample transitions without replacement within a batch.

        Input: Positive batch size no greater than current buffer size and an
            optional seeded generator.
        Output: ``TransitionBatch`` on the configured device.
        Mathematical meaning: Draws a Monte Carlo minibatch for fitting
            ``f_phi(E(s),a)`` to ``E(s')``.
        """
        if batch_size <= 0 or batch_size > self._size:
            raise ValueError("invalid transition batch size")
        indices = torch.randperm(self._size, generator=generator, device=self.device)[:batch_size]
        return TransitionBatch(
            self.observations[:self._size][indices],
            self.actions[:self._size][indices],
            self.next_observations[:self._size][indices],
        )
