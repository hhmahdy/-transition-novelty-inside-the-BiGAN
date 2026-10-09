"""Replay storage for BiGAN observation training.

BiGAN is trained from a distribution of observations rather than PPO's single
short on-policy rollout. This module stores recent observations in a fixed
circular buffer and samples them uniformly for adversarial updates.

Only observations are stored deliberately. Actions, values, and policy
statistics belong to PPO and must not influence the BiGAN data distribution.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ReplayBatch:
    """A sampled minibatch of observations for BiGAN training."""

    observations: Tensor

    def __len__(self) -> int:
        """Return the number of observations in this sampled batch.

        Input: This ``ReplayBatch`` instance.
        Output: Number of rows in ``observations``.
        Mathematical meaning: Gives the empirical batch size used to estimate
            the BiGAN discriminator and generator/encoder objectives.
        """
        return int(self.observations.shape[0])


class ReplayBuffer:
    """Fixed-capacity circular storage for BiGAN observations.

    Args:
        capacity: Maximum number of observations retained.
        observation_shape: Shape of one observation excluding batch dimension.
        storage_device: Device on which replay storage is allocated. CPU is
            generally preferred when observations are collected on GPU only
            occasionally, while a CUDA buffer can reduce transfer overhead.
        observation_dtype: Storage dtype, commonly ``torch.uint8`` for raw
            pixels or ``torch.float32`` for preprocessed observations.
    """

    def __init__(
        self,
        capacity: int,
        observation_shape: tuple[int, ...],
        storage_device: Union[torch.device, str] = "cpu",
        observation_dtype: torch.dtype = torch.float32,
    ) -> None:
        """Allocate an empty circular replay buffer.

        Input: Capacity, observation shape, storage device, and storage dtype.
        Output: An empty ``ReplayBuffer`` with preallocated observation
            storage.
        Mathematical meaning: Defines a finite empirical distribution that
            approximates the observation marginal used by BiGAN optimization.
        """
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not observation_shape or any(d <= 0 for d in observation_shape):
            raise ValueError("observation_shape must contain positive dimensions")
        if not (observation_dtype.is_floating_point or observation_dtype == torch.uint8):
            raise TypeError("observation_dtype must be floating point or uint8")
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.storage_device = torch.device(storage_device)
        self.observation_dtype = observation_dtype
        self.observations = torch.empty(
            (capacity, *observation_shape),
            dtype=observation_dtype,
            device=self.storage_device,
        )
        self._next_index = 0
        self._size = 0

    def __len__(self) -> int:
        """Return the number of valid observations currently stored.

        Input: This replay buffer.
        Output: Integer in ``[0, capacity]``.
        Mathematical meaning: Gives the support size of the current empirical
            observation distribution.
        """
        return self._size

    @property
    def is_full(self) -> bool:
        """Return whether the circular buffer has reached capacity.

        Input: This replay buffer.
        Output: Boolean indicating whether old observations are being replaced.
        Mathematical meaning: Once full, the empirical BiGAN distribution uses
            a fixed-size sliding window of recent observations.
        """
        return self._size == self.capacity

    def clear(self) -> None:
        """Discard all stored observations without reallocating storage.

        Input: This replay buffer.
        Output: No value; logical size and insertion position are reset.
        Mathematical meaning: Starts a new empirical observation distribution
            while preserving the configured memory allocation.
        """
        self._next_index = 0
        self._size = 0

    def add(self, observation: Tensor) -> None:
        """Append one observation, overwriting the oldest item when full.

        Input: One tensor shaped exactly ``observation_shape``.
        Output: No value; the observation is copied into replay storage.
        Mathematical meaning: Adds one sample to the finite observation
            marginal from which BiGAN minibatches are drawn.
        """
        observation = observation.detach().to(
            device=self.storage_device,
            dtype=self.observation_dtype,
        )
        if tuple(observation.shape) != self.observation_shape:
            raise ValueError(
                f"observation must have shape {self.observation_shape}, got {tuple(observation.shape)}"
            )
        self.observations[self._next_index].copy_(observation)
        self._next_index = (self._next_index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_batch(self, observations: Tensor) -> None:
        """Append a batch of observations in collection order.

        Input: Tensor shaped ``[batch, *observation_shape]``.
        Output: No value; every row is inserted, with oldest rows overwritten
            first when the batch exceeds remaining capacity.
        Mathematical meaning: Adds a batch of samples to the empirical
            observation marginal used for adversarial training.
        """
        observations = observations.detach()
        if observations.ndim != len(self.observation_shape) + 1:
            raise ValueError("observations must include a leading batch dimension")
        if tuple(observations.shape[1:]) != self.observation_shape:
            raise ValueError(
                f"observations must have shape [batch, {self.observation_shape}]"
            )
        for observation in observations:
            self.add(observation)

    def sample(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> ReplayBatch:
        """Uniformly sample observations without replacement within a batch.

        Input: Positive batch size no larger than the current buffer size, an
            optional seeded PyTorch generator, and an optional output device.
        Output: ``ReplayBatch`` containing ``batch_size`` observations.
        Mathematical meaning: Draws an empirical Monte Carlo minibatch for the
            expectation in the BiGAN adversarial objective.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self._size < batch_size:
            raise ValueError(
                f"cannot sample {batch_size} observations from buffer of size {self._size}"
            )
        indices = torch.randperm(
            self._size,
            generator=generator,
            device=self.storage_device,
        )[:batch_size]
        sampled = self.observations[: self._size][indices]
        if device is not None:
            sampled = sampled.to(device)
        return ReplayBatch(observations=sampled)

    def state_dict(self) -> Dict[str, Union[Tensor, int]]:
        """Return a detached checkpoint representation of replay state.

        Input: This replay buffer.
        Output: Dictionary containing valid observations, insertion position,
            and logical size.
        Mathematical meaning: Captures the empirical observation distribution
            so a resumed BiGAN run can continue from the same replay state.
        """
        return {
            "observations": self.observations[: self._size].detach().clone(),
            "next_index": self._next_index,
            "size": self._size,
        }

    def load_state_dict(self, state: Dict[str, Union[Tensor, int]]) -> None:
        """Restore replay contents from a compatible checkpoint dictionary.

        Input: State produced by ``state_dict`` with matching observation shape
            and size within the configured capacity.
        Output: No value; replay storage and circular indices are restored.
        Mathematical meaning: Reconstructs the same finite observation
            distribution used by BiGAN before checkpointing.
        """
        observations = state.get("observations")
        next_index = state.get("next_index")
        size = state.get("size")
        if not isinstance(observations, Tensor) or not isinstance(next_index, int) or not isinstance(size, int):
            raise TypeError("invalid replay state types")
        if size < 0 or size > self.capacity or next_index < 0 or next_index >= self.capacity:
            raise ValueError("invalid replay state indices")
        if tuple(observations.shape) != (size, *self.observation_shape):
            raise ValueError("replay state has an incompatible observation shape")
        self.observations[:size].copy_(
            observations.to(device=self.storage_device, dtype=self.observation_dtype)
        )
        self._size = size
        self._next_index = next_index
