"""PPO value-function approximators.

The module provides a reusable single state-value network and an explicit dual
critic for Adventurer's separate extrinsic/intrinsic advantage streams:

.. math:: V^e(s), V^i(s).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CriticNetwork(nn.Module):
    """Estimate one scalar state value for every input observation."""

    def __init__(
        self,
        observation_shape: tuple[int, ...],
        hidden_dim: int = 512,
        image_input_scale: float = 255.0,
    ) -> None:
        """Initialize the single-stream value network.

        Input: Observation shape, hidden width, and image scaling divisor.
        Output: Initialized network approximating one value function.
        Mathematical meaning: Defines a differentiable approximation to
            ``V(s)`` for one reward stream.
        """
        super().__init__()
        if not observation_shape or any(d <= 0 for d in observation_shape):
            raise ValueError("observation_shape must contain positive dimensions")
        if hidden_dim <= 0 or image_input_scale <= 0.0:
            raise ValueError("hidden_dim and image_input_scale must be positive")
        self.observation_shape = observation_shape
        self.image_input_scale = image_input_scale
        self._is_image = len(observation_shape) in (2, 3)
        if not self._is_image:
            if len(observation_shape) != 1:
                raise ValueError("observation_shape must be a vector, matrix, or image")
            self.encoder = nn.Sequential(
                nn.Linear(observation_shape[0], hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            encoded_dim = hidden_dim
        else:
            channels, height, width = self._image_dimensions(observation_shape)
            self.encoder = nn.Sequential(
                nn.Conv2d(channels, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                encoded_dim = int(self.encoder(torch.zeros(1, channels, height, width)).shape[-1])
            self.encoder.add_module(
                "projection",
                nn.Sequential(nn.Linear(encoded_dim, hidden_dim), nn.ReLU()),
            )
            encoded_dim = hidden_dim
        self.value_head = nn.Linear(encoded_dim, 1)
        self._initialize_weights()

    @staticmethod
    def _image_dimensions(observation_shape: tuple[int, ...]) -> tuple[int, int, int]:
        """Convert grayscale, CHW, or HWC shape to convolutional CHW.

        Input: Two-dimensional or three-dimensional image shape.
        Output: ``(channels,height,width)``.
        Mathematical meaning: Defines the visual state domain of ``V(s)``.
        """
        if len(observation_shape) == 2:
            return 1, observation_shape[0], observation_shape[1]
        first, second, third = observation_shape
        if first <= 4 and second > 4 and third > 4:
            return first, second, third
        return third, first, second

    def _initialize_weights(self) -> None:
        """Apply orthogonal initialization to value-network layers.

        Input: Modules owned by this critic.
        Output: No value; parameters are initialized in place.
        Mathematical meaning: Stabilizes the initial value approximation.
        """
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.value_head.weight)

    def _prepare_observation(self, observation: Tensor) -> Tensor:
        """Validate, batch, and preprocess observations.

        Input: One observation or a batch in the declared layout.
        Output: Floating-point batched tensor in encoder-compatible layout.
        Mathematical meaning: Converts raw state ``s`` into the input of
            ``V(s)``.
        """
        if observation.ndim == len(self.observation_shape):
            observation = observation.unsqueeze(0)
        if observation.ndim != len(self.observation_shape) + 1:
            raise ValueError("observation has an invalid number of dimensions")
        observation = observation.float()
        if self._is_image:
            if len(self.observation_shape) == 2:
                observation = observation.unsqueeze(1)
            elif self.observation_shape[0] <= 4 and self.observation_shape[1] > 4:
                if observation.shape[1] != self.observation_shape[0]:
                    raise ValueError("CHW image observations have wrong channels")
            else:
                if observation.shape[-1] != self.observation_shape[-1]:
                    raise ValueError("HWC image observations have wrong channels")
                observation = observation.permute(0, 3, 1, 2).contiguous()
            if observation.max().detach().item() > 1.0:
                observation = observation / self.image_input_scale
        elif observation.shape[-1] != self.observation_shape[0]:
            raise ValueError("vector observations have wrong feature dimension")
        return observation

    def forward(self, observation: Tensor) -> Tensor:
        """Compute one scalar value estimate per observation.

        Input: One observation or a batch of observations.
        Output: Tensor shaped ``[batch]`` containing ``V(s)``.
        Mathematical meaning: Estimates the expected discounted return for one
            independently defined reward stream.
        """
        prepared = self._prepare_observation(observation)
        return self.value_head(self.encoder(prepared)).squeeze(-1)


class DualCriticNetwork(nn.Module):
    """Maintain independent extrinsic and intrinsic value functions."""

    def __init__(
        self,
        observation_shape: tuple[int, ...],
        hidden_dim: int = 512,
        image_input_scale: float = 255.0,
    ) -> None:
        """Initialize two independent critic networks.

        Input: Observation shape, hidden width, and image scaling divisor.
        Output: Module containing ``extrinsic_critic`` and ``intrinsic_critic``.
        Mathematical meaning: Defines independent approximators
            ``V^e_phi(s)`` and ``V^i_psi(s)`` required by two-stream GAE.
        """
        super().__init__()
        self.extrinsic_critic = CriticNetwork(observation_shape, hidden_dim, image_input_scale)
        self.intrinsic_critic = CriticNetwork(observation_shape, hidden_dim, image_input_scale)

    def forward(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        """Return extrinsic and intrinsic value estimates.

        Input: One observation or a batch of observations.
        Output: ``(extrinsic_value, intrinsic_value)`` tensors, each shaped
            ``[batch]``.
        Mathematical meaning: Computes ``(V^e(s),V^i(s))`` without combining
            reward semantics inside either critic.
        """
        return self.extrinsic_critic(observation), self.intrinsic_critic(observation)

    def extrinsic_value(self, observation: Tensor) -> Tensor:
        """Evaluate only the extrinsic value function.

        Input: One observation or a batch.
        Output: Tensor ``[batch]`` containing ``V^e(s)``.
        Mathematical meaning: Estimates the discounted extrinsic return.
        """
        return self.extrinsic_critic(observation)

    def intrinsic_value(self, observation: Tensor) -> Tensor:
        """Evaluate only the intrinsic value function.

        Input: One observation or a batch.
        Output: Tensor ``[batch]`` containing ``V^i(s)``.
        Mathematical meaning: Estimates the discounted intrinsic return.
        """
        return self.intrinsic_critic(observation)
