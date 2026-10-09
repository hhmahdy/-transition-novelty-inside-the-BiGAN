"""Joint discriminator for Bidirectional GAN training.

The discriminator estimates whether a joint pair comes from the encoded real
joint distribution ``(x, E(x))`` or the generated joint distribution
``(G(z), z)``. It returns logits, leaving sigmoid application to the stable
binary-cross-entropy loss implementation.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class BiGANDiscriminator(nn.Module):
    """Discriminate observation-latent joint pairs.

    Args:
        observation_shape: Shape of one observation excluding batch dimension.
        latent_dim: Dimension of the latent vector paired with observations.
        feature_dim: Width of observation and joint representations.
        hidden_dim: Width of intermediate MLP layers.
        image_input_scale: Divisor for image values above one.
    """

    def __init__(
        self,
        observation_shape: tuple[int, ...],
        latent_dim: int = 128,
        feature_dim: int = 256,
        hidden_dim: int = 512,
        image_input_scale: float = 255.0,
    ) -> None:
        """Initialize the joint discriminator.

        Input: Observation shape, latent dimension, feature width, hidden
            width, and image preprocessing scale.
        Output: An initialized discriminator representing :math:`D_omega(x,z)`.
        Mathematical meaning: Defines a classifier over the product space of
            observations and latent codes.
        """
        super().__init__()
        if not observation_shape or any(d <= 0 for d in observation_shape):
            raise ValueError("observation_shape must contain positive dimensions")
        if min(latent_dim, feature_dim, hidden_dim) <= 0:
            raise ValueError("latent_dim, feature_dim, and hidden_dim must be positive")
        if image_input_scale <= 0.0:
            raise ValueError("image_input_scale must be positive")

        self.observation_shape = observation_shape
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.image_input_scale = image_input_scale
        self._is_image = len(observation_shape) in (2, 3)

        if not self._is_image:
            if len(observation_shape) != 1:
                raise ValueError("observation_shape must be a vector, matrix, or image")
            self.observation_encoder = nn.Sequential(
                nn.Linear(observation_shape[0], hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden_dim, feature_dim),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            channels, height, width = self._image_dimensions(observation_shape)
            self.observation_encoder = nn.Sequential(
                nn.Conv2d(channels, 64, 8, 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 3, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Flatten(),
            )
            with torch.no_grad():
                flattened_dim = int(
                    self.observation_encoder(
                        torch.zeros(1, channels, height, width)
                    ).shape[-1]
                )
            self.observation_encoder.add_module(
                "projection",
                nn.Sequential(
                    nn.Linear(flattened_dim, feature_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
            )

        self.latent_encoder = nn.Sequential(
            nn.Linear(latent_dim, feature_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.joint_encoder = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, feature_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.logit_head = nn.Linear(feature_dim, 1)
        self._initialize_weights()

    @staticmethod
    def _image_dimensions(observation_shape: tuple[int, ...]) -> tuple[int, int, int]:
        """Convert grayscale, CHW, or HWC shapes to convolutional CHW.

        Input: Two-dimensional ``(height, width)`` or three-dimensional image
            shape.
        Output: ``(channels, height, width)``.
        Mathematical meaning: Specifies the observation coordinates processed
            by the discriminator's visual branch.
        """
        if len(observation_shape) == 2:
            return 1, observation_shape[0], observation_shape[1]
        first, second, third = observation_shape
        if first <= 4 and second > 4 and third > 4:
            return first, second, third
        return third, first, second

    def _initialize_weights(self) -> None:
        """Initialize discriminator layers with orthogonal parameters.

        Input: Modules owned by this discriminator.
        Output: No value; weights and biases are modified in place.
        Mathematical meaning: Stabilizes the initial joint decision boundary
            in the adversarial minimax game.
        """
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.logit_head.weight, gain=1.0)
        nn.init.zeros_(self.logit_head.bias)

    def _prepare_observation(self, observation: Tensor) -> Tensor:
        """Batch, validate, and normalize observations for discrimination.

        Input: One observation or a batch in the declared layout.
        Output: Floating-point batched tensor in channel-first image layout or
            vector layout.
        Mathematical meaning: Converts the observation component of a joint
            sample into the input representation of :math:`D_omega`.
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
                    raise ValueError("CHW observation has the wrong channel count")
            else:
                if observation.shape[-1] != self.observation_shape[-1]:
                    raise ValueError("HWC observation has the wrong channel count")
                observation = observation.permute(0, 3, 1, 2).contiguous()
            if observation.max().detach().item() > 1.0:
                observation = observation / self.image_input_scale
        elif observation.shape[-1] != self.observation_shape[0]:
            raise ValueError("vector observation has the wrong feature dimension")
        return observation

    def forward_with_features(
        self,
        observation: Tensor,
        latent: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return discriminator logits and joint features for paired inputs.

        Input: Observations ``x`` and latent vectors ``z`` with matching batch
            dimensions.
        Output: ``(logits, features)`` where logits have shape ``[batch]`` and
            features have shape ``[batch, feature_dim]``.
        Mathematical meaning: Computes the joint representation
            ``h_omega(x,z)`` and scalar logit used to estimate whether the pair
            belongs to the encoded or generated joint distribution.
        """
        prepared_observation = self._prepare_observation(observation)
        if latent.ndim == 1:
            latent = latent.unsqueeze(0)
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError(f"latent must have shape [batch, {self.latent_dim}]")
        if prepared_observation.shape[0] != latent.shape[0]:
            raise ValueError("observation and latent batch sizes must match")
        latent = latent.float().to(device=prepared_observation.device)
        observation_features = self.observation_encoder(prepared_observation)
        latent_features = self.latent_encoder(latent)
        joint_features = self.joint_encoder(
            torch.cat((observation_features, latent_features), dim=-1)
        )
        logits = self.logit_head(joint_features).squeeze(-1)
        return logits, joint_features

    def forward(self, observation: Tensor, latent: Tensor) -> Tensor:
        """Return discriminator logits for observation-latent pairs.

        Input: Observations ``x`` and matching latent vectors ``z``.
        Output: Tensor of shape ``[batch]`` containing unnormalized logits.
        Mathematical meaning: Computes the scalar score
            :math:`D_omega(x,z)` before sigmoid conversion.
        """
        logits, _ = self.forward_with_features(observation, latent)
        return logits
