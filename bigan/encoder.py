"""BiGAN observation encoder.

The encoder implements the inference mapping

.. math:: z = E_psi(x),

which forms the encoded half of the real joint distribution ``(x, E(x))`` in
BiGAN training. The intermediate feature vector is exposed because the
joint discriminator and feature-matching novelty estimator may use it.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class BiGANEncoder(nn.Module):
    """Map observations to latent vectors and optionally expose features.

    Args:
        observation_shape: Shape of one observation excluding batch dimension.
        latent_dim: Dimension of the BiGAN latent space.
        feature_dim: Dimension of the encoder's intermediate representation.
        hidden_dim: Width of the vector trunk or visual projection.
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
        """Initialize the encoder and orthogonally initialize its layers.

        Input: Observation shape, latent/feature dimensions, trunk width, and
            image preprocessing scale.
        Output: An initialized encoder representing :math:`E_psi`.
        Mathematical meaning: Defines a differentiable map from observation
            space to the shared BiGAN latent space.
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
            self.backbone = nn.Sequential(
                nn.Linear(observation_shape[0], hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden_dim, feature_dim),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            channels, height, width = self._image_dimensions(observation_shape)
            convolutional_backbone = nn.Sequential(
                nn.Conv2d(channels, 64, kernel_size=8, stride=4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, kernel_size=4, stride=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Flatten(),
            )
            with torch.no_grad():
                flattened_dim = int(
                    convolutional_backbone(
                        torch.zeros(1, channels, height, width)
                    ).shape[-1]
                )
            convolutional_backbone.add_module(
                "feature_projection",
                nn.Sequential(
                    nn.Linear(flattened_dim, feature_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
            )
            self.backbone = convolutional_backbone

        self.latent_head = nn.Linear(feature_dim, latent_dim)
        self._initialize_weights()

    @staticmethod
    def _image_dimensions(observation_shape: tuple[int, ...]) -> tuple[int, int, int]:
        """Convert grayscale, CHW, or HWC declarations to CHW dimensions.

        Input: Two-dimensional ``(height, width)`` or three-dimensional CHW/HWC
            observation shape.
        Output: ``(channels, height, width)`` tuple for convolutional layers.
        Mathematical meaning: Establishes the coordinate domain of the visual
            encoder used to calculate :math:`E_psi(x)`.
        """
        if len(observation_shape) == 2:
            return 1, observation_shape[0], observation_shape[1]
        first, second, third = observation_shape
        if first <= 4 and second > 4 and third > 4:
            return first, second, third
        return third, first, second

    def _initialize_weights(self) -> None:
        """Apply orthogonal initialization to encoder layers.

        Input: Modules owned by this encoder.
        Output: No value; weights and biases are initialized in place.
        Mathematical meaning: Provides stable initial feature scales for the
            adversarial encoder-generator game.
        """
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.latent_head.weight, gain=1.0)
        nn.init.zeros_(self.latent_head.bias)

    def _prepare_observation(self, observation: Tensor) -> Tensor:
        """Batch, validate, and normalize an observation tensor.

        Input: One observation or a batch of observations in the declared
            vector, CHW, or HWC format.
        Output: Floating-point batched tensor in encoder-compatible layout.
        Mathematical meaning: Converts raw ``x`` into the numerical input used
            by the encoder mapping :math:`E_psi(x)`.
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

    def forward_features(self, observation: Tensor) -> Tensor:
        """Return intermediate encoder features before latent projection.

        Input: One observation or a batch of observations.
        Output: Feature tensor of shape ``[batch, feature_dim]``.
        Mathematical meaning: Computes the learned representation
            :math:`h_psi(x)` used for feature matching and joint discrimination.
        """
        return self.backbone(self._prepare_observation(observation))

    def forward(self, observation: Tensor) -> Tensor:
        """Encode observations into BiGAN latent vectors.

        Input: One observation or a batch of observations.
        Output: Latent tensor of shape ``[batch, latent_dim]``.
        Mathematical meaning: Computes :math:`z=E_psi(x)`, the encoded latent
            variable paired with real observations in the BiGAN objective.
        """
        return self.latent_head(self.forward_features(observation))
