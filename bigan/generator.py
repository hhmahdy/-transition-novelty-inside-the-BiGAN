"""BiGAN latent-to-observation generator.

The generator implements the mapping

.. math:: tilde{x} = G_theta(z),

which forms generated joint samples ``(G(z), z)`` for the BiGAN discriminator.
Image observations are generated in CHW internally and converted back to the
declared observation layout at the public interface.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BiGANGenerator(nn.Module):
    """Generate observations from BiGAN latent vectors.

    Args:
        observation_shape: Shape of one generated observation.
        latent_dim: Dimension of the input latent vector.
        feature_dim: Width of the vector generator trunk or initial image
            feature map.
        hidden_dim: Width of the vector generator hidden layers.
        output_activation: ``"sigmoid"`` for ``[0,1]`` pixels, ``"tanh"`` for
            ``[-1,1]`` pixels, or ``"identity"`` for unconstrained output.
    """

    def __init__(
        self,
        observation_shape: tuple[int, ...],
        latent_dim: int = 128,
        feature_dim: int = 256,
        hidden_dim: int = 512,
        output_activation: str = "sigmoid",
    ) -> None:
        """Initialize the generator architecture.

        Input: Observation shape, latent dimension, architecture widths, and
            output activation choice.
        Output: An initialized ``BiGANGenerator`` representing :math:`G_theta`.
        Mathematical meaning: Defines the differentiable inverse-style map from
            the BiGAN latent prior to observation space.
        """
        super().__init__()
        if not observation_shape or any(d <= 0 for d in observation_shape):
            raise ValueError("observation_shape must contain positive dimensions")
        if min(latent_dim, feature_dim, hidden_dim) <= 0:
            raise ValueError("latent_dim, feature_dim, and hidden_dim must be positive")
        if output_activation not in {"sigmoid", "tanh", "identity"}:
            raise ValueError("output_activation must be sigmoid, tanh, or identity")

        self.observation_shape = observation_shape
        self.latent_dim = latent_dim
        self.output_activation = output_activation
        self._is_image = len(observation_shape) in (2, 3)

        if not self._is_image:
            if len(observation_shape) != 1:
                raise ValueError("observation_shape must be a vector, matrix, or image")
            output_dim = observation_shape[0]
            self.network = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            channels, height, width = self._image_dimensions(observation_shape)
            self._image_channels = channels
            self._image_height = height
            self._image_width = width
            base_channels = max(64, feature_dim // 2)
            self.image_projection = nn.Sequential(
                nn.Linear(latent_dim, base_channels * 4 * 4),
                nn.ReLU(inplace=True),
            )
            self.image_network = nn.Sequential(
                nn.ConvTranspose2d(base_channels, base_channels // 2, 4, 2, 1),
                nn.BatchNorm2d(base_channels // 2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(base_channels // 2, base_channels // 4, 4, 2, 1),
                nn.BatchNorm2d(base_channels // 4),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(base_channels // 4, base_channels // 8, 4, 2, 1),
                nn.BatchNorm2d(base_channels // 8),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels // 8, channels, kernel_size=3, padding=1),
            )
        self._initialize_weights()

    @staticmethod
    def _image_dimensions(observation_shape: tuple[int, ...]) -> tuple[int, int, int]:
        """Convert grayscale, CHW, or HWC declarations into CHW dimensions.

        Input: Two-dimensional ``(height, width)`` or three-dimensional CHW/HWC
            observation shape.
        Output: ``(channels, height, width)`` tuple for image generation.
        Mathematical meaning: Defines the image domain of :math:`G_theta(z)`.
        """
        if len(observation_shape) == 2:
            return 1, observation_shape[0], observation_shape[1]
        first, second, third = observation_shape
        if first <= 4 and second > 4 and third > 4:
            return first, second, third
        return third, first, second

    def _initialize_weights(self) -> None:
        """Initialize generator layers using orthogonal parameters.

        Input: Modules owned by this generator.
        Output: No value; weights and biases are modified in place.
        Mathematical meaning: Stabilizes the initial Jacobian of the
            latent-to-observation mapping in the adversarial game.
        """
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _activate_output(self, output: Tensor) -> Tensor:
        """Apply the configured range transformation to generated values.

        Input: Raw generator output tensor.
        Output: Tensor transformed by sigmoid, tanh, or identity.
        Mathematical meaning: Constrains :math:`G_theta(z)` to the numerical
            observation domain expected by the encoder and discriminator.
        """
        if self.output_activation == "sigmoid":
            return torch.sigmoid(output)
        if self.output_activation == "tanh":
            return torch.tanh(output)
        return output

    def _prepare_latent(self, latent: Tensor) -> Tensor:
        """Validate and batch latent vectors.

        Input: One latent vector or a batch shaped ``[batch, latent_dim]``.
        Output: Floating-point tensor with explicit batch dimension.
        Mathematical meaning: Provides valid samples from the latent domain of
            the generator mapping :math:`G_theta(z)`.
        """
        if latent.ndim == 1:
            latent = latent.unsqueeze(0)
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"latent must have shape [batch, {self.latent_dim}]"
            )
        return latent.float()

    def forward(self, latent: Tensor) -> Tensor:
        """Generate observations from latent vectors.

        Input: Tensor shaped ``[batch, latent_dim]`` or one latent vector.
        Output: Tensor shaped ``[batch, *observation_shape]`` in the declared
            vector, grayscale, CHW, or HWC layout.
        Mathematical meaning: Computes :math:`tilde{x}=G_theta(z)` for the
            generated half of the BiGAN joint distribution.
        """
        latent = self._prepare_latent(latent)
        if not self._is_image:
            return self._activate_output(self.network(latent))

        batch_size = latent.shape[0]
        projected = self.image_projection(latent).reshape(
            batch_size,
            -1,
            4,
            4,
        )
        generated = self.image_network(projected)
        generated = F.interpolate(
            generated,
            size=(self._image_height, self._image_width),
            mode="bilinear",
            align_corners=False,
        )
        generated = self._activate_output(generated)
        if len(self.observation_shape) == 2:
            return generated[:, 0]
        if self.observation_shape[0] <= 4 and self.observation_shape[1] > 4:
            return generated
        return generated.permute(0, 2, 3, 1).contiguous()
