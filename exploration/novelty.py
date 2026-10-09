"""BiGAN-based novelty estimation.

Novelty is computed from the discrepancy between an observation and its BiGAN
reconstruction, both in pixel space and in the discriminator's learned joint
feature space. All methods return one score per observation and run without
gradients because novelty is an RL signal, not a BiGAN optimization path.
"""

from __future__ import annotations

from typing import Optional, Union

from dataclasses import dataclass

import torch
from torch import Tensor

from bigan.discriminator import BiGANDiscriminator
from bigan.encoder import BiGANEncoder
from bigan.generator import BiGANGenerator
from utils.normalization import RunningMeanVariance, RunningNormalizer


@dataclass(frozen=True)
class NoveltyComponents:
    """Per-observation pixel, feature, combined, and normalized novelty scores."""

    pixel_error: Tensor
    feature_error: Tensor
    combined_score: Tensor
    normalized_score: Tensor


class PixelReconstructionError:
    """Compute per-observation pixel reconstruction mean-squared error."""

    def __call__(self, observations: Tensor, reconstructions: Tensor) -> Tensor:
        """Return one normalized pixel error for each observation.

        Input: Observation and reconstruction tensors with identical shape
            ``[batch, *observation_shape]``.
        Output: Tensor ``[batch]`` containing mean squared pixel errors.
        Mathematical meaning: Computes the Eq. 4 pixel novelty term
            ``L_G(x)=||x-G(E(x))||_1`` as a per-sample sum of absolute
            differences over all observation coordinates.
        """
        if observations.shape != reconstructions.shape:
            raise ValueError("observations and reconstructions must have identical shapes")
        if observations.ndim < 2:
            raise ValueError("observations must include a batch dimension")
        return (observations.float() - reconstructions.float()).abs().flatten(1).sum(dim=1)


class FeatureMatchingError:
    """Compute discrepancy between real and reconstructed joint features."""

    def __init__(
        self,
        encoder: BiGANEncoder,
        discriminator: BiGANDiscriminator,
    ) -> None:
        """Store the frozen feature-producing networks.

        Input: BiGAN encoder and joint discriminator.
        Output: A feature-matching error component.
        Mathematical meaning: Defines the representation map used in
            ``e_feature(x)`` without changing network parameters.
        """
        self.encoder = encoder
        self.discriminator = discriminator

    @torch.no_grad()
    def __call__(self, observations: Tensor, reconstructions: Tensor) -> Tensor:
        """Return one joint-feature mismatch for each observation.

        Input: Real observations ``x`` and reconstructions ``x_hat`` with
            identical batch and observation shapes.
        Output: Tensor ``[batch]`` containing mean squared joint-feature error.
        Mathematical meaning: Computes the Eq. 4 discriminator-feature term
            ``L_D(x)=||f_D(x,E(x))-f_D(G(E(x)),E(x))||_1`` as a per-sample
            sum of absolute feature differences.
        """
        if observations.shape != reconstructions.shape:
            raise ValueError("observations and reconstructions must have identical shapes")
        latents = self.encoder(observations)
        _, real_features = self.discriminator.forward_with_features(observations, latents)
        _, reconstructed_features = self.discriminator.forward_with_features(
            reconstructions,
            latents,
        )
        return (real_features - reconstructed_features).abs().sum(dim=1)


class CombinedNoveltyScore:
    """Compute the paper's convex pixel-feature novelty combination."""

    def __init__(self, alpha: float = 0.9) -> None:
        """Initialize Eq. (4)'s convex-mixture parameter.

        Input: ``alpha`` in ``[0,1]``.
        Output: A callable combined-score component.
        Mathematical meaning: Defines
            ``B(s)=alpha*L_G(s)+(1-alpha)*L_D(s)``.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must satisfy 0 <= alpha <= 1")
        self.alpha = alpha

    def __call__(self, pixel_error: Tensor, feature_error: Tensor) -> Tensor:
        """Return the weighted combined novelty score.

        Input: Per-observation pixel and feature errors with identical shapes.
        Output: Tensor of the same shape containing combined novelty.
        Mathematical meaning: Computes Eq. (4)
            ``B(s)=alpha*L_G(s)+(1-alpha)*L_D(s)``.
        """
        if pixel_error.shape != feature_error.shape:
            raise ValueError("pixel and feature errors must have identical shapes")
        return self.alpha * pixel_error + (1.0 - self.alpha) * feature_error


class NormalizedNoveltyScore:
    """Maintain running novelty statistics and standardize combined scores."""

    def __init__(
        self,
        epsilon: float = 1.0e-8,
        clip_range: Optional[float] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        """Initialize the running novelty normalizer.

        Input: Positive epsilon, optional positive clipping bound, and stats
            device.
        Output: A callable online normalizer.
        Mathematical meaning: Defines ``(n(x)-mu)/sqrt(var+epsilon)``.
        """
        self.normalizer = RunningNormalizer(
            shape=(),
            epsilon=epsilon,
            clip_range=clip_range,
            device=device,
        )
        self.extrinsic_reward_statistics = RunningMeanVariance(
            shape=(),
            epsilon=epsilon,
            device=device,
        )

    @property
    def mean_extrinsic_reward(self) -> Tensor:
        """Return the running mean ``mu(r^e)`` used by Eq. (5).

        Input: This normalized novelty component.
        Output: Scalar running extrinsic-reward mean tensor.
        Mathematical meaning: Supplies the reward-scale offset in
            ``(B-mu(B)+mu(r^e))/sigma(B)``.
        """
        return self.extrinsic_reward_statistics.mean

    def update_extrinsic_reward(self, extrinsic_reward: Tensor) -> None:
        """Update the running extrinsic-reward mean.

        Input: Extrinsic reward batch, usually ``[batch]``.
        Output: No value; updates the running estimate of ``mu(r^e)``.
        Mathematical meaning: Tracks the reward scale required by Eq. (5).
        """
        self.extrinsic_reward_statistics.update(extrinsic_reward.reshape(-1))

    def __call__(
        self,
        scores: Tensor,
        update: bool = True,
        extrinsic_reward: Optional[Tensor] = None,
    ) -> Tensor:
        """Normalize novelty scores with optional running-statistics update.

        Input: Score tensor ``[batch]`` or scalar, update flag, and optional
            extrinsic reward batch used to update ``mu(r^e)``.
        Output: Eq. (5)-normalized score tensor with the same shape.
        Mathematical meaning: Computes
            ``(B(s)-mu(B)+mu(r^e))/sigma(B)``.
        """
        original_shape = scores.shape
        if extrinsic_reward is not None:
            self.update_extrinsic_reward(extrinsic_reward)
        normalized = self.normalizer.normalize(
            scores.reshape(-1),
            update=update,
            mean_offset=self.mean_extrinsic_reward,
        )
        return normalized.reshape(original_shape)


class NoveltyEstimator:
    """End-to-end BiGAN novelty estimator with independent components."""

    def __init__(
        self,
        encoder: BiGANEncoder,
        generator: BiGANGenerator,
        discriminator: BiGANDiscriminator,
        alpha: float = 0.9,
        normalization_epsilon: float = 1.0e-8,
        normalization_clip: Optional[float] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        """Construct the complete pixel-feature-normalization pipeline.

        Input: BiGAN networks, novelty coefficients, normalization settings,
            and device.
        Output: A reusable estimator returning structured novelty components.
        Mathematical meaning: Composes ``E``, ``G``, pixel error, feature
            error, weighted novelty, and online normalization.
        """
        self.encoder = encoder.to(device)
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.pixel_error = PixelReconstructionError()
        self.feature_error = FeatureMatchingError(self.encoder, self.discriminator)
        self.combined_score = CombinedNoveltyScore(alpha)
        self.normalized_score = NormalizedNoveltyScore(
            epsilon=normalization_epsilon,
            clip_range=normalization_clip,
            device=device,
        )
        self.device = torch.device(device)

    @torch.no_grad()
    def __call__(
        self,
        observations: Tensor,
        update_statistics: bool = True,
        extrinsic_reward: Optional[Tensor] = None,
    ) -> NoveltyComponents:
        """Estimate novelty and apply Eq. (5) normalization.

        Input: Observations shaped ``[batch, *observation_shape]``, statistics
            update flag, and optional extrinsic reward for ``mu(r^e)``.
        Output: ``NoveltyComponents`` containing pixel, feature, combined, and
            Eq. (5)-normalized scores.
        Mathematical meaning: Computes ``z=E(x)``, ``x_hat=G(z)``, the Eq. 4
            novelty score ``B(x)``, and then Eq. 5 normalization.
        """
        observations = observations.to(self.device).float()
        if (
            len(self.encoder.observation_shape) in (2, 3)
            and observations.numel() > 0
            and observations.detach().amax().item() > 1.0
        ):
            observations = observations / 255.0
        latents = self.encoder(observations)
        reconstructions = self.generator(latents)
        pixel_error = self.pixel_error(observations, reconstructions)
        feature_error = self.feature_error(observations, reconstructions)
        combined_score = self.combined_score(pixel_error, feature_error)
        normalized_score = self.normalized_score(
            combined_score,
            update=update_statistics,
            extrinsic_reward=extrinsic_reward,
        )
        return NoveltyComponents(
            pixel_error=pixel_error,
            feature_error=feature_error,
            combined_score=combined_score,
            normalized_score=normalized_score,
        )
