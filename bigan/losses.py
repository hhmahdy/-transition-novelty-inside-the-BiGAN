"""Loss functions for BiGAN adversarial and novelty objectives.

The discriminator receives encoded real pairs ``(x, E(x))`` and generated
pairs ``(G(z), z)``. This module keeps each mathematical term explicit so
training code can report and ablate them independently.
"""

from __future__ import annotations

from typing import Optional

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class BiGANLossValues:
    """Named BiGAN loss components for logging and optimization."""

    discriminator: Tensor
    encoder_generator: Tensor
    pixel_reconstruction: Tensor
    feature_matching: Tensor
    gradient_penalty: Tensor


def discriminator_loss(real_logits: Tensor, fake_logits: Tensor) -> Tensor:
    """Compute stable binary cross-entropy for the joint discriminator.

    Input: Real-pair logits ``D(x,E(x))`` and generated-pair logits
        ``D(G(z),z)`` with matching shapes.
    Output: Scalar discriminator loss.
    Mathematical meaning: Minimizes the negative of
        ``E[log sigmoid(real_logits)] + E[log(1-sigmoid(fake_logits))]``.
    """
    if real_logits.numel() == 0 or fake_logits.numel() == 0:
        raise ValueError("discriminator logits cannot be empty")
    real_targets = torch.ones_like(real_logits)
    fake_targets = torch.zeros_like(fake_logits)
    return 0.5 * (
        F.binary_cross_entropy_with_logits(real_logits, real_targets)
        + F.binary_cross_entropy_with_logits(fake_logits, fake_targets)
    )


def encoder_generator_loss(real_logits: Tensor, fake_logits: Tensor) -> Tensor:
    """Compute the non-saturating encoder/generator adversarial objective.

    Input: Real-pair logits and generated-pair logits from the discriminator.
    Output: Scalar loss minimized by the encoder and generator.
    Mathematical meaning: Uses target zero for encoded real pairs and target
        one for generated pairs, encouraging the two joint distributions to
        become indistinguishable under the discriminator.
    """
    real_targets = torch.zeros_like(real_logits)
    fake_targets = torch.ones_like(fake_logits)
    return 0.5 * (
        F.binary_cross_entropy_with_logits(real_logits, real_targets)
        + F.binary_cross_entropy_with_logits(fake_logits, fake_targets)
    )


def pixel_reconstruction_loss(
    observations: Tensor,
    reconstructions: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """Compute pixel reconstruction error between observations and generated images.

    Input: Real observations and generated/reconstructed observations with
        identical shape, plus ``"mean"`` or ``"none"`` reduction.
    Output: Scalar mean error or per-sample error tensor.
    Mathematical meaning: Computes the Eq. 4 L1 term
        ``||x - G(E(x))||_1`` as a sum of absolute differences over all
        observation coordinates.
    """
    if observations.shape != reconstructions.shape:
        raise ValueError("observations and reconstructions must have identical shapes")
    if reduction not in {"mean", "none"}:
        raise ValueError("reduction must be mean or none")
    per_sample_l1 = (
        observations.float() - reconstructions.float()
    ).abs().reshape(observations.shape[0], -1).sum(dim=1)
    if reduction == "mean":
        return per_sample_l1.mean()
    return per_sample_l1


def feature_matching_loss(
    real_features: Tensor,
    generated_features: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """Compute feature-space mismatch for paired real and generated samples.

    Input: Real and generated discriminator feature tensors with identical
        shapes, plus ``"mean"`` or ``"none"`` reduction.
    Output: Scalar mean squared feature error or one error per sample.
    Mathematical meaning: Computes the Eq. 4 L1 term
        ``||h_omega(x,E(x)) - h_omega(G(z),z)||_1`` as a sum of absolute
        feature differences.
    """
    if real_features.shape != generated_features.shape:
        raise ValueError("feature tensors must have identical shapes")
    if reduction not in {"mean", "none"}:
        raise ValueError("reduction must be mean or none")
    per_sample_l1 = (real_features - generated_features).abs().reshape(
        real_features.shape[0], -1
    ).sum(dim=1)
    if reduction == "mean":
        return per_sample_l1.mean()
    return per_sample_l1


def gradient_penalty(
    discriminator_logits: Tensor,
    inputs: Tensor,
    coefficient: float = 1.0,
) -> Tensor:
    """Compute an optional input-gradient norm penalty.

    Input: Per-sample discriminator logits, differentiable input tensor, and
        non-negative penalty coefficient. ``inputs`` must have
        ``requires_grad=True``.
    Output: Scalar penalty
        ``coefficient * E[(||grad_input D||_2 - 1)^2]``.
    Mathematical meaning: Regularizes the discriminator's joint score
        sensitivity and can be added to its adversarial objective.
    """
    if coefficient < 0.0:
        raise ValueError("coefficient must be non-negative")
    if not inputs.requires_grad:
        raise ValueError("inputs must require gradients for gradient penalty")
    gradients = torch.autograd.grad(
        outputs=discriminator_logits.sum(),
        inputs=inputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    norms = gradients.reshape(gradients.shape[0], -1).norm(p=2, dim=1)
    return coefficient * (norms - 1.0).square().mean()


def combine_bigan_losses(
    discriminator_component: Tensor,
    encoder_generator_component: Tensor,
    pixel_component: Optional[Tensor] = None,
    feature_component: Optional[Tensor] = None,
    gradient_penalty_component: Optional[Tensor] = None,
    pixel_coefficient: float = 0.0,
    feature_coefficient: float = 0.0,
) -> BiGANLossValues:
    """Combine explicit BiGAN loss terms and return a complete loss record.

    Input: Adversarial discriminator and encoder/generator losses, optional
        pixel/feature/gradient terms, and non-negative reconstruction/feature
        coefficients.
    Output: ``BiGANLossValues`` containing components and weighted total
        discriminator/encoder-generator objectives.
    Mathematical meaning: Forms the weighted sum of the selected objective
        terms while preserving each unweighted component for diagnostics.
    """
    if pixel_coefficient < 0.0 or feature_coefficient < 0.0:
        raise ValueError("pixel and feature coefficients must be non-negative")
    zero = discriminator_component.new_zeros(())
    pixel = pixel_component if pixel_component is not None else zero
    feature = feature_component if feature_component is not None else zero
    penalty = gradient_penalty_component if gradient_penalty_component is not None else zero
    encoder_generator = (
        encoder_generator_component
        + pixel_coefficient * pixel
        + feature_coefficient * feature
    )
    discriminator = discriminator_component + penalty
    return BiGANLossValues(
        discriminator=discriminator,
        encoder_generator=encoder_generator,
        pixel_reconstruction=pixel,
        feature_matching=feature,
        gradient_penalty=penalty,
    )
