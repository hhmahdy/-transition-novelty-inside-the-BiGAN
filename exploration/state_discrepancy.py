"""EME-style state discrepancy measured in the BiGAN latent metric space.

Adventurer scores a single state by how badly the BiGAN reconstructs it. The
metric-based exploration bonus of this module instead scores a *transition* by
how far it moves in the encoder's latent metric space,

.. math:: d_t = \\lVert E_\\psi(s_t) - E_\\psi(s_{t+1}) \\rVert_p ,

with :math:`p \\in \\{1, 2\\}`. The encoder is used purely as a learned metric:
no generator reconstruction and no discriminator pass are required, so the
bonus costs one extra encoder forward per environment step.

The full exploration bonus of Contribution 2 is

.. math:: b_t = d_t \\cdot \\min(\\max(\\zeta(r), 1), M),

where the diversity-enhanced scaling factor :math:`\\zeta(r)` is supplied by
:mod:`exploration.ensemble_scaling`. This module implements only the metric
term ``d_t``; the product is assembled in
:class:`exploration.intrinsic_reward.MetricIntrinsicReward`.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import Tensor

from bigan.encoder import BiGANEncoder

_NORM_ORDERS = {"L1": 1.0, "L2": 2.0}


class LatentStateDiscrepancy:
    """Measure consecutive-state distance in the BiGAN latent space.

    Args:
        encoder: BiGAN encoder ``E_psi`` used as a frozen metric embedding.
        latent_dim: Expected latent dimension. When omitted the encoder's own
            ``latent_dim`` is used; when supplied it is validated so that a
            mismatched checkpoint fails loudly.
        norm: ``"L1"`` or ``"L2"`` latent norm.
        device: Device on which encoding and distance computation happen.
    """

    def __init__(
        self,
        encoder: BiGANEncoder,
        latent_dim: Optional[int] = None,
        norm: str = "L2",
        normalize_latent: bool = True,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        """Bind the encoder that defines the exploration metric.

        Input: BiGAN encoder, optional expected latent dimension, norm name,
            latent-normalization switch, and device.
        Output: Initialized discrepancy operator.
        Mathematical meaning: Fixes the metric space ``(Z, ||.||_p)`` in which
            transition novelty is measured. With ``normalize_latent`` the codes
            are projected onto the unit sphere, ``z -> z/||z||_2``, before the
            norm, and the distance is rescaled by ``sqrt(latent_dim)``; the
            metric then measures *directional* displacement in a bounded space
            instead of inheriting the unbounded weight scale of the encoder.
        """
        if norm not in _NORM_ORDERS:
            raise ValueError("norm must be 'L1' or 'L2'")
        if latent_dim is not None and latent_dim != encoder.latent_dim:
            raise ValueError("latent_dim does not match the encoder latent dimension")
        self.encoder = encoder.to(device)
        self.latent_dim = int(encoder.latent_dim)
        self.norm = norm
        self.norm_order = _NORM_ORDERS[norm]
        self.normalize_latent = bool(normalize_latent)
        self.device = torch.device(device)

    def freeze(self) -> None:
        """Disable gradients for the encoder so the metric stops drifting.

        Input: This discrepancy operator.
        Output: No value; encoder parameters are set to ``requires_grad=False``
            and the module is placed in evaluation mode.
        Mathematical meaning: Fixes ``E_psi`` after BiGAN pretraining so the
            latent metric, and therefore the scale of ``d_t``, is stationary.
        """
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def encode(self, observations: Tensor) -> Tensor:
        """Encode a batch of observations into latent codes.

        Input: Observation batch ``[B, *observation_shape]``.
        Output: Latent tensor ``[B, latent_dim]``.
        Mathematical meaning: Computes ``z=E_psi(s)`` without building a
            gradient graph, because the bonus is an RL signal and never an
            encoder optimization path.
        """
        if observations.ndim < 2:
            raise ValueError("observations must include a batch dimension")
        return self.encoder(observations.to(self.device).float())

    def normalize_codes(self, codes: Tensor) -> Tensor:
        """Project latent codes onto the unit sphere when normalization is on.

        Input: Latent tensor ``[B, latent_dim]``.
        Output: Latent tensor with unit rows (or the input when normalization
            is disabled).
        Mathematical meaning: Computes ``z/||z||_2``, which bounds any
            pairwise code distance by ``2`` (``2*sqrt(latent_dim)`` after the
            rescaling) and removes the encoder's global weight scale from the
            metric.
        """
        if not self.normalize_latent:
            return codes
        return torch.nn.functional.normalize(codes, p=2.0, dim=-1)

    def distance_from_latents(self, z_t: Tensor, z_tp1: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return the metric distance plus the normalized codes.

        Input: Raw latent codes of ``s_t`` and ``s_{t+1}``, ``[B, latent_dim]``.
        Output: ``(distance, normalized z_t, normalized z_tp1)`` where the
            distance is ``||z_t - z_t+1||_p`` on the (optionally normalized)
            codes, rescaled by ``sqrt(latent_dim)``.
        Mathematical meaning: Evaluates ``d_t`` once while exposing the codes
            used, so callers can derive diagnostics (mean ``||z||``) and
            episodic-count keys without a second encoder pass.
        """
        z_t = self.normalize_codes(z_t)
        z_tp1 = self.normalize_codes(z_tp1)
        distance = (z_t - z_tp1).norm(p=self.norm_order, dim=-1)
        if self.normalize_latent:
            # Restore the pre-normalization magnitude: distances between unit
            # codes live in [0, 2], so scale by sqrt(latent_dim) to keep the
            # bonus on roughly the same order as the unnormalized metric.
            distance = distance * (self.latent_dim ** 0.5)
        return distance, z_t, z_tp1

    @torch.no_grad()
    def __call__(self, obs_t: Tensor, obs_tp1: Tensor) -> Tensor:
        """Return the latent distance travelled by each transition.

        Input: Consecutive observation batches ``s_t`` and ``s_{t+1}``, both
            shaped ``[B, *observation_shape]``.
        Output: Non-negative tensor ``[B]`` of latent distances.
        Mathematical meaning: Computes ``d_t=||E(s_t)-E(s_{t+1})||_p``, the
            metric-based novelty of one transition, on unit-sphere codes when
            latent normalization is enabled.
        """
        if obs_t.shape != obs_tp1.shape:
            raise ValueError("obs_t and obs_tp1 must have identical shapes")
        distance, _, _ = self.distance_from_latents(
            self.encode(obs_t), self.encode(obs_tp1)
        )
        return distance
