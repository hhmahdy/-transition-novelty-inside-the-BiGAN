"""Intrinsic-reward conversion without an extra scaling coefficient.

Equation (5) already places normalized novelty on the extrinsic-reward scale.
The single beta coefficient is applied later to the intrinsic advantage inside
PPO, never here.

Two novelty sources can feed this module:

* Adventurer's reconstruction novelty ``B(s)`` from :mod:`exploration.novelty`
  or its transition variant, converted by :class:`IntrinsicRewardPipeline`.
* The metric-based exploration bonus of Contribution 2,
  ``b_t = ||E(s_t)-E(s_{t+1})||_p * min(max(zeta(r), 1), M)``, produced by
  :class:`MetricIntrinsicReward`.

Both paths emit the same structural contract -- a components object exposing
``combined_score`` and ``normalized_score`` -- so the trainer, episodic memory,
and logger treat them interchangeably.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from torch import Tensor

from exploration.ensemble_scaling import EnsembleRewardVariance
from exploration.novelty import NormalizedNoveltyScore, NoveltyComponents
from exploration.state_discrepancy import LatentStateDiscrepancy
EME_MODES = {"clamped", "normalised"}


NoveltyLike = Any


@dataclass(frozen=True)
class IntrinsicRewardResult:
    """Separate extrinsic and intrinsic reward streams."""

    extrinsic_reward: Tensor
    intrinsic_reward: Tensor
    normalized_novelty: Tensor


class IntrinsicRewardCalculator:
    """Return Eq. (5) novelty as intrinsic reward without rescaling."""

    def __init__(self, clip_value: Optional[float] = None) -> None:
        """Initialize optional safety clipping only.

        Input: Optional positive clipping bound. No multiplicative coefficient
            is accepted because Eq. (5) already performs reward scaling.
        Output: Intrinsic reward converter.
        Mathematical meaning: Implements ``r_i = normalized_novelty``.
        """
        if clip_value is not None and clip_value <= 0.0:
            raise ValueError("clip_value must be positive when provided")
        self.clip_value = clip_value

    def __call__(self, normalized_novelty: Tensor) -> Tensor:
        """Return normalized novelty as intrinsic reward.

        Input: Eq. (5)-normalized novelty tensor.
        Output: Intrinsic reward tensor with identical values unless optional
            safety clipping is configured.
        Mathematical meaning: Computes ``r_t^i=r_t^i(s_t)`` with no beta.
        """
        intrinsic_reward = normalized_novelty
        if self.clip_value is not None:
            intrinsic_reward = intrinsic_reward.clamp(-self.clip_value, self.clip_value)
        return intrinsic_reward


class IntrinsicRewardPipeline:
    """Produce separate extrinsic and Eq. (5) intrinsic rewards."""

    def __init__(self, clip_value: Optional[float] = None) -> None:
        """Construct the reward converter.

        Input: Optional safety clip only.
        Output: Pipeline that never mixes or multiplicatively scales rewards.
        Mathematical meaning: Leaves beta exclusively in advantage combination.
        """
        self.calculator = IntrinsicRewardCalculator(clip_value)

    def __call__(self, extrinsic_reward: Tensor, novelty: "NoveltyLike") -> IntrinsicRewardResult:
        """Return independent reward streams.

        Input: Environment reward and any novelty result exposing
            ``normalized_score`` -- reconstruction novelty, transition novelty,
            or the metric bonus -- with matching batch shape.
        Output: Separate extrinsic reward, intrinsic reward, and normalized score.
        Mathematical meaning: Produces ``r_t^e`` unchanged and ``r_t^i`` from
            Eq. (5), with no pre-GAE mixture.
        """
        normalized_novelty = novelty.normalized_score
        if extrinsic_reward.shape != normalized_novelty.shape:
            raise ValueError("extrinsic reward and novelty shapes must match")
        return IntrinsicRewardResult(
            extrinsic_reward=extrinsic_reward,
            intrinsic_reward=self.calculator(normalized_novelty),
            normalized_novelty=normalized_novelty,
        )


@dataclass(frozen=True)
class MetricNoveltyComponents:
    """Per-transition components of the metric-based exploration bonus.

    The ``pixel_error`` and ``feature_error`` properties are compatibility
    aliases so this object can be logged and stored by the same trainer code
    that consumes :class:`exploration.novelty.NoveltyComponents`.
    """

    latent_distance: Tensor
    ensemble_variance: Tensor
    bonus_scale: Tensor
    combined_score: Tensor
    normalized_score: Tensor
    # Distance actually used inside the bonus: the geometric latent distance
    # or, when EME metric learning is enabled, the learned d_phi(z_t, z_t+1).
    bonus_distance: Optional[Tensor] = None
    # RIDE/NovelD-style habituation factor 1/sqrt(N_ep(s_t+1)) when the
    # episodic counter is enabled, else None.
    episodic_scale: Optional[Tensor] = None
    # P4 diagnostics: mean ||z|| over both transition endpoints, fraction of
    # raw ensemble variances binding each clamp, and geometric-distance
    # percentiles used to detect metric blow-ups in the logs.
    latent_norm_mean: float = 0.0
    variance_below_min_fraction: float = 0.0
    variance_above_max_fraction: float = 0.0
    latent_distance_p5: float = 0.0
    latent_distance_p50: float = 0.0
    latent_distance_p95: float = 0.0

    @property
    def pixel_error(self) -> Tensor:
        """Return the latent distance under the generic novelty name.

        Input: This components object.
        Output: Latent distance tensor ``[B]``.
        Mathematical meaning: Exposes ``||E(s_t)-E(s_{t+1})||_p`` where the
            reconstruction pipeline would report its pixel term.
        """
        return self.latent_distance

    @property
    def feature_error(self) -> Tensor:
        """Return the clamped scaling factor under the generic novelty name.

        Input: This components object.
        Output: Scaling tensor ``[B]``.
        Mathematical meaning: Exposes ``min(max(zeta(r),1),M)`` where the
            reconstruction pipeline would report its feature term.
        """
        return self.bonus_scale


class MetricIntrinsicReward:
    """Compute the BiGAN-latent, EME-scaled exploration bonus ``b_t``.

    Args:
        encoder: BiGAN encoder used as the metric embedding.
        ensemble: Optional ensemble supplying ``zeta(r)``. When ``None`` the
            scaling factor is identically one, which yields the latent-only
            ablation.
        eme_mode: How ``zeta(r)`` becomes a scaling factor.

            ``"clamped"`` is EME as published,
            ``scale = min(max(zeta, min_reward_scaling), M)``. It assumes
            ``zeta`` lives on the order of one, which holds for shaped rewards
            but not for sparse Atari rewards: when almost every target is zero
            the members agree, ``zeta << 1``, the lower clamp binds, and the
            variant degenerates to the unscaled latent bonus.

            ``"normalised"`` divides by the running mean of ``zeta`` instead,
            ``scale = zeta / E[zeta]``, capped at ``M``. The factor is then
            scale-free: it is one for a transition of typical disagreement and
            greater than one exactly where the ensemble disagrees more than it
            usually does, whatever the reward magnitude. The ratio is exact --
            no epsilon enters the numerator, because an additive constant would
            pull the factor back towards one precisely in the tiny-``zeta``
            regime the mode exists to rescue.
        max_reward_scaling: Upper clamp ``M`` on the scaling factor.
        min_reward_scaling: Lower clamp of the ``"clamped"`` mode, one in the
            EME formulation, so the bonus is never shrunk below the pure metric
            distance. It is not applied in ``"normalised"`` mode.
        zeta_momentum: Momentum of the exponential moving average estimating
            ``E[zeta]``. A cumulative mean would grow steadily staler over a
            multi-million-step run -- once ``zeta`` starts rising, every batch
            would look anomalous relative to the whole history and the factor
            would saturate at ``M``. An EMA keeps the reference level on recent
            experience, so the factor stays centred near one and continues to
            discriminate. ``0.0`` compares against the current batch only.
        zeta_epsilon: Degeneracy threshold of the normalised mode. When the
            running mean of ``zeta`` does not exceed it the ensemble carries no
            usable signal, and the scaling factor falls back to one so the
            variant reduces to the pure metric bonus instead of zeroing the
            exploration reward.
        norm: ``"L1"`` or ``"L2"`` latent norm.
        normalize_latent: L2-normalize the latent codes before measuring the
            distance (rescaled by ``sqrt(latent_dim)``). On by default: the
            encoder output is an unbounded linear head, and an adversarially
            drifting weight scale otherwise leaks straight into ``d_t`` and
            destroys the metric (observed as distance blow-ups in training).
        episodic_counter: Optional episodic visit counter; when supplied the
            bonus is habituated with the RIDE/NovelD factor
            ``1/sqrt(N_ep(s_{t+1}))``, so revisited states stop paying and
            death/respawn transitions no longer dominate the signal.
        metric_head: Optional trained EME metric head; when supplied the bonus
            distance is EME's learned metric ``d_phi(z_t, z_{t+1})`` (Eq. 9:
            value difference + bootstrapped distance + policy KL) instead of
            the geometric displacement.
        normalize: Whether to map the raw bonus onto the extrinsic-reward scale
            with Adventurer's Eq. (5) running normalization. Keeping this on
            makes the metric bonus directly comparable with the baseline's
            intrinsic reward and keeps ``beta`` meaningful across variants.
        normalization_epsilon: Stability constant of the running normalizer.
        normalization_clip: Optional symmetric bound applied to the normalized
            bonus. Latent distances are far more concentrated than
            reconstruction errors, so early rollouts can produce a very small
            ``sigma(b)`` and an exploding Eq. (5) ratio; the default bound of
            five keeps the intrinsic stream on the same order as the extrinsic
            one without changing the ranking of transitions.
        clip_value: Optional symmetric clip applied to the final bonus.
        device: Device used for encoding and statistics.
    """

    def __init__(
        self,
        encoder,
        ensemble: Optional[EnsembleRewardVariance] = None,
        max_reward_scaling: float = 5.0,
        min_reward_scaling: float = 1.0,
        eme_mode: str = "clamped",
        zeta_momentum: float = 0.99,
        zeta_epsilon: float = 1.0e-12,
        norm: str = "L2",
        normalize_latent: bool = True,
        episodic_counter=None,
        metric_head=None,
        normalize: bool = True,
        normalization_epsilon: float = 1.0e-8,
        normalization_clip: Optional[float] = 5.0,
        clip_value: Optional[float] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        """Assemble the metric term, the scaling term, and the reward scaling.

        Input: Encoder, optional ensemble, clamp bounds, norm, latent
            normalization switch, optional episodic counter and metric head,
            normalization settings, and device.
        Output: Initialized bonus operator.
        Mathematical meaning: Defines
            ``b_t = d_t * zeta_scale [* 1/sqrt(N_ep(s_t+1))]``, where ``d_t``
            is the learned metric ``d_phi(z_t, z_t+1)`` when a head is
            supplied and the (unit-sphere) latent distance otherwise, and
            ``zeta_scale`` is ``min(max(zeta,1),M)`` in clamped mode and
            ``min(zeta/E[zeta], M)`` in normalised mode, evaluated at
            ``(s_t, a_t)`` per EME Eq. (10).
        """
        if eme_mode not in EME_MODES:
            raise ValueError("eme_mode must be 'clamped' or 'normalised'")
        if zeta_epsilon <= 0.0:
            raise ValueError("zeta_epsilon must be positive")
        if not 0.0 <= zeta_momentum < 1.0:
            raise ValueError("zeta_momentum must satisfy 0 <= momentum < 1")
        if max_reward_scaling <= 0.0 or min_reward_scaling <= 0.0:
            raise ValueError("scaling bounds must be positive")
        if max_reward_scaling < min_reward_scaling:
            raise ValueError("max_reward_scaling must be at least min_reward_scaling")
        if clip_value is not None and clip_value <= 0.0:
            raise ValueError("clip_value must be positive when provided")
        self.device = torch.device(device)
        self.discrepancy = LatentStateDiscrepancy(
            encoder, norm=norm, normalize_latent=normalize_latent, device=self.device
        )
        self.ensemble = ensemble
        self.max_reward_scaling = float(max_reward_scaling)
        self.min_reward_scaling = float(min_reward_scaling)
        self.eme_mode = eme_mode
        self.zeta_epsilon = float(zeta_epsilon)
        self.zeta_momentum = float(zeta_momentum)
        self._mean_zeta: Optional[Tensor] = None
        self.episodic_counter = episodic_counter
        self.metric_head = metric_head
        self.normalize = bool(normalize)
        self.calculator = IntrinsicRewardCalculator(clip_value)
        if normalization_clip is not None and normalization_clip <= 0.0:
            raise ValueError("normalization_clip must be positive when provided")
        self.normalized_score = NormalizedNoveltyScore(
            epsilon=normalization_epsilon,
            clip_range=normalization_clip,
            device=self.device,
        )

    @property
    def mean_zeta(self) -> float:
        """Return the current estimate of the mean ensemble variance.

        Input: This bonus operator.
        Output: Non-negative float, zero before any batch has been seen.
        Mathematical meaning: Estimates ``E[zeta(r)]`` by exponential moving
            average, the reference level against which the normalised mode
            measures disagreement.
        """
        if self._mean_zeta is None:
            return 0.0
        return float(self._mean_zeta.item())

    def _update_mean_zeta(self, variance: Tensor) -> Tensor:
        """Fold one batch of variances into the moving reference level.

        Input: Ensemble variance batch ``[B]``.
        Output: Updated scalar estimate of ``E[zeta]``.
        Mathematical meaning: Applies
            ``m <- momentum*m + (1-momentum)*mean(zeta_batch)``, seeded by the
            first observed batch mean so no zero-initialization bias is carried
            into early scaling factors.
        """
        batch_mean = variance.detach().mean().to(self.device)
        if self._mean_zeta is None:
            self._mean_zeta = batch_mean.clone()
        else:
            self._mean_zeta.mul_(self.zeta_momentum).add_(
                batch_mean * (1.0 - self.zeta_momentum)
            )
        return self._mean_zeta

    def scaling_factor(
        self,
        obs_t: Tensor,
        actions: Optional[Tensor] = None,
        update_statistics: bool = True,
    ) -> tuple[Tensor, Tensor, float, float]:
        """Return the raw ensemble variance and the resulting scaling factor.

        Input: Current-observation batch ``[B, *observation_shape]``, the
            actions taken in those states, and a flag controlling whether the
            running ``zeta`` mean is updated.
        Output: ``(zeta, scale, below_min_fraction, above_max_fraction)`` where
            the first two are tensors shaped ``[B]`` and the fractions are
            floats reporting how much of the raw variance binds each clamp.
        Mathematical meaning: Computes
            ``zeta(r)=Var_k(hat r_k(s_t, a_t))`` -- EME Eq. (10) evaluates the
            scaling factor at ``s_t`` under the policy's action -- and maps it
            to ``min(max(zeta,min),M)`` in clamped mode or to
            ``min(zeta/E[zeta],M)`` in normalised mode. Without an ensemble, or
            with a degenerate one, both terms are one.
        """
        if self.ensemble is None:
            ones = torch.ones(obs_t.shape[0], device=self.device)
            return ones.clone(), ones, 1.0, 0.0
        variance = self.ensemble.get_variance(obs_t, actions).to(self.device).float()
        below_min_fraction = float((variance <= self.min_reward_scaling).float().mean())
        above_max_fraction = float((variance >= self.max_reward_scaling).float().mean())
        if self.eme_mode == "clamped":
            scale = variance.clamp(
                min=self.min_reward_scaling, max=self.max_reward_scaling
            )
            return variance, scale, below_min_fraction, above_max_fraction
        if update_statistics:
            self._update_mean_zeta(variance)
        mean_zeta = torch.as_tensor(
            self.mean_zeta, device=variance.device, dtype=variance.dtype
        )
        if float(mean_zeta) <= self.zeta_epsilon:
            # A degenerate ensemble carries no signal: fall back to the pure
            # metric bonus rather than zeroing the exploration reward.
            return variance, torch.ones_like(variance), below_min_fraction, above_max_fraction
        scale = variance / mean_zeta
        return (
            variance,
            scale.clamp(max=self.max_reward_scaling),
            below_min_fraction,
            above_max_fraction,
        )

    @torch.no_grad()
    def __call__(
        self,
        obs_t: Tensor,
        obs_tp1: Tensor,
        extrinsic_reward: Optional[Tensor] = None,
        update_statistics: bool = True,
        done: Optional[Tensor] = None,
        actions: Optional[Tensor] = None,
    ) -> MetricNoveltyComponents:
        """Compute the exploration bonus for one batch of transitions.

        Input: Consecutive observation batches, the matching extrinsic reward
            used to centre Eq. (5), a statistics-update flag, the done mask of
            the transitions, and the actions taken at ``s_t``.
        Output: ``MetricNoveltyComponents`` whose ``normalized_score`` is the
            intrinsic reward handed to PPO.
        Mathematical meaning: Evaluates ``b_t`` and, when normalization is on,
            ``(b_t-mu(b)+mu(r^e))/sigma(b)``. Transitions with ``done`` set get
            a zero distance: a death or respawn frame jump is not exploration
            progress, and paying it trains the agent to terminate episodes.
        """
        obs_t = obs_t.to(self.device)
        obs_tp1 = obs_tp1.to(self.device)
        latent_distance, z_t, z_tp1 = self.discrepancy.distance_from_latents(
            self.discrepancy.encode(obs_t), self.discrepancy.encode(obs_tp1)
        )
        latent_norm_mean = float(
            0.5 * (z_t.norm(dim=-1).mean() + z_tp1.norm(dim=-1).mean())
        )
        if self.metric_head is not None:
            bonus_distance = self.metric_head(z_t, z_tp1)
        else:
            bonus_distance = latent_distance
        if done is not None:
            keep = (~done.reshape(-1).to(self.device).bool()).float()
            bonus_distance = bonus_distance * keep
        variance, scale, below_min_fraction, above_max_fraction = self.scaling_factor(
            obs_t, actions, update_statistics
        )
        episodic_scale: Optional[Tensor] = None
        if self.episodic_counter is not None:
            episodic_scale = self.episodic_counter.scale(z_tp1)
            if done is not None:
                episodic_scale = episodic_scale * keep
            self.episodic_counter.reset(done)
            bonus = bonus_distance * scale * episodic_scale
        else:
            bonus = bonus_distance * scale
        percentiles = torch.quantile(
            latent_distance.detach().float(),
            torch.tensor([0.05, 0.5, 0.95], device=latent_distance.device),
        )
        if self.normalize:
            normalized = self.normalized_score(
                bonus,
                update=update_statistics,
                extrinsic_reward=extrinsic_reward,
            )
        else:
            normalized = bonus
        return MetricNoveltyComponents(
            latent_distance=latent_distance,
            ensemble_variance=variance,
            bonus_scale=scale,
            combined_score=bonus,
            normalized_score=self.calculator(normalized),
            bonus_distance=bonus_distance,
            episodic_scale=episodic_scale,
            latent_norm_mean=latent_norm_mean,
            variance_below_min_fraction=below_min_fraction,
            variance_above_max_fraction=above_max_fraction,
            latent_distance_p5=float(percentiles[0]),
            latent_distance_p50=float(percentiles[1]),
            latent_distance_p95=float(percentiles[2]),
        )
