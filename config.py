"""Configuration objects for reproducible Adventurer experiments.

This module contains validated, immutable dataclasses for the PPO agent, the
BiGAN novelty model, and the training pipeline. Keeping configuration separate
from implementation makes experiment comparisons auditable and prevents
implicit hyperparameters from being introduced in individual modules.

The mathematical quantities represented here are used by later modules:

* PPO discounting and GAE use :math:`gamma` and :math:`lambda` (PPO/GAE,
  Equations (8)--(10) in the paper's PPO formulation).
* PPO clipping uses :math:`epsilon` in the clipped surrogate objective
  (Equation (7)).
* BiGAN optimization uses the learning rates, batch size, latent dimension,
  and update schedule in the adversarial objective (Equations (1)--(6)).
* Novelty weighting uses the reconstruction/feature coefficients and the
  intrinsic-reward coefficient in the Adventurer objective.

All dataclasses are frozen so that a run cannot accidentally mutate its
hyperparameters after initialization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Optional


class DeviceType(str, Enum):
    """Allowed device-selection policies for experiment execution."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


#: Allowed implementations of the latent transition novelty. ``"legacy"`` keeps
#: the historical ``--novelty transition`` behavior unchanged; ``"master_l2"``
#: is the Master's thesis method ``N_T = ||f(E(s_t),a_t) - E(s_{t+1})||_2``.
TRANSITION_VARIANTS = ("legacy", "master_l2")

#: Master's thesis experiment methods accepted by ``main.py --method``. ``state``
#: is the Adventurer BiGAN state-novelty baseline, ``transition`` is the BiGAN
#: action-conditioned transition novelty.
MASTER_METHODS = ("state", "transition")


@dataclass(frozen=True)
class SeedConfig:
    """Control pseudorandom seeds and deterministic PyTorch behavior.

    Attributes:
        seed: Base integer seed used for Python, NumPy, PyTorch, and
            environment seeding.
        deterministic: Whether deterministic backend algorithms should be
            requested where supported.
        benchmark: Whether cuDNN benchmarking is enabled. It should normally
            be false for strict reproducibility.
    """

    seed: int = 0
    deterministic: bool = True
    benchmark: bool = False

    def __post_init__(self) -> None:
        """Validate the seed and deterministic-backend settings.

        Input: The values supplied to this dataclass.
        Output: No value; raises ``ValueError`` for an invalid seed/settings
            combination.
        Mathematical meaning: The seed identifies one reproducible sample
            path from the random transition and optimization processes.
        """
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.deterministic and self.benchmark:
            raise ValueError("benchmark must be false in deterministic mode")


@dataclass(frozen=True)
class EnvironmentConfig:
    """Describe the interface and episode limits of the RL environment."""

    observation_shape: tuple[int, ...] = (84, 84, 4)
    action_dim: int = 4
    discrete_actions: bool = True
    num_parallel_envs: int = 96
    max_episode_steps: int = 27_000
    frame_stack: int = 4
    normalize_pixels: bool = True

    def __post_init__(self) -> None:
        """Validate observation and action-space dimensions.

        Input: Environment dimensions and preprocessing settings.
        Output: No value; raises ``ValueError`` if dimensions are unusable.
        Mathematical meaning: The observation shape defines the domain of the
            encoder, actor, and critic; ``action_dim`` defines the policy's
            categorical support or the action-vector width.
        """
        if not self.observation_shape or any(d <= 0 for d in self.observation_shape):
            raise ValueError("observation_shape must contain only positive dimensions")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if self.num_parallel_envs <= 0:
            raise ValueError("num_parallel_envs must be positive")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if self.frame_stack <= 0:
            raise ValueError("frame_stack must be positive")


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparameters for the from-scratch PPO implementation.

    ``gamma`` and ``gae_lambda`` define the temporal-difference residual and
    exponentially weighted GAE estimator. ``clip_epsilon`` defines the trust
    region used by PPO's clipped policy ratio. ``value_clip_epsilon`` is
    optional PPO2-style value clipping and is disabled by default because the
    policy objective and value objective are configured independently.
    """

    gamma: float = 0.997
    gae_lambda: float = 0.95
    intrinsic_gamma: float = 0.90
    intrinsic_gae_lambda: float = 0.95
    clip_epsilon: float = 0.10
    value_clip_epsilon: Optional[float] = None
    entropy_coefficient: float = 0.01
    intrinsic_advantage_coefficient: float = 0.3
    value_loss_coefficient: float = 0.5
    intrinsic_value_loss_coefficient: float = 0.5
    learning_rate: float = 1.0e-4
    critic_learning_rate: Optional[float] = None
    intrinsic_critic_learning_rate: Optional[float] = None
    adam_epsilon: float = 1.0e-5
    max_grad_norm: float = 0.5
    rollout_steps: int = 128
    update_epochs: int = 4
    minibatch_size: int = 32
    normalize_advantages: bool = True
    lr_decay: bool = True
    lr_final_fraction: float = 0.0

    def __post_init__(self) -> None:
        """Validate PPO and GAE parameters.

        Input: PPO optimizer, objective, and rollout settings.
        Output: No value; raises ``ValueError`` for mathematically invalid
            ranges or incompatible batching.
        Mathematical meaning: The ranges enforce valid discounted returns,
            GAE weights, probability-ratio clipping, and complete minibatches
            for every PPO epoch.
        """
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must satisfy 0 <= gamma < 1")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must satisfy 0 <= gae_lambda <= 1")
        if not 0.0 <= self.intrinsic_gamma < 1.0:
            raise ValueError("intrinsic_gamma must satisfy 0 <= intrinsic_gamma < 1")
        if not 0.0 <= self.intrinsic_gae_lambda <= 1.0:
            raise ValueError("intrinsic_gae_lambda must satisfy 0 <= intrinsic_gae_lambda <= 1")
        if self.clip_epsilon <= 0.0:
            raise ValueError("clip_epsilon must be positive")
        if self.value_clip_epsilon is not None and self.value_clip_epsilon <= 0.0:
            raise ValueError("value_clip_epsilon must be positive when provided")
        if (
            self.entropy_coefficient < 0.0
            or self.value_loss_coefficient < 0.0
            or self.intrinsic_value_loss_coefficient < 0.0
        ):
            raise ValueError("loss coefficients must be non-negative")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.critic_learning_rate is not None and self.critic_learning_rate <= 0.0:
            raise ValueError("critic_learning_rate must be positive when provided")
        if (
            self.intrinsic_critic_learning_rate is not None
            and self.intrinsic_critic_learning_rate <= 0.0
        ):
            raise ValueError(
                "intrinsic_critic_learning_rate must be positive when provided"
            )
        if self.adam_epsilon <= 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("adam_epsilon and max_grad_norm must be positive")
        if self.rollout_steps <= 0 or self.update_epochs <= 0 or self.minibatch_size <= 0:
            raise ValueError("rollout_steps, update_epochs, and minibatch_size must be positive")
        if self.rollout_steps % self.minibatch_size != 0:
            raise ValueError("rollout_steps must be divisible by minibatch_size")
        if not 0.0 <= self.lr_final_fraction <= 1.0:
            raise ValueError("lr_final_fraction must be in [0, 1]")


@dataclass(frozen=True)
class BiGANConfig:
    """Architecture and optimization settings for the BiGAN model."""

    latent_dim: int = 128
    feature_dim: int = 256
    hidden_dim: int = 512
    learning_rate: float = 2.0e-4
    adam_beta1: float = 0.5
    adam_beta2: float = 0.999
    adam_epsilon: float = 1.0e-8
    batch_size: int = 64
    discriminator_steps: int = 1
    generator_encoder_steps: int = 1
    gradient_penalty_coefficient: float = 0.0
    warmup_updates: int = 0
    update_interval: int = 1

    def __post_init__(self) -> None:
        """Validate BiGAN dimensions, optimizer values, and update schedule.

        Input: BiGAN architecture and adversarial-training parameters.
        Output: No value; raises ``ValueError`` for invalid settings.
        Mathematical meaning: Positive latent/feature dimensions define the
            encoder-generator coupling, while non-negative coefficients define
            the weighted adversarial and optional regularization objectives.
        """
        if min(self.latent_dim, self.feature_dim, self.hidden_dim, self.batch_size) <= 0:
            raise ValueError("BiGAN dimensions and batch_size must be positive")
        if self.learning_rate <= 0.0 or self.adam_epsilon <= 0.0:
            raise ValueError("BiGAN learning_rate and adam_epsilon must be positive")
        if not 0.0 <= self.adam_beta1 < 1.0 or not 0.0 <= self.adam_beta2 < 1.0:
            raise ValueError("Adam beta values must be in [0, 1)")
        if self.discriminator_steps <= 0 or self.generator_encoder_steps <= 0:
            raise ValueError("BiGAN update counts must be positive")
        if self.gradient_penalty_coefficient < 0.0:
            raise ValueError("gradient_penalty_coefficient must be non-negative")
        if self.warmup_updates < 0 or self.update_interval <= 0:
            raise ValueError("warmup_updates must be non-negative and update_interval positive")


@dataclass(frozen=True)
class NoveltyConfig:
    """Settings for state or transition novelty exploration.

    ``novelty_type`` selects the intrinsic signal:

    * ``"state"`` -- Adventurer's reconstruction novelty
      ``B(s) = alpha*L_G(s) + (1-alpha)*L_D(s)``. This is the Master's thesis
      baseline (``main.py --method state``).
    * ``"transition"`` -- a latent forward-model novelty. Two implementations
      exist and ``transition_variant`` selects between them:

      - ``"legacy"`` (default, unchanged) -- the historical variant
        ``T = alpha_t*||z'-f(z,a)||_1 + (1-alpha_t)*||f_D(s',z')-f_D(G(f(z,a)),f(z,a))||_1``
        trained with an L1 loss on a raw ``(s,a,s')`` replay buffer. Selected by
        ``main.py --novelty transition``.
      - ``"master_l2"`` -- the Master's thesis method
        ``N_T = ||f(E(s_t),a_t) - E(s_{t+1})||_2``, trained with the MSE loss
        ``(1/B) * sum_i ||f(E(s_t),a_t) - E(s_{t+1})||_2^2`` on a cached
        ``(z_t, a_t, z_{t+1})`` latent buffer, with no generator or
        discriminator term. Selected by ``main.py --method transition``.

    Both variants reuse the same BiGAN encoder, the same Equation (5) reward-scale
    normalization, and the same PPO configuration; only the intrinsic signal and
    the forward-model objective differ. ``transition_batch_size`` is used by the
    ``"master_l2"`` variant (the legacy variant keeps using ``BiGANConfig.batch_size``).
    """

    novelty_type: str = "state"
    alpha: float = 0.9
    transition_alpha: float = 0.9
    running_momentum: float = 0.99
    normalization_epsilon: float = 1.0e-8
    clip_intrinsic_reward: Optional[float] = None
    transition_hidden_dim: int = 256
    transition_learning_rate: float = 1.0e-4
    transition_update_epochs: int = 1
    transition_max_grad_norm: float = 0.5
    # Master's thesis switch: "legacy" preserves the historical transition
    # novelty exactly, "master_l2" selects the pure L2 forward-prediction error.
    transition_variant: str = "legacy"
    transition_batch_size: int = 64

    def __post_init__(self) -> None:
        """Validate novelty weighting and running-normalization parameters.

        Input: Reconstruction/feature weights, reward scale, and statistics
            settings.
        Output: No value; raises ``ValueError`` for invalid ranges.
        Mathematical meaning: The two non-negative coefficients form the
            combined novelty score; momentum and epsilon stabilize its online
            normalization before reward scaling.
        """
        if self.novelty_type not in {"state", "transition"}:
            raise ValueError("novelty_type must be 'state' or 'transition'")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must satisfy 0 <= alpha <= 1")
        if not 0.0 <= self.transition_alpha <= 1.0:
            raise ValueError("transition_alpha must satisfy 0 <= transition_alpha <= 1")
        if self.transition_hidden_dim <= 0 or self.transition_learning_rate <= 0.0:
            raise ValueError("transition model dimensions and learning rate must be positive")
        if self.transition_update_epochs <= 0 or self.transition_max_grad_norm <= 0.0:
            raise ValueError("transition update epochs and gradient norm must be positive")
        if self.transition_variant not in TRANSITION_VARIANTS:
            raise ValueError("transition_variant must be 'legacy' or 'master_l2'")
        if self.transition_batch_size <= 0:
            raise ValueError("transition_batch_size must be positive")
        if not 0.0 <= self.running_momentum < 1.0:
            raise ValueError("running_momentum must be in [0, 1)")
        if self.normalization_epsilon <= 0.0:
            raise ValueError("normalization_epsilon must be positive")
        if self.clip_intrinsic_reward is not None and self.clip_intrinsic_reward <= 0.0:
            raise ValueError("clip_intrinsic_reward must be positive when provided")


@dataclass(frozen=True)
class MetricEMEConfig:
    """Contribution 2: BiGAN latent discrepancy + EME variance scaling.

    The exploration bonus replaces Adventurer's reconstruction novelty with

    ``b_t = ||E(s_t) - E(s_{t+1})||_p * min(max(Var(hat r_1..hat r_K), 1), M)``.

    ``enabled`` switches the bonus on; ``ensemble_scaling`` switches the EME
    factor ``zeta(r)`` on. Enabling only the former gives the latent-only
    ablation in which the scaling factor is identically one.

    ``eme_mode`` selects how ``zeta(r)`` becomes a scaling factor:

    * ``"clamped"`` -- ``min(max(zeta, min_reward_scaling), M)``, EME as
      published. Assumes ``zeta`` is of order one, which fails for sparse
      rewards: near-zero regression targets make the members agree, the lower
      clamp binds, and the variant collapses onto the unscaled latent bonus.
    * ``"normalised"`` -- ``min(zeta/E[zeta], M)``, scale-free and therefore
      informative at any reward magnitude. Recommended for sparse-reward Atari.
      ``E[zeta]`` is an exponential moving average with momentum
      ``zeta_momentum``, and ``zeta_epsilon`` is the degeneracy threshold below
      which the factor falls back to one.
    """

    enabled: bool = False
    ensemble_scaling: bool = True
    ensemble_size: int = 5                # K
    eme_mode: str = "clamped"             # {clamped, normalised}
    max_reward_scaling: float = 5.0       # M
    min_reward_scaling: float = 1.0       # lower clamp of zeta(r), clamped mode
    zeta_momentum: float = 0.99           # EMA momentum of E[zeta]
    zeta_epsilon: float = 1.0e-12         # degeneracy threshold, normalised mode
    latent_norm: str = "L2"               # {L1, L2}
    # P0b: L2-normalize the latent codes before measuring ||z_t - z_t+1||_p and
    # rescale by sqrt(latent_dim). Without this the distance inherits the
    # (unbounded, drifting) scale of the encoder weights and can explode.
    normalize_latent_embeddings: bool = True
    # P2b: RIDE/NovelD-style episodic habituation, b_t = d_t / sqrt(N_ep(s_t+1)).
    # Without it the consecutive-state distance never habituates and the largest
    # distances come from death/respawn transitions.
    episodic_count_scaling: bool = False
    episodic_count_resolution: int = 32
    # P4: fraction of freshly collected (s, a, r) triples routed to a shared
    # held-out set used to report an ensemble validation loss.
    ensemble_validation_fraction: float = 0.05
    # P2c (optional): train EME's actual learned metric d_phi of Eq. (9)
    # (reward-difference + bootstrapped next-state distance + policy KL) on top
    # of the encoder embedding, and use d_phi(s_t, s_t+1) as the bonus distance.
    # "latent" keeps the geometric ||E(s_t)-E(s_t+1)||_p distance.
    metric_learning: str = "none"         # {none, eme}
    metric_learning_updates_per_rollout: int = 32
    metric_learning_batch_size: int = 256
    ensemble_hidden_dim: int = 256
    ensemble_learning_rate: float = 1.0e-3
    ensemble_batch_size: int = 64
    ensemble_min_buffer_size: int = 128
    ensemble_buffer_capacity: int = 100_000
    ensemble_bootstrap_probability: float = 0.5
    ensemble_max_grad_norm: float = 0.5
    # P1: EME specifies reward models g(s, a) trained repeatedly on bootstrap
    # resamples. One gradient step per rollout against 12,288 new samples left
    # the members unfitted, so zeta was driven by input drift, not by reward
    # structure; 64 steps per rollout brings each sample back about once every
    # three rollouts.
    ensemble_updates_per_rollout: int = 64
    ensemble_input: str = "latent"        # {latent, observation}
    normalize_bonus: bool = True
    bonus_normalization_clip: Optional[float] = 5.0
    freeze_encoder_after_updates: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate the metric bonus and ensemble hyperparameters.

        Input: Values supplied to this dataclass.
        Output: No value; raises ``ValueError`` for invalid settings.
        Mathematical meaning: Guarantees a well-defined metric ``||.||_p``, a
            non-degenerate ensemble variance (``K>=2``), a supported scaling
            mode, and a clamp interval ``[min, M]`` that never inverts.
        """
        if self.latent_norm not in {"L1", "L2"}:
            raise ValueError("latent_norm must be 'L1' or 'L2'")
        if self.eme_mode not in {"clamped", "normalised"}:
            raise ValueError("eme_mode must be 'clamped' or 'normalised'")
        if self.metric_learning not in {"none", "eme"}:
            raise ValueError("metric_learning must be 'none' or 'eme'")
        if self.episodic_count_resolution <= 0:
            raise ValueError("episodic_count_resolution must be positive")
        if self.metric_learning_updates_per_rollout < 0:
            raise ValueError("metric_learning_updates_per_rollout must be non-negative")
        if self.metric_learning_batch_size <= 0:
            raise ValueError("metric_learning_batch_size must be positive")
        if not 0.0 <= self.ensemble_validation_fraction < 1.0:
            raise ValueError("ensemble_validation_fraction must be in [0, 1)")
        if self.zeta_epsilon <= 0.0:
            raise ValueError("zeta_epsilon must be positive")
        if not 0.0 <= self.zeta_momentum < 1.0:
            raise ValueError("zeta_momentum must satisfy 0 <= zeta_momentum < 1")
        if self.ensemble_input not in {"latent", "observation"}:
            raise ValueError("ensemble_input must be 'latent' or 'observation'")
        if self.ensemble_size < 2:
            raise ValueError("ensemble_size must be at least two for a variance")
        if self.min_reward_scaling <= 0.0 or self.max_reward_scaling <= 0.0:
            raise ValueError("reward scaling bounds must be positive")
        if self.max_reward_scaling < self.min_reward_scaling:
            raise ValueError("max_reward_scaling must be at least min_reward_scaling")
        if self.ensemble_hidden_dim <= 0 or self.ensemble_learning_rate <= 0.0:
            raise ValueError("ensemble hidden dimension and learning rate must be positive")
        if self.ensemble_batch_size <= 0 or self.ensemble_min_buffer_size <= 0:
            raise ValueError("ensemble batch and minimum buffer sizes must be positive")
        if self.ensemble_min_buffer_size < self.ensemble_batch_size:
            raise ValueError("ensemble_min_buffer_size must be at least ensemble_batch_size")
        if self.ensemble_buffer_capacity < self.ensemble_min_buffer_size:
            raise ValueError("ensemble_buffer_capacity must hold at least min_buffer_size samples")
        if not 0.0 < self.ensemble_bootstrap_probability <= 1.0:
            raise ValueError("ensemble_bootstrap_probability must be in (0, 1]")
        if self.ensemble_max_grad_norm <= 0.0:
            raise ValueError("ensemble_max_grad_norm must be positive")
        if self.ensemble_updates_per_rollout <= 0:
            raise ValueError("ensemble_updates_per_rollout must be positive")
        if self.bonus_normalization_clip is not None and self.bonus_normalization_clip <= 0.0:
            raise ValueError("bonus_normalization_clip must be positive when provided")
        if self.freeze_encoder_after_updates is not None and self.freeze_encoder_after_updates < 0:
            raise ValueError("freeze_encoder_after_updates must be non-negative when provided")


@dataclass(frozen=True)
class TrainingConfig:
    """Top-level schedule, logging, checkpoint, and device settings."""

    total_environment_steps: int = 12_288_000
    device: DeviceType = DeviceType.AUTO
    log_interval_updates: int = 1
    checkpoint_interval_updates: int = 100
    output_directory: str = "runs/adventurer"
    tensorboard_directory: Optional[str] = None
    plot_directory: Optional[str] = None
    enable_tensorboard: bool = True
    save_plots: bool = True
    plot_interval_updates: int = 1
    score_smoothing_window: int = 10
    figure_width: float = 8.0
    figure_height: float = 5.0
    figure_dpi: int = 150
    use_intrinsic_reward: bool = True
    resettable: bool = False
    episodic_memory_size: int = 10

    def __post_init__(self) -> None:
        """Validate training duration and operational intervals.

        Input: Training schedule, output path, device policy, and intrinsic
            reward switch.
        Output: No value; raises ``ValueError`` for invalid schedules.
        Mathematical meaning: The intrinsic advantage scale is configured once
            in ``PPOConfig.intrinsic_advantage_coefficient``.
        """
        if self.total_environment_steps <= 0:
            raise ValueError("total_environment_steps must be positive")
        if self.log_interval_updates <= 0 or self.checkpoint_interval_updates <= 0:
            raise ValueError("logging and checkpoint intervals must be positive")
        if self.plot_interval_updates <= 0:
            raise ValueError("plot_interval_updates must be positive")
        if self.score_smoothing_window <= 0:
            raise ValueError("score_smoothing_window must be positive")
        if self.figure_width <= 0.0 or self.figure_height <= 0.0:
            raise ValueError("figure dimensions must be positive")
        if self.figure_dpi <= 0:
            raise ValueError("figure_dpi must be positive")
        if self.episodic_memory_size <= 0:
            raise ValueError("episodic_memory_size must be positive")


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete validated configuration for one Adventurer experiment."""

    seed: SeedConfig = SeedConfig()
    environment: EnvironmentConfig = EnvironmentConfig()
    ppo: PPOConfig = PPOConfig()
    bigan: BiGANConfig = BiGANConfig()
    novelty: NoveltyConfig = NoveltyConfig()
    metric_eme: MetricEMEConfig = MetricEMEConfig()
    training: TrainingConfig = TrainingConfig()

    def __post_init__(self) -> None:
        """Validate relationships spanning multiple configuration sections.

        Input: Fully constructed component configurations.
        Output: No value; raises ``ValueError`` for incompatible components.
        Mathematical meaning: The rollout schedule must contain an integer
            number of PPO updates, and the BiGAN input dimensions must be
            compatible with the declared environment observation.
        """
        samples_per_update = (
            self.environment.num_parallel_envs * self.ppo.rollout_steps
        )
        if self.training.total_environment_steps % samples_per_update != 0:
            raise ValueError(
                "total_environment_steps must be divisible by "
                "num_parallel_envs * rollout_steps"
            )
        if self.environment.discrete_actions and self.environment.action_dim <= 0:
            raise ValueError("discrete action spaces require a positive action_dim")
        if self.metric_eme.enabled and self.novelty.novelty_type != "state":
            raise ValueError(
                "metric_eme replaces the state novelty bonus and cannot be combined "
                "with novelty_type='transition'"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible nested representation of the configuration.

        Input: The initialized experiment configuration.
        Output: A new dictionary containing all dataclass fields, with enum
            values represented by their strings.
        Mathematical meaning: This is not a learning operation; it records the
            exact parameter point at which the objective was optimized.
        """
        result = asdict(self)
        result["training"]["device"] = self.training.device.value
        return result
