"""Command-line entry point for Adventurer experiments.

This module assembles the validated configuration, environment adapter,
logger, and end-to-end trainer. Algorithmic implementation remains in the
modular agent, BiGAN, exploration, and trainer modules.

Master's thesis experiment
--------------------------
``--method state`` runs the Adventurer BiGAN state-novelty baseline and
``--method transition`` runs the BiGAN action-conditioned transition novelty
``N_T = ||f(E(s_t),a_t) - E(s_{t+1})||_2``. Both use the identical environment,
preprocessing, PPO configuration, and training budget; only the intrinsic
signal differs. ``--method`` is mutually exclusive with the legacy selectors
(``--novelty``, ``--novelty-type``) and with the legacy EME/metric flags, so an
ambiguous or unfair comparison cannot be launched by accident.

Every run writes ``config.json`` (experimental configuration) and
``run_info.json`` (execution/reproducibility metadata: git commit, library
versions, resolved device, key settings).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from config import (
    MASTER_METHODS,
    DeviceType,
    EnvironmentConfig,
    ExperimentConfig,
    MetricEMEConfig,
    TrainingConfig,
)
from environments.adapter import SingleEnvironmentAdapter, make_gymnasium_environment
from environments.vector_adapter import VectorEnvironmentAdapter
from trainer import AdventurerTrainer
from evaluation.comparison import save_game_score_comparison
from utils.seed import derive_seed
from utils.logger import ExperimentLogger
from utils.run_metadata import resolve_device_label, write_run_info


def _parse_boolean(value: str) -> bool:
    """Parse a permissive command-line boolean.

    Input: String such as ``True``, ``false``, ``1``, or ``no``.
    Output: Corresponding Python boolean.
    Mathematical meaning: None; selects a discrete experiment variant.
    """
    normalized = str(value).strip().lower()
    if normalized in {"true", "t", "yes", "y", "1"}:
        return True
    if normalized in {"false", "f", "no", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


NOVELTY_ALIASES = {
    # V1 baseline: Adventurer reconstruction novelty B(s).
    "bigan": ("state", False),
    "state": ("state", False),
    # Adventurer transition novelty T(s,a,s').
    "transition": ("transition", False),
    # V2/V3: metric bonus ||E(s_t)-E(s_{t+1})||_p, optionally EME-scaled.
    "latent_discrepancy": ("state", True),
}

# Master's thesis experiment (--method). Only two intrinsic-reward signals are
# compared, and both use the identical environment, preprocessing, PPO
# configuration, and training budget:
#
#   state      -> Adventurer BiGAN state novelty B(s)                (baseline)
#   transition -> BiGAN latent transition novelty
#                 N_T = ||f(E(s_t),a_t) - E(s_{t+1})||_2              (proposed)
#
# The legacy transition novelty (--novelty transition) is a different, older
# implementation and is deliberately NOT selected by --method transition.
METHOD_TO_NOVELTY = {
    "state": ("state", "legacy", False),
    "transition": ("transition", "master_l2", False),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line experiment settings.

    Input: Command-line arguments from the current process.
    Output: Namespace containing environment, seed, duration, device, output,
        and intrinsic-reward options.
    Mathematical meaning: Selects the experimental parameter point used by the
        stochastic PPO/BiGAN optimization process.
    """
    parser = argparse.ArgumentParser(description="Train Adventurer with PPO and BiGAN exploration")
    parser.add_argument(
        "--environment-id",
        "--env",
        dest="environment_id",
        default="CartPole-v1",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-parallel-envs", type=int, default=96)
    parser.add_argument("--total-environment-steps", type=int, default=12_288_000)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--output-directory", type=Path, default=Path("runs/adventurer"))
    parser.add_argument("--tensorboard-directory", type=Path, default=None)
    parser.add_argument("--plot-directory", type=Path, default=None)
    parser.add_argument("--disable-tensorboard", action="store_true")
    parser.add_argument("--disable-plots", action="store_true")
    parser.add_argument("--plot-interval-updates", type=int, default=1)
    parser.add_argument("--score-smoothing-window", type=int, default=10)
    parser.add_argument("--figure-width", type=float, default=8.0)
    parser.add_argument("--figure-height", type=float, default=5.0)
    parser.add_argument("--figure-dpi", type=int, default=150)
    parser.add_argument("--device", choices=[item.value for item in DeviceType], default="auto")
    parser.add_argument("--disable-intrinsic-reward", action="store_true")
    parser.add_argument(
        "--intrinsic-advantage-coefficient",
        "--intrinsic-reward-mix",
        dest="intrinsic_advantage_coefficient",
        type=float,
        default=0.3,
        help="single beta applied to intrinsic advantage after independent GAE",
    )
    parser.add_argument(
        "--method",
        choices=list(MASTER_METHODS),
        default=None,
        help=(
            "Master's thesis experiment selector: 'state' runs the Adventurer "
            "BiGAN state-novelty baseline, 'transition' runs the BiGAN "
            "action-conditioned transition novelty "
            "N_T = ||f(E(s_t),a_t) - E(s_{t+1})||_2. Mutually exclusive with "
            "--novelty/--novelty-type and with the legacy EME/metric flags."
        ),
    )
    parser.add_argument("--novelty-type", choices=["state", "transition"], default=None)
    parser.add_argument(
        "--novelty",
        choices=sorted(NOVELTY_ALIASES),
        default=None,
        help=(
            "experiment variant: 'bigan' reproduces Adventurer novelty, "
            "'latent_discrepancy' uses the BiGAN latent metric bonus, "
            "'transition' uses latent forward-model novelty"
        ),
    )
    parser.add_argument(
        "--eme",
        type=_parse_boolean,
        nargs="?",
        const=True,
        default=None,
        help="scale the latent bonus by the EME ensemble reward variance zeta(r)",
    )
    parser.add_argument("--ensemble_K", "--ensemble-size", dest="ensemble_size", type=int, default=5)
    parser.add_argument(
        "--max-reward-scaling",
        dest="max_reward_scaling",
        type=float,
        default=5.0,
        help="upper clamp M applied to zeta(r)",
    )
    parser.add_argument(
        "--eme-mode",
        choices=["clamped", "normalised", "normalized"],
        default="clamped",
        help=(
            "'clamped' is EME as published, min(max(zeta,1),M); 'normalised' "
            "divides zeta by its running mean, which stays informative when "
            "sparse rewards drive the raw variance far below one"
        ),
    )
    parser.add_argument(
        "--zeta-momentum",
        type=float,
        default=0.99,
        help="EMA momentum of E[zeta] used by --eme-mode normalised",
    )
    parser.add_argument("--latent-norm", choices=["L1", "L2"], default="L2")
    parser.add_argument("--ensemble-input", choices=["latent", "observation"], default="latent")
    parser.add_argument("--ensemble-hidden-dim", type=int, default=256)
    parser.add_argument("--ensemble-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--ensemble-batch-size", type=int, default=64)
    parser.add_argument("--ensemble-min-buffer-size", type=int, default=128)
    parser.add_argument("--ensemble-bootstrap-probability", type=float, default=0.5)
    parser.add_argument("--ensemble-updates-per-rollout", type=int, default=64)
    parser.add_argument(
        "--freeze-encoder-after-updates",
        type=int,
        default=None,
        help="freeze the BiGAN encoder once this many PPO updates have started",
    )
    parser.add_argument(
        "--disable-latent-normalization",
        action="store_true",
        help="measure d_t on raw encoder codes instead of unit-sphere codes",
    )
    parser.add_argument(
        "--episodic-count-scaling",
        type=_parse_boolean,
        nargs="?",
        const=True,
        default=False,
        help=(
            "habituate the metric bonus with the RIDE/NovelD episodic count "
            "factor 1/sqrt(N_ep(s_t+1)); this is the documented V2 ablation"
        ),
    )
    parser.add_argument("--episodic-count-resolution", type=int, default=32)
    parser.add_argument(
        "--metric-learning",
        choices=["none", "eme"],
        default="none",
        help=(
            "'eme' trains EME's learned metric d_phi (Eq. 9: value difference "
            "+ bootstrapped distance + policy KL) on frozen encoder codes and "
            "uses it as the bonus distance; it also freezes the encoder"
        ),
    )
    parser.add_argument("--metric-learning-updates-per-rollout", type=int, default=32)
    parser.add_argument("--metric-learning-batch-size", type=int, default=256)
    parser.add_argument(
        "--ensemble-validation-fraction",
        type=float,
        default=0.05,
        help="fraction of (s,a,r) triples held out for the ensemble validation loss",
    )
    parser.add_argument(
        "--bonus-normalization-clip",
        type=float,
        default=5.0,
        help="symmetric bound on the Eq. (5)-normalized bonus",
    )
    parser.add_argument(
        "--disable-bonus-normalization",
        action="store_true",
        help="feed the raw bonus b_t to PPO instead of Eq. (5) reward-scale normalization",
    )
    parser.add_argument("--state-alpha", type=float, default=0.9)
    parser.add_argument("--transition-alpha", type=float, default=0.9)
    parser.add_argument(
        "--transition-batch-size",
        dest="transition_batch_size",
        type=int,
        default=64,
        help=(
            "minibatch size B of the Master's forward-model MSE objective; the "
            "thesis experiment uses the default 64 for both methods, and only "
            "tiny smoke configurations need a smaller value"
        ),
    )
    parser.add_argument("--compare-baseline-directory", type=Path, default=None)
    parser.add_argument("--compare-transition-directory", type=Path, default=None)
    parser.add_argument("--comparison-output-directory", type=Path, default=None)
    parser.add_argument("--comparison-smoothing-window", type=int, default=10)
    parser.add_argument(
        "--resettable",
        action="store_true",
        help="enable Algorithm 2 episodic-memory state restoration",
    )
    parser.add_argument("--episodic-memory-size", type=int, default=10)
    parser.add_argument("--bigan-batch-size", type=int, default=64)
    return _validate_method_selection(parser.parse_args(), parser)


def _validate_method_selection(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> argparse.Namespace:
    """Reject ambiguous or unfair combinations of the Master's ``--method`` flag.

    Input: Parsed arguments and the parser used for error reporting.
    Output: The same namespace when the selection is unambiguous.
    Mathematical meaning: Guarantees that the two compared conditions differ
        only in the intrinsic-reward signal, never in the novelty variant or in
        an additional legacy bonus term.
    """
    if args.method is None:
        return args
    if args.novelty is not None:
        parser.error(
            f"--method {args.method} cannot be combined with --novelty {args.novelty}: "
            "--method selects the Master's thesis variant, --novelty selects the "
            "legacy variant (for example '--novelty transition' is the old L1 + "
            "discriminator-feature transition novelty, not the Master's method)"
        )
    if args.novelty_type is not None:
        parser.error(
            f"--method {args.method} cannot be combined with --novelty-type "
            f"{args.novelty_type}: use one selector only"
        )
    legacy_bonus_flags = []
    if args.eme:
        legacy_bonus_flags.append("--eme")
    if args.metric_learning != "none":
        legacy_bonus_flags.append("--metric-learning")
    if args.episodic_count_scaling:
        legacy_bonus_flags.append("--episodic-count-scaling")
    if legacy_bonus_flags:
        parser.error(
            f"--method {args.method} cannot be combined with "
            f"{', '.join(legacy_bonus_flags)}: the Master's thesis experiment "
            "compares BiGAN state novelty with BiGAN transition novelty only, "
            "with no EME, ensemble, adaptive, or visit-count bonus"
        )
    return args


def _space_dimensions(environment: object) -> tuple[tuple[int, ...], int, bool]:
    """Extract observation and action dimensions from a wrapped environment.

    Input: Initialized single-environment adapter.
    Output: ``(observation_shape, action_dim, discrete_actions)``.
    Mathematical meaning: Defines the state domain and action support of the
        policy, critic, encoder, generator, and discriminator.
    """
    if isinstance(environment, VectorEnvironmentAdapter):
        source_environment = environment.environments[0]
    else:
        source_environment = environment
    # Callers normally hand in the pipeline's single-environment wrapper, which
    # keeps the gymnasium environment it wraps in ``environment``; a raw
    # gymnasium-style environment (for example the Unity adapter returned by
    # ``make_noisy_tv_environment("unity")``) is accepted as well.
    inner_environment = getattr(source_environment, "environment", source_environment)
    observation_space = getattr(inner_environment, "observation_space", None)
    action_space = getattr(inner_environment, "action_space", None)
    if observation_space is None or action_space is None:
        raise TypeError("environment must expose observation_space and action_space")
    observation_shape = getattr(source_environment, "observation_shape", None)
    if observation_shape is None and hasattr(observation_space, "shape"):
        observation_shape = tuple(observation_space.shape)
    if not observation_shape:
        raise ValueError("observation space shape cannot be empty")
    if hasattr(action_space, "n"):
        return observation_shape, int(action_space.n), True
    action_shape = getattr(action_space, "shape", None)
    if action_shape is None or len(action_shape) != 1:
        raise ValueError("continuous action space must expose one-dimensional shape")
    return observation_shape, int(action_shape[0]), False


def build_config(
    args: argparse.Namespace,
    environment: object,
) -> ExperimentConfig:
    """Build a validated configuration using discovered environment spaces.

    Input: Parsed command-line namespace and initialized environment adapter.
    Output: Complete ``ExperimentConfig`` with dimensions matching the actual
        environment and CLI-selected training settings.
    Mathematical meaning: Couples PPO/BiGAN function domains to the MDP while
        preserving all remaining defaults and objective coefficients.
    """
    observation_shape, action_dim, discrete_actions = _space_dimensions(environment)
    base = ExperimentConfig()
    environment_config = replace(
        base.environment,
        observation_shape=observation_shape,
        action_dim=action_dim,
        discrete_actions=discrete_actions,
        num_parallel_envs=args.num_parallel_envs,
    )
    ppo_config = replace(
        base.ppo,
        rollout_steps=args.rollout_steps,
        minibatch_size=args.minibatch_size,
        intrinsic_advantage_coefficient=args.intrinsic_advantage_coefficient,
    )
    tensorboard_directory = (
        str(args.tensorboard_directory)
        if args.tensorboard_directory is not None
        else str(args.output_directory / "tensorboard")
    )
    plot_directory = (
        str(args.plot_directory)
        if args.plot_directory is not None
        else str(args.output_directory / "figures")
    )
    training_config = replace(
        base.training,
        total_environment_steps=args.total_environment_steps,
        device=DeviceType(args.device),
        output_directory=str(args.output_directory),
        tensorboard_directory=tensorboard_directory,
        plot_directory=plot_directory,
        enable_tensorboard=not args.disable_tensorboard,
        save_plots=not args.disable_plots,
        plot_interval_updates=args.plot_interval_updates,
        score_smoothing_window=args.score_smoothing_window,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        figure_dpi=args.figure_dpi,
        use_intrinsic_reward=not args.disable_intrinsic_reward,
        resettable=args.resettable,
        episodic_memory_size=args.episodic_memory_size,
    )
    if args.method is not None:
        # Master's thesis selection: exactly one intrinsic-reward signal, no
        # legacy metric/EME bonus.
        novelty_type, transition_variant, metric_enabled = METHOD_TO_NOVELTY[args.method]
    else:
        novelty_type, metric_enabled = NOVELTY_ALIASES.get(
            args.novelty, (args.novelty_type or "state", False)
        )
        transition_variant = "legacy"
    novelty_config = replace(
        base.novelty,
        novelty_type=novelty_type,
        transition_variant=transition_variant,
        alpha=args.state_alpha,
        transition_alpha=args.transition_alpha,
        transition_batch_size=args.transition_batch_size,
    )
    # V2 is the latent metric without ensemble scaling (optionally habituated
    # with episodic counts); V3/V4 add the EME variance scaling modes.
    ensemble_scaling = bool(args.eme) and args.method is None
    metric_eme_config = replace(
        base.metric_eme,
        enabled=metric_enabled or ensemble_scaling,
        ensemble_scaling=ensemble_scaling,
        ensemble_size=args.ensemble_size,
        # "normalized" is accepted as a spelling variant of "normalised".
        eme_mode="clamped" if args.eme_mode == "clamped" else "normalised",
        zeta_momentum=args.zeta_momentum,
        max_reward_scaling=args.max_reward_scaling,
        latent_norm=args.latent_norm,
        ensemble_input=args.ensemble_input,
        ensemble_hidden_dim=args.ensemble_hidden_dim,
        ensemble_learning_rate=args.ensemble_learning_rate,
        ensemble_batch_size=args.ensemble_batch_size,
        ensemble_min_buffer_size=args.ensemble_min_buffer_size,
        ensemble_bootstrap_probability=args.ensemble_bootstrap_probability,
        ensemble_updates_per_rollout=args.ensemble_updates_per_rollout,
        normalize_bonus=not args.disable_bonus_normalization,
        bonus_normalization_clip=args.bonus_normalization_clip,
        freeze_encoder_after_updates=args.freeze_encoder_after_updates,
        normalize_latent_embeddings=not args.disable_latent_normalization,
        episodic_count_scaling=args.episodic_count_scaling,
        episodic_count_resolution=args.episodic_count_resolution,
        metric_learning=args.metric_learning,
        metric_learning_updates_per_rollout=args.metric_learning_updates_per_rollout,
        metric_learning_batch_size=args.metric_learning_batch_size,
        ensemble_validation_fraction=args.ensemble_validation_fraction,
    )
    bigan_config = replace(base.bigan, batch_size=args.bigan_batch_size)
    seed_config = replace(base.seed, seed=args.seed)
    return replace(
        base,
        seed=seed_config,
        environment=environment_config,
        ppo=ppo_config,
        bigan=bigan_config,
        novelty=novelty_config,
        metric_eme=metric_eme_config,
        training=training_config,
    )


def make_training_environment(environment_id: str, num_parallel_envs: int, seed: int) -> VectorEnvironmentAdapter:
    """Construct N independently seeded environments for batched rollouts.

    Input: Registered environment ID, positive parallel-environment count, and
        base seed.
    Output: ``VectorEnvironmentAdapter`` containing N child environments.
    Mathematical meaning: Constructs the product MDP used to collect N
        transitions per environment step.
    """
    if num_parallel_envs <= 0:
        raise ValueError("num_parallel_envs must be positive")
    environments = [
        make_gymnasium_environment(environment_id)
        for _ in range(num_parallel_envs)
    ]
    vector_environment = VectorEnvironmentAdapter(environments)
    vector_environment.seed(seed)
    return vector_environment


def main() -> int:
    """Create and run one Adventurer experiment.

    Input: Process command-line arguments.
    Output: Exit status ``0`` after successful training.
    Mathematical meaning: Executes repeated environment sampling, novelty
        reward construction, GAE estimation, PPO optimization, and BiGAN
        optimization for the configured environment-step budget.
    """
    args = parse_args()
    environment = make_training_environment(
        args.environment_id,
        args.num_parallel_envs,
        args.seed,
    )
    try:
        config = build_config(args, environment)
        output_directory = Path(config.training.output_directory)
        with ExperimentLogger(output_directory, config) as logger:
            # Execution metadata; config.json (written by the logger) remains
            # the record of the experimental configuration.
            write_run_info(
                output_directory,
                config,
                method=args.method,
                environment_id=args.environment_id,
                device_label=resolve_device_label(config.training.device.value),
            )
            trainer = AdventurerTrainer(environment, config, logger)
            trainer.train()
            trainer.close()
        if args.compare_baseline_directory is not None and args.compare_transition_directory is not None:
            comparison_output = args.comparison_output_directory or (args.output_directory / "comparison")
            save_game_score_comparison(
                args.compare_baseline_directory / "metrics.jsonl",
                args.compare_transition_directory / "metrics.jsonl",
                comparison_output,
                smoothing_window=args.comparison_smoothing_window,
                dpi=300,
            )
    except Exception:
        environment.close()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
