"""Vectorized Adventurer training pipeline.

The trainer implements separate extrinsic/intrinsic reward streams, two-stream
GAE, PPO with two critics, BiGAN novelty, optional resettable episodic memory,
and N parallel environments. With the default Atari configuration it collects
96*128=12,288 transitions per PPO rollout.

Three intrinsic-reward paths are available and exactly one is active per run:

* ``novelty_type="state"`` -- Adventurer's BiGAN reconstruction novelty
  ``B(s)``. This is the Master's thesis baseline (``main.py --method state``).
* ``novelty_type="transition"`` with ``transition_variant="master_l2"`` -- the
  Master's thesis method ``N_T = ||f(E(s_t),a_t) - E(s_{t+1})||_2`` from
  :mod:`exploration.master_transition` (``main.py --method transition``). It
  executes no EME/metric code and uses a cached latent transition buffer.
* ``novelty_type="transition"`` with ``transition_variant="legacy"`` (default)
  -- the historical transition novelty, preserved unchanged for backward
  compatibility (``main.py --novelty transition``).

The legacy metric/EME bonus (``metric_eme.enabled``) and the resettable
episodic memory (``training.resettable``) are off by default and are not part
of the Master's thesis experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor

from agent.actor import ActorNetwork
from agent.advantage import (
    TwoStreamAdvantageEstimate,
    compute_two_stream_returns_and_advantages,
)
from agent.critic import DualCriticNetwork
from agent.ppo import PPOAgent, PPOUpdateMetrics
from agent.rollout_buffer import RolloutBuffer
from bigan.discriminator import BiGANDiscriminator
from bigan.encoder import BiGANEncoder
from bigan.generator import BiGANGenerator
from bigan.trainer import BiGANTrainer, BiGANUpdateMetrics
from config import ExperimentConfig
from environments.vector_adapter import VectorEnvironmentAdapter
from exploration.ensemble_scaling import EnsembleRewardVariance, EnsembleUpdateMetrics
from exploration.episodic_count import EpisodicLatentCountScaling
from exploration.episodic_memory import EpisodicMemory
from exploration.eme_metric import EMEMetricLearner
from exploration.intrinsic_reward import IntrinsicRewardPipeline, MetricIntrinsicReward
from exploration.master_transition import (
    LatentTransitionBuffer,
    LatentTransitionNovelty,
    LatentTransitionTrainer,
    MasterTransitionUpdateMetrics,
)
from exploration.novelty import NoveltyEstimator
from exploration.transition_novelty import (
    LatentForwardModel,
    TransitionNoveltyEstimator,
    TransitionNoveltyTrainer,
    TransitionUpdateMetrics,
)
from utils.logger import ExperimentLogger
from utils.replay import ReplayBuffer
from utils.transition_replay import TransitionReplayBuffer
from utils.seed import create_torch_generator, derive_seed, seed_everything
from utils.visualization import TrainingPlotter


@dataclass(frozen=True)
class MetricEMEMetrics:
    """Diagnostics of the BiGAN-latent, EME-scaled exploration bonus."""

    latent_distance: float
    ensemble_variance: float
    mean_ensemble_variance: float
    bonus_scale: float
    bonus: float
    encoder_frozen: bool
    ensemble: Optional[EnsembleUpdateMetrics] = None
    # P4 instrumentation: detects metric blow-ups, dead scaling factors, and
    # under-trained ensembles directly from the training logs.
    latent_norm_mean: float = 0.0
    bonus_distance: float = 0.0
    episodic_scale_mean: Optional[float] = None
    variance_below_min_fraction: float = 0.0
    variance_above_max_fraction: float = 0.0
    latent_distance_p5: float = 0.0
    latent_distance_p50: float = 0.0
    latent_distance_p95: float = 0.0
    metric_learning_loss: Optional[float] = None


@dataclass(frozen=True)
class TrainingIterationMetrics:
    """Diagnostics from one vectorized rollout and update cycle."""

    update: int
    environment_steps: int
    extrinsic_reward: float
    intrinsic_reward: float
    total_reward: float
    pixel_novelty: float
    feature_novelty: float
    normalized_novelty: float
    ppo: PPOUpdateMetrics
    bigan: BiGANUpdateMetrics
    transition: Optional[Union[TransitionUpdateMetrics, MasterTransitionUpdateMetrics]] = None
    metric_eme: Optional[MetricEMEMetrics] = None


class AdventurerTrainer:
    """Train Adventurer with N parallel environments."""

    def __init__(
        self,
        environment: VectorEnvironmentAdapter,
        config: ExperimentConfig,
        logger: Optional[ExperimentLogger] = None,
    ) -> None:
        """Construct vectorized networks, buffers, and exploration modules.

        Input: Vector environment, complete configuration, and optional logger.
        Output: Initialized trainer with ``N`` parallel rollout streams.
        Mathematical meaning: Instantiates the product-MDP policy/value,
            BiGAN, and two independent reward-stream optimization processes.
        """
        self.environment = environment
        self.config = config
        self.device = self._resolve_device(config.training.device.value)
        self.num_envs = environment.num_envs
        if self.num_envs != config.environment.num_parallel_envs:
            raise ValueError("environment count does not match configuration")
        seed_everything(config.seed)
        self.environment.seed(config.seed.seed)
        self.logger = logger
        self.resettable = config.training.resettable
        self.episodic_memory = EpisodicMemory(config.training.episodic_memory_size)
        self.memory_generator = create_torch_generator(
            derive_seed(config.seed.seed, 0xE91), device="cpu"
        )
        if self.resettable and not self.environment.supports_state_restore():
            raise RuntimeError(
                "resettable=True requires simulator clone/restore support in every environment"
            )

        plot_directory = config.training.plot_directory
        if plot_directory is None:
            plot_directory = f"{config.training.output_directory}/figures"
        self.plotter = (
            TrainingPlotter(
                output_directory=plot_directory,
                smoothing_window=config.training.score_smoothing_window,
                interval_updates=config.training.plot_interval_updates,
                figure_width=config.training.figure_width,
                figure_height=config.training.figure_height,
                dpi=config.training.figure_dpi,
            )
            if config.training.save_plots
            else None
        )
        observation_shape = config.environment.observation_shape
        action_dim = config.environment.action_dim
        self.actor = ActorNetwork(
            observation_shape, action_dim, config.environment.discrete_actions
        ).to(self.device)
        self.critic = DualCriticNetwork(observation_shape).to(self.device)
        total_updates = config.training.total_environment_steps // (
            config.ppo.rollout_steps * self.num_envs
        )
        if total_updates <= 0:
            raise ValueError("training budget must contain at least one complete vector rollout")
        self.ppo = PPOAgent(
            self.actor,
            self.critic,
            config.ppo,
            total_updates,
            beta=config.ppo.intrinsic_advantage_coefficient,
            device=self.device,
        )
        self.encoder = BiGANEncoder(
            observation_shape,
            config.bigan.latent_dim,
            config.bigan.feature_dim,
            config.bigan.hidden_dim,
        )
        self.generator = BiGANGenerator(
            observation_shape,
            config.bigan.latent_dim,
            config.bigan.feature_dim,
            config.bigan.hidden_dim,
        )
        self.discriminator = BiGANDiscriminator(
            observation_shape,
            config.bigan.latent_dim,
            config.bigan.feature_dim,
            config.bigan.hidden_dim,
        )
        self.bigan = BiGANTrainer(
            self.encoder,
            self.generator,
            self.discriminator,
            config.bigan,
            observation_shape,
            self.device,
        )
        self.novelty = None
        self.transition_novelty = None
        self.transition_trainer = None
        self.transition_replay_buffer = None
        # Master's thesis transition novelty (--method transition). It is a
        # separate, explicitly selected variant; the legacy transition novelty
        # below is preserved unchanged.
        self.master_transition_enabled = (
            config.novelty.novelty_type == "transition"
            and config.novelty.transition_variant == "master_l2"
        )
        self.master_transition_novelty: Optional[LatentTransitionNovelty] = None
        self.master_transition_trainer: Optional[LatentTransitionTrainer] = None
        self.master_transition_buffer: Optional[LatentTransitionBuffer] = None
        if config.novelty.novelty_type == "state":
            self.novelty = NoveltyEstimator(
                self.encoder,
                self.generator,
                self.discriminator,
                config.novelty.alpha,
                config.novelty.normalization_epsilon,
                device=self.device,
            )
        elif self.master_transition_enabled:
            # N_T = ||f(E(s_t), a_t) - E(s_{t+1})||_2 with an MSE-trained latent
            # forward model. Same encoder, same Equation (5) normalization, and
            # same PPO configuration as the state-novelty baseline; no generator
            # or discriminator term is evaluated and no EME component is used.
            master_forward_model = LatentForwardModel(
                config.bigan.latent_dim,
                action_dim,
                config.novelty.transition_hidden_dim,
                config.environment.discrete_actions,
            )
            self.master_transition_novelty = LatentTransitionNovelty(
                self.encoder,
                master_forward_model,
                config.novelty.normalization_epsilon,
                device=self.device,
            )
            self.master_transition_trainer = LatentTransitionTrainer(
                master_forward_model,
                learning_rate=config.novelty.transition_learning_rate,
                batch_size=config.novelty.transition_batch_size,
                max_grad_norm=config.novelty.transition_max_grad_norm,
                update_epochs=config.novelty.transition_update_epochs,
                device=self.device,
            )
            # Cached (z_t, a_t, z_{t+1}) triples instead of raw observations:
            # a raw (s, a, s') buffer of one rollout costs ~10 GB at the
            # 96x128 Montezuma configuration, the latent buffer ~12 MB.
            self.master_transition_buffer = LatentTransitionBuffer(
                max(
                    config.ppo.rollout_steps * self.num_envs,
                    config.novelty.transition_batch_size,
                ),
                config.bigan.latent_dim,
                () if config.environment.discrete_actions else (action_dim,),
                self.device,
                torch.long if config.environment.discrete_actions else torch.float32,
            )
        else:
            forward_model = LatentForwardModel(
                config.bigan.latent_dim,
                action_dim,
                config.novelty.transition_hidden_dim,
                config.environment.discrete_actions,
            )
            self.transition_novelty = TransitionNoveltyEstimator(
                self.encoder,
                self.generator,
                self.discriminator,
                forward_model,
                config.novelty.transition_alpha,
                config.novelty.normalization_epsilon,
                device=self.device,
            )
            self.transition_trainer = TransitionNoveltyTrainer(
                self.transition_novelty,
                config.novelty.transition_learning_rate,
                config.novelty.transition_update_epochs,
                config.novelty.transition_max_grad_norm,
            )
            self.transition_replay_buffer = TransitionReplayBuffer(
                max(config.ppo.rollout_steps * self.num_envs, config.bigan.batch_size),
                observation_shape,
                () if config.environment.discrete_actions else (action_dim,),
                self.device,
                torch.float32,
                torch.long if config.environment.discrete_actions else torch.float32,
            )
        self.metric_eme_config = config.metric_eme
        self.metric_reward: Optional[MetricIntrinsicReward] = None
        self.reward_ensemble: Optional[EnsembleRewardVariance] = None
        self.episodic_counts: Optional[EpisodicLatentCountScaling] = None
        self.eme_metric_learner: Optional[EMEMetricLearner] = None
        if config.metric_eme.enabled:
            self._build_metric_exploration_bonus(observation_shape)
        self.reward_pipeline = IntrinsicRewardPipeline(config.novelty.clip_intrinsic_reward)
        self.rollout_buffer = RolloutBuffer(
            config.ppo.rollout_steps,
            self.num_envs,
            observation_shape,
            () if config.environment.discrete_actions else (action_dim,),
            self.device,
            torch.float32,
            torch.long if config.environment.discrete_actions else torch.float32,
        )
        self.replay_buffer = ReplayBuffer(
            max(config.ppo.rollout_steps * self.num_envs, config.bigan.batch_size),
            observation_shape,
            self.device,
            torch.float32,
        )
        self._observation: Optional[Tensor] = None
        self.episode_returns = torch.zeros(self.num_envs, dtype=torch.float64)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.long)
        self.environment_steps = 0

    def _build_metric_exploration_bonus(self, observation_shape: tuple) -> None:
        """Create the latent-discrepancy bonus and its EME reward ensemble.

        Input: Observation shape of the environment.
        Output: No value; ``self.metric_reward`` and, when EME scaling is on,
            ``self.reward_ensemble`` are constructed.
        Mathematical meaning: Instantiates
            ``b_t=||E(s_t)-E(s_{t+1})||_p * min(max(zeta(r),1),M)`` by pairing
            the BiGAN encoder metric with an ensemble estimate of the epistemic
            reward variance ``zeta(r)``.
        """
        settings = self.config.metric_eme
        action_dim = self.config.environment.action_dim
        discrete_actions = self.config.environment.discrete_actions
        if settings.ensemble_scaling:
            if settings.ensemble_input == "latent":
                state_feature_dim = self.config.bigan.latent_dim
                feature_extractor = self.encoder
            else:
                state_feature_dim = 1
                for dimension in observation_shape:
                    state_feature_dim *= int(dimension)
                feature_extractor = None
            # EME's reward models are g(s, a): the action terms are appended to
            # the state features inside the ensemble.
            self.reward_ensemble = EnsembleRewardVariance(
                input_dim=state_feature_dim,
                action_dim=action_dim,
                discrete_actions=discrete_actions,
                ensemble_size=settings.ensemble_size,
                hidden_dim=settings.ensemble_hidden_dim,
                learning_rate=settings.ensemble_learning_rate,
                buffer_capacity=settings.ensemble_buffer_capacity,
                batch_size=settings.ensemble_batch_size,
                min_buffer_size=settings.ensemble_min_buffer_size,
                bootstrap_probability=settings.ensemble_bootstrap_probability,
                validation_fraction=settings.ensemble_validation_fraction,
                max_grad_norm=settings.ensemble_max_grad_norm,
                feature_extractor=feature_extractor,
                device=self.device,
                generator=create_torch_generator(
                    derive_seed(self.config.seed.seed, 0xE3E), device=self.device
                ),
            )
        if settings.episodic_count_scaling:
            self.episodic_counts = EpisodicLatentCountScaling(
                self.num_envs, resolution=settings.episodic_count_resolution
            )
        self.metric_reward = MetricIntrinsicReward(
            encoder=self.encoder,
            ensemble=self.reward_ensemble,
            max_reward_scaling=settings.max_reward_scaling,
            min_reward_scaling=settings.min_reward_scaling,
            eme_mode=settings.eme_mode,
            zeta_momentum=settings.zeta_momentum,
            zeta_epsilon=settings.zeta_epsilon,
            norm=settings.latent_norm,
            normalize_latent=settings.normalize_latent_embeddings,
            episodic_counter=self.episodic_counts,
            normalize=settings.normalize_bonus,
            normalization_epsilon=self.config.novelty.normalization_epsilon,
            normalization_clip=settings.bonus_normalization_clip,
            clip_value=self.config.novelty.clip_intrinsic_reward,
            device=self.device,
        )
        if settings.metric_learning == "eme":
            # EME Eq. (9) regresses d_phi on encoder codes; the metric is only
            # well defined in a stationary embedding, so the encoder freezes
            # immediately (the BiGAN generator/discriminator keep training).
            self.eme_metric_learner = EMEMetricLearner(
                discrepancy=self.metric_reward.discrepancy,
                actor=self.actor,
                action_dim=action_dim,
                discrete_actions=discrete_actions,
                gamma=self.config.ppo.gamma,
                hidden_dim=settings.ensemble_hidden_dim,
                learning_rate=settings.ensemble_learning_rate,
                batch_size=settings.metric_learning_batch_size,
                max_grad_norm=settings.ensemble_max_grad_norm,
                device=self.device,
                generator=create_torch_generator(
                    derive_seed(self.config.seed.seed, 0x9E4), device=self.device
                ),
            )
            self.metric_reward.metric_head = self.eme_metric_learner.head
            if not self.bigan.encoder_frozen:
                self._freeze_encoder()
        if settings.freeze_encoder_after_updates == 0:
            self._freeze_encoder()

    def _freeze_encoder(self) -> None:
        """Freeze the BiGAN encoder so the latent metric stops drifting.

        Input: This trainer.
        Output: No value; the BiGAN trainer stops stepping the encoder.
        Mathematical meaning: Holds ``E_psi`` fixed so that the exploration
            bonus compares latent codes across time in one stationary space.
        """
        self.bigan.freeze_encoder()
        if self.metric_reward is not None:
            self.metric_reward.discrepancy.freeze()

    def _maybe_freeze_encoder(self, update_index: int) -> None:
        """Freeze the encoder once the configured pretraining budget elapses.

        Input: Zero-based index of the PPO update about to be executed.
        Output: No value; freezing happens at most once.
        Mathematical meaning: Separates a BiGAN pretraining phase, in which the
            metric is still learned, from an exploitation phase in which the
            metric is fixed.
        """
        threshold = self.config.metric_eme.freeze_encoder_after_updates
        if threshold is None or self.bigan.encoder_frozen:
            return
        if update_index >= threshold:
            self._freeze_encoder()

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        """Resolve automatic, CPU, or CUDA device selection.

        Input: Device policy string.
        Output: Usable torch device.
        Mathematical meaning: Selects the substrate for batched optimization.
        """
        if device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device(device_name)

    def _reset_if_needed(self) -> Tensor:
        """Reset all environments when the trainer has no active batch.

        Input: Trainer state.
        Output: Batched observation tensor ``[N,...]``.
        Mathematical meaning: Establishes the initial product-MDP state batch.
        """
        if self._observation is None:
            observations, _ = self.environment.reset(
                [derive_seed(self.config.seed.seed, index) for index in range(self.num_envs)]
            )
            self._observation = observations.to(self.device)
        return self._observation

    def _reset_done(self, observations: Tensor, done: Tensor) -> Tensor:
        """Reset completed environment slots, optionally from episodic memory.

        Input: Terminal next-observation batch and boolean done mask ``[N]``.
        Output: Next current-observation batch with completed slots reset.
        Mathematical meaning: Starts new trajectories in the product MDP while
            preserving active trajectories.
        """
        current = observations.clone()
        for index in torch.nonzero(done, as_tuple=False).flatten().tolist():
            memory_entry = (
                self.episodic_memory.sample_previous(self.memory_generator)
                if self.resettable
                else None
            )
            if memory_entry is not None:
                if memory_entry.environment_snapshot is None:
                    raise RuntimeError("episodic memory entry lacks simulator snapshot")
                current[index] = self.environment.restore_state(
                    index, memory_entry.environment_snapshot
                )
            else:
                current[index] = self.environment.reset_one(
                    index,
                    derive_seed(self.config.seed.seed, self.environment_steps + index),
                )
            self.episode_returns[index] = 0.0
            self.episode_lengths[index] = 0
        return current

    def _record_completed_episodes(self, done: Tensor) -> None:
        """Record scores and lengths for all episodes ending in one batch step.

        Input: Boolean done mask ``[N]``.
        Output: No value; completed scores are sent to plotter and logger.
        Mathematical meaning: Records empirical episodic extrinsic returns.
        """
        for index in torch.nonzero(done, as_tuple=False).flatten().tolist():
            score = float(self.episode_returns[index].item())
            length = int(self.episode_lengths[index].item())
            timestep = self.environment_steps + index
            if self.plotter is not None:
                self.plotter.record_episode(timestep, score, length)
            if self.logger is not None:
                self.logger.log(
                    timestep,
                    {"episode/game_score": score, "episode/length": length},
                )

    def _collect_rollout(self) -> Dict[str, float]:
        """Collect T batched transitions and preserve both reward streams.

        Input: Current trainer state and configured rollout horizon T.
        Output: Mean reward and novelty diagnostics across ``T*N`` samples.
        Mathematical meaning: Stores separate ``r^e``, ``r^i``, ``V^e``, and
            ``V^i`` trajectories for two-stream GAE.
        """
        self.rollout_buffer.reset()
        if self.resettable:
            self.episodic_memory.begin_epoch()
        sums = {
            "extrinsic": 0.0,
            "intrinsic": 0.0,
            "total": 0.0,
            "pixel": 0.0,
            "feature": 0.0,
            "normalized": 0.0,
            "latent_distance": 0.0,
            "ensemble_variance": 0.0,
            "bonus_scale": 0.0,
            "bonus": 0.0,
            "latent_norm": 0.0,
            "bonus_distance": 0.0,
            "episodic_scale": 0.0,
            "variance_below_min": 0.0,
            "variance_above_max": 0.0,
            "latent_distance_p5": 0.0,
            "latent_distance_p50": 0.0,
            "latent_distance_p95": 0.0,
        }
        for _ in range(self.config.ppo.rollout_steps):
            observations = self._reset_if_needed()
            actions, log_probabilities, extrinsic_values, intrinsic_values = self.ppo.select_action(observations)
            transition = self.environment.step(actions)
            next_observations = transition.observations.to(self.device)
            extrinsic_rewards = transition.rewards.to(self.device)
            if self.config.metric_eme.enabled:
                # Contribution 2: the reconstruction novelty B(s) is replaced by
                # b_t = d_t * min(max(zeta(r),1),M) [* 1/sqrt(N_ep(s_t+1))].
                assert self.metric_reward is not None
                novelty = self.metric_reward(
                    observations,
                    next_observations,
                    extrinsic_reward=extrinsic_rewards,
                    update_statistics=True,
                    done=transition.done,
                    actions=actions,
                )
                if self.reward_ensemble is not None:
                    # EME Eq. (8): store (s_t, a_t) with target r_{t+1}, the
                    # reward received for taking a_t in s_t.
                    self.reward_ensemble.add(
                        observations, extrinsic_rewards, actions=actions
                    )
                sums["latent_distance"] += float(novelty.latent_distance.detach().sum().cpu())
                sums["ensemble_variance"] += float(novelty.ensemble_variance.detach().sum().cpu())
                sums["bonus_scale"] += float(novelty.bonus_scale.detach().sum().cpu())
                sums["bonus"] += float(novelty.combined_score.detach().sum().cpu())
                sums["latent_norm"] += novelty.latent_norm_mean * self.num_envs
                sums["bonus_distance"] += float(novelty.bonus_distance.detach().sum().cpu())
                if novelty.episodic_scale is not None:
                    sums["episodic_scale"] += float(novelty.episodic_scale.detach().sum().cpu())
                sums["variance_below_min"] += (
                    novelty.variance_below_min_fraction * self.num_envs
                )
                sums["variance_above_max"] += (
                    novelty.variance_above_max_fraction * self.num_envs
                )
                sums["latent_distance_p5"] += novelty.latent_distance_p5 * self.num_envs
                sums["latent_distance_p50"] += novelty.latent_distance_p50 * self.num_envs
                sums["latent_distance_p95"] += novelty.latent_distance_p95 * self.num_envs
            elif self.config.novelty.novelty_type == "state":
                assert self.novelty is not None
                novelty = self.novelty(
                    next_observations,
                    update_statistics=True,
                    extrinsic_reward=extrinsic_rewards,
                )
            elif self.master_transition_enabled:
                # Master's thesis method: cache (z_t, a_t, z_{t+1}) with the
                # encoder representation available at collection time. The
                # latents are never re-encoded later.
                assert self.master_transition_novelty is not None
                novelty = self.master_transition_novelty(
                    observations,
                    actions,
                    next_observations,
                    extrinsic_rewards,
                )
                assert self.master_transition_buffer is not None
                self.master_transition_buffer.add_batch(
                    novelty.latent, actions, novelty.next_latent
                )
            else:
                assert self.transition_novelty is not None
                novelty = self.transition_novelty(
                    observations,
                    actions,
                    next_observations,
                    extrinsic_rewards,
                )
                assert self.transition_replay_buffer is not None
                self.transition_replay_buffer.add_batch(
                    observations.detach(), actions.detach(), next_observations.detach()
                )
            reward_result = self.reward_pipeline(extrinsic_rewards, novelty)
            if self.resettable:
                snapshots = self.environment.clone_states()
                for index in range(self.num_envs):
                    self.episodic_memory.add(
                        next_observations[index],
                        float(novelty.combined_score[index].item()),
                        snapshots[index],
                    )
            self.rollout_buffer.add(
                observations,
                actions,
                reward_result.extrinsic_reward,
                reward_result.intrinsic_reward,
                transition.terminated,
                transition.truncated,
                extrinsic_values,
                intrinsic_values,
                log_probabilities,
                next_observations,
            )
            self.replay_buffer.add_batch(next_observations.detach())
            done = transition.done
            self.episode_returns += extrinsic_rewards.detach().double().cpu()
            self.episode_lengths += 1
            self._record_completed_episodes(done.cpu())
            self.environment_steps += self.num_envs
            self._observation = self._reset_done(next_observations, done)
            sums["extrinsic"] += float(reward_result.extrinsic_reward.detach().sum().cpu())
            sums["intrinsic"] += float(reward_result.intrinsic_reward.detach().sum().cpu())
            sums["total"] += float((reward_result.extrinsic_reward + reward_result.intrinsic_reward).detach().sum().cpu())
            sums["pixel"] += float(novelty.pixel_error.detach().sum().cpu())
            sums["feature"] += float(novelty.feature_error.detach().sum().cpu())
            sums["normalized"] += float(novelty.normalized_score.detach().sum().cpu())
        self.rollout_buffer.set_final_observation(self._reset_if_needed())
        denominator = float(self.config.ppo.rollout_steps * self.num_envs)
        return {key: value / denominator for key, value in sums.items()}

    def _update_metric_exploration_bonus(
        self,
        summary: Dict[str, float],
        tensors: Optional[Dict[str, Tensor]] = None,
    ) -> Optional[MetricEMEMetrics]:
        """Fit the reward ensemble and summarize the bonus for one rollout.

        Input: Rollout means produced by ``_collect_rollout`` and, when the EME
            learned metric is enabled, the rollout transition tensors used to
            regress ``d_phi``.
        Output: ``MetricEMEMetrics`` when the metric bonus is enabled, else
            ``None``.
        Mathematical meaning: Performs stochastic gradient steps on the ``K``
            reward regressors (and on EME's learned metric when configured) so
            ``zeta(r)`` contracts where reward structure has been learned, and
            reports the mean latent distance, variance, clamped scale, and
            bonus over the ``T*N`` collected transitions.
        """
        if not self.config.metric_eme.enabled:
            return None
        ensemble_metrics: Optional[EnsembleUpdateMetrics] = None
        if self.reward_ensemble is not None:
            for _ in range(self.config.metric_eme.ensemble_updates_per_rollout):
                ensemble_metrics = self.reward_ensemble.update()
        metric_learning_loss: Optional[float] = None
        if self.eme_metric_learner is not None:
            if tensors is None:
                raise ValueError("EME metric learning requires the rollout tensors")
            for _ in range(self.config.metric_eme.metric_learning_updates_per_rollout):
                metric_learning_loss = self.eme_metric_learner.update(
                    observations=tensors["observations"],
                    actions=tensors["actions"],
                    extrinsic_rewards=tensors["extrinsic_rewards"],
                    terminated=tensors["terminated"],
                    ensemble=self.reward_ensemble,
                )
        assert self.metric_reward is not None
        episodic_scale_mean: Optional[float] = None
        if self.episodic_counts is not None:
            episodic_scale_mean = summary["episodic_scale"]
        return MetricEMEMetrics(
            latent_distance=summary["latent_distance"],
            ensemble_variance=summary["ensemble_variance"],
            mean_ensemble_variance=self.metric_reward.mean_zeta,
            bonus_scale=summary["bonus_scale"],
            bonus=summary["bonus"],
            encoder_frozen=bool(self.bigan.encoder_frozen),
            ensemble=ensemble_metrics,
            latent_norm_mean=summary["latent_norm"],
            bonus_distance=summary["bonus_distance"],
            episodic_scale_mean=episodic_scale_mean,
            variance_below_min_fraction=summary["variance_below_min"],
            variance_above_max_fraction=summary["variance_above_max"],
            latent_distance_p5=summary["latent_distance_p5"],
            latent_distance_p50=summary["latent_distance_p50"],
            latent_distance_p95=summary["latent_distance_p95"],
            metric_learning_loss=metric_learning_loss,
        )

    @staticmethod
    def _metric_eme_log_entries(metrics: Optional[MetricEMEMetrics]) -> Dict[str, object]:
        """Return the loggable scalars of the metric exploration bonus.

        Input: Optional metric-bonus diagnostics for one update.
        Output: Mapping of log tags to values; empty when the bonus is off.
        Mathematical meaning: Exposes ``||E(s_t)-E(s_{t+1})||_p``, ``zeta(r)``,
            the clamped scale, their product ``b_t``, and the health checks
            (encoder-code norm, clamp-binding fractions, distance percentiles,
            episodic habituation, held-out ensemble loss, metric-learning
            loss), so a degenerate metric or scaling factor is visible in the
            logs immediately.
        """
        if metrics is None:
            return {}
        entries: Dict[str, object] = {
            "intrinsic/latent_distance": metrics.latent_distance,
            "intrinsic/ensemble_variance": metrics.ensemble_variance,
            "intrinsic/mean_ensemble_variance": metrics.mean_ensemble_variance,
            "intrinsic/bonus_scale": metrics.bonus_scale,
            "intrinsic/bonus": metrics.bonus,
            "intrinsic/encoder_frozen": float(metrics.encoder_frozen),
            "intrinsic/latent_norm": metrics.latent_norm_mean,
            "intrinsic/bonus_distance": metrics.bonus_distance,
            "intrinsic/variance_below_min_fraction": metrics.variance_below_min_fraction,
            "intrinsic/variance_above_max_fraction": metrics.variance_above_max_fraction,
            "intrinsic/latent_distance_p5": metrics.latent_distance_p5,
            "intrinsic/latent_distance_p50": metrics.latent_distance_p50,
            "intrinsic/latent_distance_p95": metrics.latent_distance_p95,
            "ensemble": metrics.ensemble,
        }
        if metrics.episodic_scale_mean is not None:
            entries["intrinsic/episodic_scale"] = metrics.episodic_scale_mean
        if metrics.metric_learning_loss is not None:
            entries["intrinsic/metric_learning_loss"] = metrics.metric_learning_loss
        return entries

    def train(self) -> List[TrainingIterationMetrics]:
        """Run vectorized collection, two-stream GAE, PPO, and BiGAN updates.

        Input: No arguments; uses immutable experiment configuration.
        Output: Per-update diagnostics.
        Mathematical meaning: Computes independent ``A^e`` and ``A^i`` over
            ``T*N`` samples, then PPO uses ``A^e+beta*A^i``.
        """
        results: List[TrainingIterationMetrics] = []
        samples_per_update = self.config.ppo.rollout_steps * self.num_envs
        total_updates = self.config.training.total_environment_steps // samples_per_update
        for update_index in range(total_updates):
            self._maybe_freeze_encoder(update_index)
            summary = self._collect_rollout()
            tensors = self.rollout_buffer.tensors()
            extrinsic_last_value, intrinsic_last_value = self.critic(
                self.rollout_buffer.observations[-1]
            )
            estimate: TwoStreamAdvantageEstimate = compute_two_stream_returns_and_advantages(
                tensors["extrinsic_rewards"],
                tensors["intrinsic_rewards"],
                tensors["extrinsic_values"],
                tensors["intrinsic_values"],
                tensors["terminated"],
                tensors["truncated"],
                extrinsic_last_value.detach(),
                intrinsic_last_value.detach(),
                self.config.ppo.gamma,
                self.config.ppo.gae_lambda,
                self.config.ppo.intrinsic_gamma,
                self.config.ppo.intrinsic_gae_lambda,
                self.config.ppo.intrinsic_advantage_coefficient,
                self.config.ppo.normalize_advantages,
            )
            ppo_metrics = self.ppo.update(self.rollout_buffer, estimate)
            bigan_metrics = self.bigan.update(self.replay_buffer)
            if self.master_transition_enabled:
                # One MSE forward-model update per PPO update; the optimizer
                # holds only f_phi parameters, so the BiGAN encoder keeps being
                # trained exclusively by the BiGAN adversarial objective.
                assert self.master_transition_trainer is not None
                assert self.master_transition_buffer is not None
                transition_metrics = self.master_transition_trainer.update(
                    self.master_transition_buffer
                )
            elif self.config.novelty.novelty_type == "transition":
                assert self.transition_trainer is not None
                assert self.transition_replay_buffer is not None
                transition_metrics = self.transition_trainer.update(
                    self.transition_replay_buffer,
                    self.config.bigan.batch_size,
                )
            else:
                transition_metrics = None
            metric_eme_metrics = self._update_metric_exploration_bonus(
                summary, tensors
            )
            metrics = TrainingIterationMetrics(
                update_index,
                self.environment_steps,
                summary["extrinsic"],
                summary["intrinsic"],
                summary["total"],
                summary["pixel"],
                summary["feature"],
                summary["normalized"],
                ppo_metrics,
                bigan_metrics,
                transition_metrics,
                metric_eme_metrics,
            )
            results.append(metrics)
            if self.logger is not None:
                self.logger.log(
                    self.environment_steps,
                    {
                        "update": update_index,
                        "reward/extrinsic": metrics.extrinsic_reward,
                        "reward/intrinsic": metrics.intrinsic_reward,
                        "reward/total": metrics.total_reward,
                        "novelty/pixel": metrics.pixel_novelty,
                        "novelty/feature": metrics.feature_novelty,
                        "novelty/normalized": metrics.normalized_novelty,
                        "ppo": metrics.ppo,
                        "bigan": metrics.bigan,
                        "transition": metrics.transition,
                        **self._metric_eme_log_entries(metrics.metric_eme),
                    },
                )
            if self.plotter is not None:
                self.plotter.maybe_plot(update_index)
        if self.plotter is not None:
            self.plotter.close()
        return results

    def close(self) -> None:
        """Close all environment, plotting, and logger resources.

        Input: This trainer.
        Output: No value; all resources are finalized.
        Mathematical meaning: Completes one reproducible vectorized run.
        """
        if self.plotter is not None:
            self.plotter.close()
        self.environment.close()
        if self.logger is not None:
            self.logger.close()
