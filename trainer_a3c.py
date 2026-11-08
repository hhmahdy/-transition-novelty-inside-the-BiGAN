"""Asynchronous Advantage Actor-Critic training pipeline for Adventurer.

This module adds A3C next to PPO. It is a *subclass* of
:class:`trainer.AdventurerTrainer` on purpose: the base initialiser builds the
environment-facing buffers and the whole exploration pipeline (BiGAN encoder,
generator, discriminator, ``NoveltyEstimator``, latent transition novelty,
metric/EME bonus, Equation (5) intrinsic-reward pipeline, BiGAN replay buffer,
plotter, logger) with exactly the code PPO uses. The subclass therefore cannot
silently diverge from PPO's BiGAN architecture, novelty formulation, reward
normalisation, or logging conventions; it only replaces the optimisation loop
(PPO's fixed rollout + clipped epochs) with asynchronous actor-learners.

What is identical to PPO
------------------------
* the environments, observation preprocessing, and action handling;
* the actor and critic architectures (``agent.actor.ActorNetwork`` and
  ``agent.critic.DualCriticNetwork``), including initialisation;
* the BiGAN networks, their adversarial objective, and the freeze schedule;
* the novelty computation and its Equation (5) running normalisation
  (``IntrinsicRewardPipeline`` is the same object, called with the same
  arguments as in ``AdventurerTrainer._collect_rollout``);
* the two reward streams, the separate extrinsic/intrinsic critics, and the
  ``A^e + beta A^i`` policy advantage (via
  ``agent.advantage.compute_two_stream_returns_and_advantages``);
* termination/truncation semantics (termination masks the bootstrap,
  truncation keeps it);
* the output-directory, ``config.json``, ``metrics.jsonl``, ``run.log``, and
  figure conventions.

What A3C adds
-------------
* ``num_parallel_envs`` asynchronous actor-learners, one environment each, each
  with its own seeded random streams and its own local copy of the network;
* one shared global network with one shared Adam optimizer, updated through a
  lock as gradients arrive from the actor-learners;
* n-step returns over ``a3c.n_steps`` transitions instead of a full rollout;
* entropy regularisation and gradient-norm clipping with the existing PPO
  coefficients.

Scheduling note
---------------
The BiGAN (and latent forward-model) optimisation that PPO performs once per
rollout is performed here once per ``a3c.bigan_update_interval`` applied
actor-learner updates, whichever actor-learner happens to complete the update
that crosses the cadence. The work is serialised by the exploration lock, so the
BiGAN networks and the novelty normalisers are never read and written at the same
time by two threads.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch

from agent.a3c import A3CAgent, A3CActorCritic, A3CUpdateMetrics
from bigan.trainer import BiGANUpdateMetrics
from config import ExperimentConfig
from environments.vector_adapter import VectorEnvironmentAdapter
from exploration.intrinsic_reward import IntrinsicRewardResult
from exploration.master_transition import MasterTransitionUpdateMetrics
from exploration.transition_novelty import TransitionUpdateMetrics
from trainer import AdventurerTrainer, MetricEMEMetrics
from utils.logger import ExperimentLogger
from workers.a3c_worker import A3CSegment, A3CWorker


@dataclass(frozen=True)
class A3CIterationMetrics:
    """Diagnostics aggregated over one logged window of actor-learner updates.

    The field layout mirrors :class:`trainer.TrainingIterationMetrics` so the
    same analysis code can read a PPO run and an A3C run side by side. The
    optimisation diagnostics live in ``a3c`` instead of ``ppo`` because the
    objective is different (no clipping, n-step returns), while the reward,
    novelty, BiGAN, transition-novelty, and metric/EME fields are the same
    quantities PPO reports.
    """

    update: int
    environment_steps: int
    applied_updates: int
    extrinsic_reward: float
    intrinsic_reward: float
    total_reward: float
    pixel_novelty: float
    feature_novelty: float
    normalized_novelty: float
    a3c: A3CUpdateMetrics
    bigan: Optional[BiGANUpdateMetrics] = None
    transition: Optional[
        Union[TransitionUpdateMetrics, MasterTransitionUpdateMetrics]
    ] = None
    metric_eme: Optional[MetricEMEMetrics] = None


class A3CTrainer(AdventurerTrainer):
    """Train Adventurer's exploration pipeline with asynchronous A3C.

    The constructor reuses :class:`trainer.AdventurerTrainer` for every shared
    component and adds the A3C agent, the actor-learner threads, and the
    thread-safe bookkeeping described in the module docstring.
    """

    def __init__(
        self,
        environment: VectorEnvironmentAdapter,
        config: ExperimentConfig,
        logger: Optional[ExperimentLogger] = None,
    ) -> None:
        """Build the shared exploration pipeline and the A3C agent.

        Input: Vector environment (one child per actor-learner), complete
            configuration, and optional logger.
        Output: Initialized A3C trainer.
        Mathematical meaning: Instantiates the shared global
            ``(pi_theta, V^e_phi, V^i_psi)`` parameter point, the BiGAN
            exploration pipeline, and the state needed to schedule asynchronous
            updates.
        """
        if config.algorithm != "a3c":
            raise ValueError(
                "A3CTrainer requires ExperimentConfig.algorithm='a3c'"
            )
        if config.training.resettable:
            raise ValueError(
                "algorithm='a3c' does not support training.resettable: Algorithm 2's "
                "episodic-memory state restoration is defined per batched PPO "
                "rollout index and has no A3C counterpart. The option is off by "
                "default and is not part of the Master's thesis experiment."
            )
        super().__init__(environment, config, logger)
        # The base initialiser allocates PPO-only objects. A3C never uses them,
        # and the rollout buffer alone is a full 96x128 observation batch at the
        # Atari configuration, so release both instead of holding them idle.
        self.rollout_buffer = None  # type: ignore[assignment]
        self.ppo = None  # type: ignore[assignment]
        self.a3c_config = config.a3c
        self.num_workers = self.num_envs
        # The base trainer's actor and critic are the shared *global* network.
        self.actor_critic = A3CActorCritic(self.actor, self.critic)
        self.agent = A3CAgent(
            self.actor_critic,
            config.ppo,
            total_environment_steps=config.training.total_environment_steps,
            num_workers=self.num_workers,
            n_steps=self.a3c_config.n_steps,
            beta=config.ppo.intrinsic_advantage_coefficient,
            device=self.device,
        )
        # Lock order (never acquire an earlier lock while holding a later one):
        # logging -> bookkeeping -> exploration.
        self._exploration_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._steps_claimed = 0
        self._last_exploration_update = 0
        self._pending_updates: List[A3CUpdateMetrics] = []
        self._pending_bigan: List[BiGANUpdateMetrics] = []
        self._pending_transition: List[Any] = []
        self._pending_metric_eme: List[Optional[MetricEMEMetrics]] = []
        self._stream_sums = self._empty_stream_sums()
        self._stream_samples = 0
        self._log_sums = self._empty_stream_sums()
        self._log_samples = 0
        self._eme_tensors: Optional[Dict[str, torch.Tensor]] = None
        self._results: List[A3CIterationMetrics] = []
        self._logged_updates = 0
        self._workers: List[A3CWorker] = []

    @staticmethod
    def _empty_stream_sums() -> Dict[str, float]:
        """Return a zeroed reward/novelty accumulator.

        Input: None.
        Output: Mapping of the same accumulator keys PPO's rollout summary uses.
        Mathematical meaning: Prepares the empirical means of the extrinsic and
            intrinsic reward streams and of the novelty components.
        """
        return {
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

    # ------------------------------------------------------------------
    # Environment budget and episode bookkeeping (called from worker threads)
    # ------------------------------------------------------------------

    def claim_steps(self, count: int) -> int:
        """Reserve environment steps from the shared training budget.

        Input: Number of steps the calling actor-learner would like to collect.
        Output: Granted step count, possibly shorter than ``count`` and zero
            when the budget is exhausted.
        Mathematical meaning: Partitions the finite interaction budget
            ``total_environment_steps`` across the asynchronous actor-learners
            without double counting.
        """
        if count <= 0:
            return 0
        with self._state_lock:
            remaining = (
                self.config.training.total_environment_steps - self._steps_claimed
            )
            granted = min(count, remaining)
            if granted <= 0:
                return 0
            self._steps_claimed += granted
            self.environment_steps += granted
            return int(granted)

    def record_episode(self, score: float, length: int) -> None:
        """Record one completed episode from an actor-learner.

        Input: Episodic extrinsic return and episode length.
        Output: No value; the score is sent to the plotter and logger.
        Mathematical meaning: Records one empirical sample of the episodic
            return, exactly as PPO's ``_record_completed_episodes`` does.
        """
        with self._state_lock:
            timestep = self.environment_steps
        with self._log_lock:
            if self.plotter is not None:
                self.plotter.record_episode(timestep, score, length)
            if self.logger is not None:
                self.logger.log(
                    timestep,
                    {"episode/game_score": score, "episode/length": length},
                )

    def should_stop(self) -> bool:
        """Return whether actor-learners must stop.

        Input: This trainer.
        Output: ``True`` after the budget is exhausted or a worker failed.
        Mathematical meaning: Guarantees termination of the asynchronous
            optimisation processes.
        """
        if self._stop_event.is_set():
            return True
        with self._state_lock:
            return self._steps_claimed >= self.config.training.total_environment_steps

    def request_stop(self) -> None:
        """Ask every actor-learner to stop after its current segment.

        Input: This trainer.
        Output: No value; the stop event is set.
        Mathematical meaning: Propagates one actor-learner's fatal error to the
            others so a failed run cannot hang.
        """
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Exploration rewards (called from worker threads)
    # ------------------------------------------------------------------

    def segment_intrinsic_rewards(self, segment: A3CSegment) -> IntrinsicRewardResult:
        """Compute the existing exploration rewards for one collected segment.

        Input: ``A3CSegment`` of ``n`` transitions collected by one
            actor-learner.
        Output: ``IntrinsicRewardResult`` with the unchanged extrinsic reward,
            the Equation (5) intrinsic reward, and the normalized novelty.
        Mathematical meaning: Applies exactly the novelty branch that
            ``AdventurerTrainer._collect_rollout`` applies for the configured
            run -- BiGAN state novelty ``B(s)``, the Master's latent transition
            novelty ``N_T``, the legacy transition novelty, or the metric/EME
            latent bonus ``b_t`` -- over the segment's ``n`` transitions, and
            feeds the result through the same ``IntrinsicRewardPipeline``.

        The whole body runs under the exploration lock: the novelty
        normalisers, the BiGAN networks, and the replay buffers are shared by
        all actor-learners, and the lock keeps their state a single
        well-defined sequence instead of one interleaved stream per worker.
        """
        observations = segment.observations.to(self.device)
        actions = segment.actions.to(self.device)
        next_observations = segment.next_observations.to(self.device)
        extrinsic_rewards = segment.extrinsic_rewards.to(self.device)
        with self._exploration_lock:
            sums = self._empty_stream_sums()
            if self.config.metric_eme.enabled:
                # Same branch as PPO: b_t = d_t * min(max(zeta(r),1),M).
                assert self.metric_reward is not None
                done = segment.terminated | segment.truncated
                novelty = self.metric_reward(
                    observations,
                    next_observations,
                    extrinsic_reward=extrinsic_rewards,
                    update_statistics=True,
                    done=done,
                    actions=actions,
                )
                if self.reward_ensemble is not None:
                    self.reward_ensemble.add(
                        observations, extrinsic_rewards, actions=actions
                    )
                sums["latent_distance"] += float(
                    novelty.latent_distance.detach().sum().cpu()
                )
                sums["ensemble_variance"] += float(
                    novelty.ensemble_variance.detach().sum().cpu()
                )
                sums["bonus_scale"] += float(novelty.bonus_scale.detach().sum().cpu())
                sums["bonus"] += float(novelty.combined_score.detach().sum().cpu())
                sums["latent_norm"] += novelty.latent_norm_mean * len(segment)
                sums["bonus_distance"] += float(
                    novelty.bonus_distance.detach().sum().cpu()
                )
                if novelty.episodic_scale is not None:
                    sums["episodic_scale"] += float(
                        novelty.episodic_scale.detach().sum().cpu()
                    )
                sums["variance_below_min"] += (
                    novelty.variance_below_min_fraction * len(segment)
                )
                sums["variance_above_max"] += (
                    novelty.variance_above_max_fraction * len(segment)
                )
                sums["latent_distance_p5"] += novelty.latent_distance_p5 * len(segment)
                sums["latent_distance_p50"] += novelty.latent_distance_p50 * len(segment)
                sums["latent_distance_p95"] += novelty.latent_distance_p95 * len(segment)
                # EME's learned metric regresses d_phi on rollout triples; the
                # most recent segment provides them.
                self._eme_tensors = {
                    "observations": observations.detach(),
                    "actions": actions.detach(),
                    "extrinsic_rewards": extrinsic_rewards.detach(),
                    "terminated": segment.terminated.detach(),
                }
            elif self.config.novelty.novelty_type == "state":
                assert self.novelty is not None
                novelty = self.novelty(
                    next_observations,
                    update_statistics=True,
                    extrinsic_reward=extrinsic_rewards,
                )
            elif self.master_transition_enabled:
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
                    observations.detach(),
                    actions.detach(),
                    next_observations.detach(),
                )
            reward_result = self.reward_pipeline(extrinsic_rewards, novelty)
            # BiGAN observation replay, exactly as PPO adds next observations.
            self.replay_buffer.add_batch(next_observations.detach())
            sums["extrinsic"] += float(
                reward_result.extrinsic_reward.detach().sum().cpu()
            )
            sums["intrinsic"] += float(
                reward_result.intrinsic_reward.detach().sum().cpu()
            )
            sums["total"] += float(
                (
                    reward_result.extrinsic_reward
                    + reward_result.intrinsic_reward
                ).detach().sum().cpu()
            )
            sums["pixel"] += float(novelty.pixel_error.detach().sum().cpu())
            sums["feature"] += float(novelty.feature_error.detach().sum().cpu())
            sums["normalized"] += float(novelty.normalized_score.detach().sum().cpu())
            self._accumulate(sums, len(segment))
        return reward_result

    def _accumulate(self, sums: Dict[str, float], samples: int) -> None:
        """Merge one segment's reward/novelty sums into both accumulators.

        Input: Per-segment sums and the segment length; caller holds the
            exploration lock.
        Output: No value; the exploration-step and logging accumulators are
            advanced.
        Mathematical meaning: Accumulates the numerators of the empirical means
            reported for the metric bonus and for the run's logs.
        """
        for key, value in sums.items():
            self._stream_sums[key] += value
            self._log_sums[key] += value
        self._stream_samples += samples
        self._log_samples += samples

    def _take_stream_summary(self) -> Dict[str, float]:
        """Return and reset the exploration-step accumulator.

        Input: This trainer; caller holds the exploration lock.
        Output: Mean reward/novelty values over the segments since the last
            exploration step.
        Mathematical meaning: Provides the quantities
            ``AdventurerTrainer._update_metric_exploration_bonus`` reports.
        """
        summary = {
            key: (value / self._stream_samples if self._stream_samples else 0.0)
            for key, value in self._stream_sums.items()
        }
        self._stream_sums = self._empty_stream_sums()
        self._stream_samples = 0
        return summary

    def _take_log_summary(self) -> Dict[str, float]:
        """Return and reset the logging accumulator.

        Input: This trainer; caller holds the exploration lock.
        Output: Mean reward/novelty values over the segments since the last log
            record.
        Mathematical meaning: Provides the reward and novelty means reported in
            ``metrics.jsonl``, comparable to PPO's per-rollout means.
        """
        summary = {
            key: (value / self._log_samples if self._log_samples else 0.0)
            for key, value in self._log_sums.items()
        }
        self._log_sums = self._empty_stream_sums()
        self._log_samples = 0
        return summary

    # ------------------------------------------------------------------
    # Scheduled BiGAN / forward-model optimisation (called from workers)
    # ------------------------------------------------------------------

    def on_segment_complete(self, worker_index: int, metrics: A3CUpdateMetrics) -> None:
        """Consume one applied actor-learner update.

        Input: Identifier of the worker that finished a segment and the update
            diagnostics.
        Output: No value; the update is recorded and, when the cadence is
            reached, the BiGAN/forward-model/metric optimisation runs.
        Mathematical meaning: Realises PPO's "one exploration update per policy
            update" schedule in the asynchronous setting, without letting two
            threads train the BiGAN networks simultaneously.
        """
        with self._state_lock:
            self._pending_updates.append(metrics)
            applied_updates = self.agent.applied_updates
            due = (
                applied_updates - self._last_exploration_update
                >= self.a3c_config.bigan_update_interval
            )
            if due:
                self._last_exploration_update = applied_updates
        if due:
            self._exploration_step(applied_updates)
        self._maybe_log()

    def _exploration_step(self, applied_updates: int) -> None:
        """Run the BiGAN, transition-model, and metric updates.

        Input: Number of actor-learner updates applied so far.
        Output: No value; optimiser steps are recorded for the next log record.
        Mathematical meaning: Performs the same BiGAN adversarial update,
            latent forward-model MSE update, and (when configured) EME ensemble
            and metric updates that PPO performs once per rollout.
        """
        with self._exploration_lock:
            # The freeze schedule is expressed in applied updates, the A3C
            # analogue of PPO's update index.
            self._maybe_freeze_encoder(applied_updates)
            bigan_metrics = self.bigan.update(self.replay_buffer)
            transition_metrics: Optional[
                Union[TransitionUpdateMetrics, MasterTransitionUpdateMetrics]
            ] = None
            if self.master_transition_enabled:
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
            summary = self._take_stream_summary()
            tensors = self._eme_tensors
            metric_eme_metrics = self._update_metric_exploration_bonus(
                summary, tensors
            )
        with self._state_lock:
            self._pending_bigan.append(bigan_metrics)
            self._pending_transition.append(transition_metrics)
            self._pending_metric_eme.append(metric_eme_metrics)

    # ------------------------------------------------------------------
    # Logging (called from workers, serialised)
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_update_metrics(updates: List[A3CUpdateMetrics]) -> A3CUpdateMetrics:
        """Average a window of actor-learner updates.

        Input: Non-empty list of per-update diagnostics.
        Output: ``A3CUpdateMetrics`` whose fields are the window means.
        Mathematical meaning: Reports the mean objective values of the window,
            as PPO reports the mean over its minibatch epochs.
        """
        count = float(len(updates))
        return A3CUpdateMetrics(
            **{
                field: sum(getattr(item, field) for item in updates) / count
                for field in A3CUpdateMetrics.__dataclass_fields__
            }
        )

    @staticmethod
    def _mean_bigan_metrics(
        records: List[BiGANUpdateMetrics],
    ) -> Optional[BiGANUpdateMetrics]:
        """Average the BiGAN diagnostics collected in one logging window.

        Input: BiGAN update records produced since the previous log record.
        Output: Mean ``BiGANUpdateMetrics``, or ``None`` when no BiGAN update
            happened in the window.
        Mathematical meaning: Reports the adversarial objective values of the
            window while keeping the schedule counters of the last update.
        """
        if not records:
            return None
        count = float(len(records))
        last = records[-1]
        return BiGANUpdateMetrics(
            updated=any(record.updated for record in records),
            discriminator_loss=sum(item.discriminator_loss for item in records) / count,
            encoder_generator_loss=sum(
                item.encoder_generator_loss for item in records
            ) / count,
            reconstruction_loss=sum(item.reconstruction_loss for item in records) / count,
            discriminator_real_logit=sum(
                item.discriminator_real_logit for item in records
            ) / count,
            discriminator_fake_logit=sum(
                item.discriminator_fake_logit for item in records
            ) / count,
            update_count=last.update_count,
        )

    def _maybe_log(self, force: bool = False) -> None:
        """Emit one metrics record per completed logging window.

        Input: ``force`` to emit a partial final window.
        Output: No value; ``metrics.jsonl``, ``run.log``, TensorBoard, and the
            figures are updated and the record is appended to the results.
        Mathematical meaning: Records the optimisation, reward, and novelty
            trajectory of the asynchronous run at environment-step milestones.
        """
        interval = self.config.training.log_interval_updates
        while True:
            with self._log_lock:
                with self._state_lock:
                    pending = len(self._pending_updates)
                    if pending == 0 or (pending < interval and not force):
                        return
                    take = min(interval, pending)
                    updates = self._pending_updates[:take]
                    del self._pending_updates[:take]
                    bigan_records = self._pending_bigan
                    self._pending_bigan = []
                    transition_records = self._pending_transition
                    self._pending_transition = []
                    metric_records = self._pending_metric_eme
                    self._pending_metric_eme = []
                    applied_updates = self.agent.applied_updates
                    environment_steps = self.environment_steps
                with self._exploration_lock:
                    summary = self._take_log_summary()
                metric_eme = next(
                    (item for item in reversed(metric_records) if item is not None),
                    None,
                )
                transition = next(
                    (item for item in reversed(transition_records) if item is not None),
                    None,
                )
                update_index = self._logged_updates
                self._logged_updates += 1
                metrics = A3CIterationMetrics(
                    update=update_index,
                    environment_steps=environment_steps,
                    applied_updates=applied_updates,
                    extrinsic_reward=summary["extrinsic"],
                    intrinsic_reward=summary["intrinsic"],
                    total_reward=summary["total"],
                    pixel_novelty=summary["pixel"],
                    feature_novelty=summary["feature"],
                    normalized_novelty=summary["normalized"],
                    a3c=self._mean_update_metrics(updates),
                    bigan=self._mean_bigan_metrics(bigan_records),
                    transition=transition,
                    metric_eme=metric_eme,
                )
                self._results.append(metrics)
                if self.logger is not None:
                    self.logger.log(
                        environment_steps,
                        {
                            "update": update_index,
                            "algorithm": "a3c",
                            "reward/extrinsic": metrics.extrinsic_reward,
                            "reward/intrinsic": metrics.intrinsic_reward,
                            "reward/total": metrics.total_reward,
                            "novelty/pixel": metrics.pixel_novelty,
                            "novelty/feature": metrics.feature_novelty,
                            "novelty/normalized": metrics.normalized_novelty,
                            "a3c": metrics.a3c,
                            "bigan": metrics.bigan,
                            "transition": metrics.transition,
                            **self._metric_eme_log_entries(metrics.metric_eme),
                        },
                    )
                if self.plotter is not None:
                    self.plotter.maybe_plot(update_index)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _make_workers(self) -> List[A3CWorker]:
        """Create one actor-learner per configured parallel environment.

        Input: This trainer.
        Output: List of unstarted ``A3CWorker`` objects.
        Mathematical meaning: Instantiates the asynchronous optimisation
            processes that share the global parameter point.
        """
        workers: List[A3CWorker] = []
        for index in range(self.num_workers):
            workers.append(
                A3CWorker(
                    worker_index=index,
                    environment=self.environment.environments[index],
                    agent=self.agent,
                    context=self,
                    config=self.config,
                    seed=self.config.seed.seed,
                    n_steps=self.a3c_config.n_steps,
                    device=self.device,
                )
            )
        return workers

    def train(self) -> List[A3CIterationMetrics]:
        """Run the asynchronous actor-learners to the environment-step budget.

        Input: No arguments; uses the immutable experiment configuration.
        Output: Per-log-window diagnostics.
        Mathematical meaning: Executes ``total_environment_steps`` environment
            transitions distributed over the actor-learners, optimising
            ``pi_theta``, ``V^e_phi``, and ``V^i_psi`` by n-step advantage
            actor-critic updates while the BiGAN novelty model is trained on the
            replay of visited observations.
        """
        self._workers = self._make_workers()
        for worker in self._workers:
            worker.start()
        for worker in self._workers:
            # A worker failure sets the stop event, so every join terminates.
            worker.join()
        for worker in self._workers:
            if worker.error is not None:
                raise worker.error
        self._maybe_log(force=True)
        if self.plotter is not None:
            self.plotter.close()
        return list(self._results)
