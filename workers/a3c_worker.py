"""Asynchronous actor-learners for the A3C agent.

One :class:`A3CWorker` is one actor-learner: it drives one environment, samples
``n``-step segments with its local copy of the shared actor-critic, asks the
training pipeline for the *existing* Adventurer intrinsic reward of those
transitions, computes the n-step two-stream advantages and A3C gradients, and
pushes those gradients to the shared global network.

The worker deliberately contains no exploration mathematics. Novelty, Equation
(5) reward-scale normalization, and the BiGAN replay are all supplied by the
training pipeline through the small :class:`A3CWorkerContext` contract, so the
A3C path cannot drift away from PPO's reward formulation.

Why threads (and not processes)? The original A3C actor-learners are CPU threads
of one process sharing a global network, which is exactly what is done here: the
global parameters and the single shared Adam optimizer live in this process, the
actor-learners run concurrently, and the exploration pipeline (one BiGAN
encoder/generator/discriminator, one novelty normalizer, one replay buffer) is
shared by all of them without serialising its state through a pipe. With
processes each actor-learner would need its own drifting copy of the BiGAN
encoder *and* its own running novelty statistics, which would change the
intrinsic-reward signal that the thesis is measuring.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from typing import Any, Optional, Protocol, Union

import torch
from torch import Tensor

from agent.a3c import A3CAgent, A3CActorCritic, A3CUpdateMetrics
from config import ExperimentConfig
from environments.adapter import SingleEnvironmentAdapter
from utils.seed import derive_seed


@dataclass(frozen=True)
class A3CSegment:
    """One collected ``n``-step segment of a single actor-learner.

    The tensor layout matches the trainer's rollout buffer convention:
    ``observations[t]`` is the state at which ``actions[t]`` was taken,
    ``next_observations[t]`` is the state returned by the environment (the
    terminal observation when the episode ended), and ``extrinsic_values[t]`` is
    ``V(s_t)``. ``final_observation`` is the state following the last transition,
    which is the bootstrap state of the n-step return unless the last transition
    terminated.

    ``intrinsic_rewards`` starts as zeros and is filled by the training
    pipeline; every other field is produced by the actor-learner.
    """

    observations: Tensor
    actions: Tensor
    next_observations: Tensor
    terminated: Tensor
    truncated: Tensor
    extrinsic_rewards: Tensor
    intrinsic_rewards: Tensor
    extrinsic_values: Tensor
    intrinsic_values: Tensor
    final_observation: Tensor

    def __len__(self) -> int:
        """Return the number of transitions in this segment.

        Input: This segment.
        Output: Positive integer ``n``.
        Mathematical meaning: The horizon of the n-step return estimator.
        """
        return int(self.observations.shape[0])


class A3CWorkerContext(Protocol):
    """Contract between an actor-learner and the training pipeline.

    The trainer implements these methods; the worker only uses them, so the
    exploration pipeline can change without touching the worker loop.
    """

    def claim_steps(self, count: int) -> int:
        """Reserve up to ``count`` environment steps from the global budget."""

    def segment_intrinsic_rewards(self, segment: A3CSegment) -> Any:
        """Return the pipeline's extrinsic/intrinsic reward streams."""

    def record_episode(self, score: float, length: int) -> None:
        """Record one completed episode score and length."""

    def on_segment_complete(self, worker_index: int, metrics: A3CUpdateMetrics) -> None:
        """Consume one applied actor-learner update."""

    def should_stop(self) -> bool:
        """Return whether the run must stop (budget spent or fatal error)."""

    def request_stop(self) -> None:
        """Signal every actor-learner to stop after its current segment."""


class A3CWorker:
    """One asynchronous actor-learner running in its own thread.

    Args:
        worker_index: Zero-based actor-learner identifier.
        environment: The environment adapter this actor-learner drives.
        agent: Shared A3C agent holding the global network.
        context: Training-pipeline contract described by
            :class:`A3CWorkerContext`.
        config: Complete validated experiment configuration.
        seed: Base experiment seed; this worker derives its own environment and
            action-sampling streams from it.
        n_steps: Segment length ``n`` of the n-step return estimator.
        device: Device of the local and shared networks.
    """

    def __init__(
        self,
        worker_index: int,
        environment: SingleEnvironmentAdapter,
        agent: A3CAgent,
        context: A3CWorkerContext,
        config: ExperimentConfig,
        seed: int,
        n_steps: int,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        """Initialize the actor-learner and its independent random streams.

        Input: Worker identity, environment, agent, pipeline context,
            configuration, seed, segment length, and device.
        Output: An actor-learner whose thread has not been started yet.
        Mathematical meaning: Fixes this actor-learner's initial-state and
            action-sampling distributions as independent deterministic streams
            derived from the experiment seed.
        """
        if worker_index < 0:
            raise ValueError("worker_index must be non-negative")
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        self.worker_index = worker_index
        self.environment = environment
        self.agent = agent
        self.context = context
        self.config = config
        self.device = torch.device(device)
        self.n_steps = int(n_steps)
        self.discrete_actions = bool(config.environment.discrete_actions)
        # Independent streams: one for the environment's initial states, one for
        # the policy's action samples. Splitting the RNG per worker is what
        # makes `--seed` meaningful with several actor-learners.
        self._environment_seed = derive_seed(derive_seed(seed, 0xA3C), worker_index)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(derive_seed(derive_seed(seed, 0x5A3), worker_index))
        self.generator = generator
        self._episode_index = 0
        self._observation: Optional[Tensor] = None
        self._episode_return = 0.0
        self._episode_length = 0
        self.segments_completed = 0
        self.episodes_completed = 0
        self.last_metrics: Optional[A3CUpdateMetrics] = None
        self.error: Optional[BaseException] = None
        self._thread = threading.Thread(
            target=self._run_safely, name=f"a3c-worker-{worker_index}", daemon=True
        )

    def start(self) -> None:
        """Start the actor-learner thread.

        Input: This worker.
        Output: No value; the thread begins collecting segments.
        Mathematical meaning: Launches one asynchronous optimisation process
            sharing the global parameters.
        """
        self._thread.start()

    def join(self, timeout: Optional[float] = None) -> None:
        """Wait for the actor-learner thread to finish.

        Input: Optional timeout in seconds.
        Output: No value; raises the worker's exception if it failed.
        Mathematical meaning: Terminates one asynchronous optimisation process
            and propagates any fatal error to the caller.
        """
        self._thread.join(timeout)
        if self.error is not None:
            raise self.error

    @property
    def is_alive(self) -> bool:
        """Return whether the actor-learner thread is still running.

        Input: This worker.
        Output: Boolean thread liveness.
        Mathematical meaning: None; operational status only.
        """
        return self._thread.is_alive()

    def _run_safely(self) -> None:
        """Run the actor-learner loop and convert failures into a stop signal.

        Input: This worker.
        Output: No value; ``error`` is set if the loop raised.
        Mathematical meaning: Guarantees that one failing actor-learner cannot
            leave the remaining ones waiting forever.
        """
        try:
            self._run()
        except BaseException as error:  # noqa: BLE001 - propagated to join()
            self.error = error
            self.context.request_stop()

    def _run(self) -> None:
        """Collect segments and push gradients until the budget is exhausted.

        Input: This worker.
        Output: No value; the actor-learner stops when no steps remain.
        Mathematical meaning: Repeatedly samples an n-step trajectory and
            performs one asynchronous actor-critic update.
        """
        local_network = self.agent.new_local_network()
        self._observation = self._reset_environment()
        while not self.context.should_stop():
            granted_steps = self.context.claim_steps(self.n_steps)
            if granted_steps <= 0:
                break
            self.agent.synchronize_local(local_network)
            segment = self._collect_segment(local_network, granted_steps)
            reward_result = self.context.segment_intrinsic_rewards(segment)
            segment = replace(
                segment, intrinsic_rewards=reward_result.intrinsic_reward.detach()
            )
            with torch.no_grad():
                final_extrinsic, final_intrinsic = local_network.critic(
                    segment.final_observation
                )
            estimate = self.agent.estimate_returns(
                segment,
                final_extrinsic.reshape(()),
                final_intrinsic.reshape(()),
            )
            metrics = self.agent.update(
                local_network, segment.observations, segment.actions, estimate
            )
            self.segments_completed += 1
            self.last_metrics = metrics
            self.context.on_segment_complete(self.worker_index, metrics)

    def _reset_environment(self) -> Tensor:
        """Reset this worker's environment from its seeded episode stream.

        Input: This worker.
        Output: Initial observation tensor of the new episode.
        Mathematical meaning: Samples a new initial state ``s_0`` from this
            actor-learner's deterministic initial-state stream.
        """
        seed = derive_seed(self._environment_seed, self._episode_index)
        self._episode_index += 1
        observation, _ = self.environment.reset(seed=seed)
        return observation.to(self.device)

    def _collect_segment(self, local_network: A3CActorCritic, length: int) -> A3CSegment:
        """Collect ``length`` transitions with the local policy.

        Input: Local actor-critic and the number of transitions to collect.
        Output: ``A3CSegment`` with zero intrinsic rewards.
        Mathematical meaning: Samples ``a_t ~ pi_theta(.|s_t)`` and records the
            transition ``(s_t, a_t, r^e_t, s_(t+1), terminated_t, truncated_t)``
            together with ``V^e(s_t)`` and ``V^i(s_t)`` for the GAE recurrence.
        """
        observations = []
        actions = []
        next_observations = []
        terminated = []
        truncated = []
        rewards = []
        extrinsic_values = []
        intrinsic_values = []
        for _ in range(length):
            assert self._observation is not None
            observation = self._observation
            action, _, _, extrinsic_value, intrinsic_value = local_network.act(
                observation, self.generator
            )
            step = self.environment.step(action.detach().reshape(-1))
            next_observation = step.observation.to(self.device)
            observations.append(observation)
            actions.append(
                action.detach().reshape(-1)[0]
                if self.discrete_actions
                else action.detach().reshape(-1)
            )
            next_observations.append(next_observation)
            terminated.append(step.terminated)
            truncated.append(step.truncated)
            rewards.append(float(step.reward))
            extrinsic_values.append(extrinsic_value.detach().reshape(-1))
            intrinsic_values.append(intrinsic_value.detach().reshape(-1))
            self._episode_return += float(step.reward)
            self._episode_length += 1
            if step.done:
                self.context.record_episode(self._episode_return, self._episode_length)
                self.episodes_completed += 1
                self._episode_return = 0.0
                self._episode_length = 0
                self._observation = self._reset_environment()
            else:
                self._observation = next_observation
        action_dtype = torch.long if self.discrete_actions else torch.float32
        return A3CSegment(
            observations=torch.stack(observations, dim=0),
            actions=torch.stack(actions, dim=0).to(dtype=action_dtype),
            next_observations=torch.stack(next_observations, dim=0),
            terminated=torch.tensor(terminated, dtype=torch.bool, device=self.device),
            truncated=torch.tensor(truncated, dtype=torch.bool, device=self.device),
            extrinsic_rewards=torch.tensor(
                rewards, dtype=torch.float32, device=self.device
            ),
            intrinsic_rewards=torch.zeros(length, dtype=torch.float32, device=self.device),
            extrinsic_values=torch.cat(extrinsic_values),
            intrinsic_values=torch.cat(intrinsic_values),
            final_observation=self._observation,
        )
