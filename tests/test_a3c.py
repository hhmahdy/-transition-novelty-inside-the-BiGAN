"""Tests for the additive A3C integration.

The suite checks the six validations required by the integration brief:

1. PPO regression -- the default configuration is still PPO, the PPO trainer is
   still selected by default, and the new ``a3c`` configuration section cannot
   change a PPO run's numbers.
2. Standard A3C -- an extrinsic-only advantage configuration trains.
3. BiGAN state novelty + A3C -- the intrinsic reward A3C receives is numerically
   identical to the intrinsic reward PPO's pipeline produces for the same
   transitions and the same BiGAN weights (no novelty re-implementation).
4. BiGAN transition novelty + A3C -- the same equality for the Master's
   ``N_T = ||f(E(s_t),a_t) - E(s_(t+1))||_2`` variant.
5. Training stability -- finite losses and gradients, non-zero parameter
   movement, worker/global synchronisation, terminated episodes, no
   deadlocks, discrete *and* continuous action spaces.
6. Reproducibility -- seeds are derived per worker and a single-actor-learner
   run repeats exactly.

All runs use tiny CPU environments so the suite finishes in seconds.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

import numpy as np
import torch

from agent.a3c import A3CActorCritic, A3CAgent, A3CUpdateMetrics
from agent.actor import ActorNetwork
from agent.critic import DualCriticNetwork
from config import (
    ALGORITHMS,
    A3CConfig,
    DeviceType,
    ExperimentConfig,
)
from environments.adapter import SingleEnvironmentAdapter
from environments.vector_adapter import VectorEnvironmentAdapter
from trainer import AdventurerTrainer
from trainer_a3c import A3CTrainer
from utils.logger import ExperimentLogger
from utils.run_metadata import write_run_info
from utils.seed import derive_seed
from workers.a3c_worker import A3CSegment

OBSERVATION_DIM = 6
ACTION_DIM = 3
OBSERVATION_SHAPE = (OBSERVATION_DIM,)


class _TinyEnvironment:
    """Minimal Gymnasium-style environment used by the A3C tests.

    Works with a discrete ``spaces.Discrete`` action space or a continuous
    ``spaces.Box`` one, and with vector or image observations, so the same stub
    exercises the categorical and Gaussian policy paths. Episodes terminate
    after ``episode_limit`` steps and the extrinsic reward is sparse for the
    discrete case and dense for the continuous one.
    """

    def __init__(
        self,
        observation_shape: Tuple[int, ...] = OBSERVATION_SHAPE,
        action_dim: int = ACTION_DIM,
        discrete: bool = True,
        episode_limit: int = 5,
    ) -> None:
        """Declare the spaces of the tiny MDP.

        Input: Observation shape, action dimension, action-space type, and
            episode length.
        Output: An uninitialized environment.
        Mathematical meaning: Defines a small finite-horizon MDP whose returns
            are cheap to estimate.
        """
        import gymnasium as gym

        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=observation_shape,
            dtype=np.float32,
        )
        if discrete:
            self.action_space = gym.spaces.Discrete(action_dim)
        else:
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
            )
        self.discrete = discrete
        self.episode_limit = episode_limit
        self.observation_shape = observation_shape
        self._rng = np.random.default_rng(0)
        self._steps = 0
        self.unwrapped = self

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Start a new episode from a seeded initial-state distribution.

        Input: Optional episode seed and ignored options.
        Output: ``(observation, info)`` following the Gymnasium API.
        Mathematical meaning: Samples the initial state ``s_0``.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._steps = 0
        observation = self._rng.uniform(-1.0, 1.0, size=self.observation_shape)
        return observation.astype(np.float32), {}

    def step(self, action: Any):
        """Advance the tiny MDP by one transition.

        Input: Discrete or continuous action.
        Output: ``(observation, reward, terminated, truncated, info)``.
        Mathematical meaning: Pays ``+1`` for discrete action one and the
            negative action magnitude for continuous actions; the episode
            terminates after ``episode_limit`` transitions.
        """
        self._steps += 1
        observation = self._rng.uniform(-1.0, 1.0, size=self.observation_shape)
        if self.discrete:
            reward = 1.0 if int(np.asarray(action).reshape(-1)[0]) == 1 else 0.0
        else:
            reward = -float(np.abs(np.asarray(action)).sum())
        terminated = self._steps >= self.episode_limit
        return observation.astype(np.float32), float(reward), terminated, False, {}

    def close(self) -> None:
        """Release nothing; the stub owns no resources.

        Input: This environment.
        Output: No value.
        Mathematical meaning: None.
        """
        return None


def _vector_environment(
    num_envs: int,
    seed: int,
    observation_shape: Tuple[int, ...] = OBSERVATION_SHAPE,
    action_dim: int = ACTION_DIM,
    discrete: bool = True,
) -> VectorEnvironmentAdapter:
    """Build and seed a vector environment of tiny MDPs.

    Input: Environment count, seed, observation shape, action dimension, and
        action-space type.
    Output: Seeded ``VectorEnvironmentAdapter``.
    Mathematical meaning: Constructs the product MDP one actor-learner per
        environment consumes.
    """
    environments = [
        SingleEnvironmentAdapter(
            _TinyEnvironment(observation_shape, action_dim, discrete)
        )
        for _ in range(num_envs)
    ]
    vector_environment = VectorEnvironmentAdapter(environments)
    vector_environment.seed(seed)
    return vector_environment


def _tiny_config(
    output_directory: Path,
    algorithm: str = "a3c",
    novelty_type: str = "state",
    transition_variant: str = "legacy",
    seed: int = 0,
    num_envs: int = 2,
    rollout_steps: int = 8,
    n_steps: int = 8,
    total_steps: int = 256,
    bigan_batch_size: int = 8,
    discrete: bool = True,
    observation_shape: Tuple[int, ...] = OBSERVATION_SHAPE,
    action_dim: int = ACTION_DIM,
    ppo_overrides: Optional[Dict[str, Any]] = None,
    a3c_overrides: Optional[Dict[str, Any]] = None,
    training_overrides: Optional[Dict[str, Any]] = None,
) -> ExperimentConfig:
    """Build a validated tiny configuration for one A3C or PPO smoke run.

    Input: Output directory, algorithm, novelty selection, schedule sizes, and
        optional field overrides.
    Output: ``ExperimentConfig`` with CPU device, no TensorBoard, no plots.
    Mathematical meaning: Keeps every schedule relationship valid
        (``total_steps`` divisible by both ``num_envs*rollout_steps`` and
        ``num_envs*a3c.n_steps``) at a scale that runs in seconds.
    """
    base = ExperimentConfig()
    ppo = replace(base.ppo, rollout_steps=rollout_steps, minibatch_size=4)
    if ppo_overrides:
        ppo = replace(ppo, **ppo_overrides)
    a3c = replace(base.a3c, n_steps=n_steps)
    if a3c_overrides:
        a3c = replace(a3c, **a3c_overrides)
    training = replace(
        base.training,
        total_environment_steps=total_steps,
        device=DeviceType.CPU,
        output_directory=str(output_directory),
        tensorboard_directory=None,
        plot_directory=None,
        enable_tensorboard=False,
        save_plots=False,
    )
    if training_overrides:
        training = replace(training, **training_overrides)
    return replace(
        base,
        seed=replace(base.seed, seed=seed),
        environment=replace(
            base.environment,
            observation_shape=observation_shape,
            action_dim=action_dim,
            discrete_actions=discrete,
            num_parallel_envs=num_envs,
        ),
        ppo=ppo,
        bigan=replace(base.bigan, batch_size=bigan_batch_size),
        novelty=replace(
            base.novelty,
            novelty_type=novelty_type,
            transition_variant=transition_variant,
            transition_batch_size=bigan_batch_size,
        ),
        training=training,
        algorithm=algorithm,
        a3c=a3c,
    )


def _segment_from_arrays(
    observations: torch.Tensor,
    actions: torch.Tensor,
    next_observations: torch.Tensor,
    terminated: torch.Tensor,
    extrinsic_rewards: torch.Tensor,
) -> A3CSegment:
    """Assemble an ``A3CSegment`` from explicit arrays for equality tests.

    Input: Observations, actions, next observations, termination mask, and
        extrinsic rewards of ``n`` transitions.
    Output: Segment with zero intrinsic rewards.
    Mathematical meaning: Provides the exact transitions whose intrinsic reward
        two pipelines must agree on.
    """
    count = observations.shape[0]
    return A3CSegment(
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        terminated=terminated,
        truncated=torch.zeros_like(terminated),
        extrinsic_rewards=extrinsic_rewards,
        intrinsic_rewards=torch.zeros(count),
        extrinsic_values=torch.zeros(count),
        intrinsic_values=torch.zeros(count),
        final_observation=next_observations[-1],
    )


class A3CConfigurationTest(unittest.TestCase):
    """Configuration-level contract of the A3C integration."""

    def test_ppo_is_still_the_default_algorithm(self) -> None:
        """The default configuration must keep selecting PPO.

        Input: ``ExperimentConfig()``.
        Output: ``algorithm == "ppo"``; no test-side mutation.
        Mathematical meaning: Preserves the existing experiment default.
        """
        config = ExperimentConfig()
        self.assertEqual(config.algorithm, "ppo")
        self.assertIn("ppo", ALGORITHMS)
        self.assertIn("a3c", ALGORITHMS)

    def test_a3c_adds_no_hidden_hyperparameters(self) -> None:
        """A3C must not introduce new reward or optimisation hyperparameters.

        Input: ``A3CConfig`` field names.
        Output: Exactly the segment length and the BiGAN update cadence.
        Mathematical meaning: Guarantees that A3C reuses ``PPOConfig`` for every
            shared quantity (gamma, lambdas, entropy coefficient, losses, clip,
            learning rate, beta) instead of re-tuning the algorithm.
        """
        self.assertEqual(
            sorted(A3CConfig.__dataclass_fields__),
            ["bigan_update_interval", "n_steps"],
        )

    def test_invalid_algorithm_is_rejected(self) -> None:
        """An unknown algorithm name must fail validation.

        Input: ``algorithm="a2c"``.
        Output: ``ValueError``.
        Mathematical meaning: Prevents silently training the wrong objective.
        """
        with self.assertRaises(ValueError):
            replace(ExperimentConfig(), algorithm="a2c")

    def test_a3c_requires_aligned_segments(self) -> None:
        """The A3C budget must cover whole worker segments.

        Input: ``total_environment_steps`` not divisible by
            ``num_parallel_envs * a3c.n_steps``.
        Output: ``ValueError``.
        Mathematical meaning: Keeps the interaction budget an integer number of
            n-step segments across the actor-learners.
        """
        with self.assertRaises(ValueError):
            replace(
                ExperimentConfig(),
                algorithm="a3c",
                environment=replace(
                    ExperimentConfig().environment, num_parallel_envs=2
                ),
                ppo=replace(ExperimentConfig().ppo, rollout_steps=8),
                training=replace(
                    ExperimentConfig().training, total_environment_steps=250
                ),
                a3c=A3CConfig(n_steps=8),
            )

    def test_trainer_requires_the_matching_algorithm(self) -> None:
        """``A3CTrainer`` must reject a PPO configuration.

        Input: PPO config with an A3C trainer.
        Output: ``ValueError`` from the constructor.
        Mathematical meaning: Keeps the two optimisation rules from being mixed
            by accident.
        """
        directory = Path(tempfile.mkdtemp(prefix="a3c_cfg_"))
        self.addCleanup(shutil.rmtree, directory, True)
        config = _tiny_config(directory, algorithm="ppo")
        environment = _vector_environment(2, 0)
        self.addCleanup(environment.close)
        with self.assertRaises(ValueError):
            A3CTrainer(environment, config, None)

    def test_resettable_is_rejected_for_a3c_only(self) -> None:
        """Algorithm 2's resettable memory has no A3C counterpart.

        Input: A3C config with ``training.resettable=True``.
        Output: ``ValueError`` explaining the limitation.
        Mathematical meaning: Makes an unsupported combination explicit instead
            of silently changing its meaning.
        """
        directory = Path(tempfile.mkdtemp(prefix="a3c_cfg_"))
        self.addCleanup(shutil.rmtree, directory, True)
        config = _tiny_config(
            directory, algorithm="a3c", training_overrides={"resettable": True}
        )
        environment = _vector_environment(2, 0)
        self.addCleanup(environment.close)
        with self.assertRaises(ValueError):
            A3CTrainer(environment, config, None)


class A3CNetworkTest(unittest.TestCase):
    """The actor-critic network, local/global synchronisation, and updates."""

    def setUp(self) -> None:
        """Build a small discrete actor-critic and agent.

        Input: None.
        Output: Initialized attributes on the test case.
        Mathematical meaning: Defines a small parameter point to exercise.
        """
        self.config = ExperimentConfig()
        self.network = A3CActorCritic(
            ActorNetwork(OBSERVATION_SHAPE, ACTION_DIM, True),
            DualCriticNetwork(OBSERVATION_SHAPE),
        )
        self.agent = A3CAgent(
            self.network,
            self.config.ppo,
            total_environment_steps=512,
            num_workers=2,
            n_steps=8,
            beta=0.3,
            device="cpu",
        )

    def tearDown(self) -> None:
        """Clear the deterministic-algorithms flag set by trainer runs.

        Input: This test case.
        Output: No value.
        Mathematical meaning: Restores global PyTorch state expected by other
            test modules.
        """
        torch.use_deterministic_algorithms(False)

    def test_separate_policy_and_value_outputs(self) -> None:
        """The actor-critic must expose distinct policy and value outputs.

        Input: A batch of observations.
        Output: Policy logits ``[B, A]`` and two value vectors ``[B]``.
        Mathematical meaning: Confirms A3C keeps separate actor and critic
            heads, as required by advantage actor-critic.
        """
        observations = torch.randn(5, OBSERVATION_DIM)
        policy, extrinsic_value, intrinsic_value = self.network(observations)
        self.assertEqual(tuple(policy.shape), (5, ACTION_DIM))
        self.assertEqual(tuple(extrinsic_value.shape), (5,))
        self.assertEqual(tuple(intrinsic_value.shape), (5,))
        self.assertFalse(torch.equal(extrinsic_value, intrinsic_value))

    def test_continuous_actions_are_supported(self) -> None:
        """A Gaussian policy must work with a continuous action space.

        Input: Actor-critic with ``discrete_actions=False``.
        Output: Actions shaped ``[B, A]`` and one log-probability/entropy per
            row.
        Mathematical meaning: Verifies A3C does not assume discrete actions.
        """
        network = A3CActorCritic(
            ActorNetwork(OBSERVATION_SHAPE, ACTION_DIM, False),
            DualCriticNetwork(OBSERVATION_SHAPE),
        )
        observation = torch.randn(OBSERVATION_DIM)
        action, log_probability, entropy, extrinsic_value, intrinsic_value = network.act(
            observation
        )
        self.assertEqual(tuple(action.shape), (1, ACTION_DIM))
        self.assertEqual(tuple(log_probability.shape), (1,))
        self.assertEqual(tuple(entropy.shape), (1,))
        self.assertEqual(tuple(extrinsic_value.shape), (1,))
        self.assertEqual(tuple(intrinsic_value.shape), (1,))

    def test_local_networks_are_independent_and_synchronisable(self) -> None:
        """Local copies must be private snapshots of the shared network.

        Input: Two local networks and one shared-parameter update.
        Output: The untouched local copy stays stale and a synchronised copy
            matches the shared network.
        Mathematical meaning: Implements A3C's parameter pull/push semantics.
        """
        first = self.agent.new_local_network()
        second = self.agent.new_local_network()
        self.assertIsNot(first.actor, self.agent.network.actor)
        with torch.no_grad():
            for parameter in self.agent.network.parameters():
                parameter.add_(0.01)
        self.assertFalse(
            torch.allclose(
                next(first.actor.parameters()), next(self.agent.network.parameters())
            )
        )
        self.agent.synchronize_local(second)
        for local_parameter, global_parameter in zip(
            second.parameters(), self.agent.network.parameters()
        ):
            self.assertTrue(torch.allclose(local_parameter, global_parameter))

    def test_update_moves_global_parameters_with_finite_metrics(self) -> None:
        """One A3C update must be finite and must change the shared network.

        Input: A synthetic eight-step segment.
        Output: Finite ``A3CUpdateMetrics`` and updated global parameters.
        Mathematical meaning: Confirms the actor-critic gradient step is
            well-defined and applied.
        """
        local = self.agent.new_local_network()
        observations = torch.randn(8, OBSERVATION_DIM)
        actions = torch.randint(0, ACTION_DIM, (8,))
        segment = _segment_from_arrays(
            observations,
            actions,
            torch.randn(8, OBSERVATION_DIM),
            torch.zeros(8, dtype=torch.bool),
            torch.randn(8),
        )
        segment = replace(segment, intrinsic_rewards=torch.randn(8))
        estimate = self.agent.estimate_returns(
            segment, torch.zeros(()), torch.zeros(())
        )
        before = [parameter.detach().clone() for parameter in self.agent.network.parameters()]
        metrics = self.agent.update(local, segment.observations, segment.actions, estimate)
        after = [parameter.detach().clone() for parameter in self.agent.network.parameters()]
        for name in (
            "policy_loss",
            "extrinsic_value_loss",
            "intrinsic_value_loss",
            "value_loss",
            "entropy",
            "total_loss",
            "gradient_norm",
            "learning_rate",
            "advantage_mean",
            "advantage_std",
        ):
            self.assertTrue(math.isfinite(getattr(metrics, name)), name)
        self.assertGreater(metrics.gradient_norm, 0.0)
        self.assertTrue(
            any(
                not torch.allclose(previous, current)
                for previous, current in zip(before, after)
            )
        )
        self.assertEqual(self.agent.applied_updates, 1)

    def test_zero_beta_uses_extrinsic_advantages_only(self) -> None:
        """``beta = 0`` must reduce the policy advantage to the extrinsic stream.

        Input: Two-stream estimate with ``intrinsic_advantage_coefficient=0``.
        Output: ``combined_advantages == extrinsic.advantages``.
        Mathematical meaning: This is the project's existing "standard agent
            configuration" (no intrinsic exploration) expressed through the
            unchanged PPO coefficient, so A3C needs no new switch.
        """
        agent = A3CAgent(
            self.network,
            self.config.ppo,
            total_environment_steps=512,
            num_workers=2,
            n_steps=8,
            beta=0.0,
            device="cpu",
        )
        segment = _segment_from_arrays(
            torch.randn(8, OBSERVATION_DIM),
            torch.randint(0, ACTION_DIM, (8,)),
            torch.randn(8, OBSERVATION_DIM),
            torch.zeros(8, dtype=torch.bool),
            torch.randn(8),
        )
        segment = replace(segment, intrinsic_rewards=torch.randn(8))
        estimate = agent.estimate_returns(segment, torch.zeros(()), torch.zeros(()))
        self.assertTrue(
            torch.allclose(
                estimate.combined_advantages, estimate.extrinsic.advantages
            )
        )

    def test_generated_action_stream_is_reproducible(self) -> None:
        """A per-worker generator must make action sampling repeatable.

        Input: Two generators seeded identically.
        Output: Identical action sequences.
        Mathematical meaning: Confirms each actor-learner owns a deterministic
            random stream.
        """
        observation = torch.randn(OBSERVATION_DIM)
        sequences = []
        for _ in range(2):
            generator = torch.Generator()
            generator.manual_seed(1234)
            actions = [
                int(self.network.act(observation, generator)[0].reshape(-1)[0])
                for _ in range(10)
            ]
            sequences.append(actions)
        self.assertEqual(sequences[0], sequences[1])


class A3CExplorationEquivalenceTest(unittest.TestCase):
    """A3C must receive exactly the novelty PPO's pipeline defines."""

    def tearDown(self) -> None:
        """Clear the deterministic-algorithms flag set by trainer construction.

        Input: This test case.
        Output: No value.
        Mathematical meaning: Restores global PyTorch state.
        """
        torch.use_deterministic_algorithms(False)

    def _paired_trainers(
        self,
        novelty_type: str,
        transition_variant: str,
    ) -> Tuple[AdventurerTrainer, A3CTrainer, VectorEnvironmentAdapter, VectorEnvironmentAdapter]:
        """Build a PPO and an A3C trainer over identical exploration settings.

        Input: Novelty type and transition variant.
        Output: ``(ppo_trainer, a3c_trainer, ppo_env, a3c_env)`` with the A3C
            trainer's BiGAN (and forward model) weights copied from PPO's.
        Mathematical meaning: Isolates the reward pipeline so that only the
            optimisation rule differs between the two trainers.
        """
        directory = Path(tempfile.mkdtemp(prefix="a3c_equiv_"))
        self.addCleanup(shutil.rmtree, directory, True)
        ppo_config = _tiny_config(
            directory / "ppo",
            algorithm="ppo",
            novelty_type=novelty_type,
            transition_variant=transition_variant,
            total_steps=128,
        )
        a3c_config = replace(
            ppo_config,
            algorithm="a3c",
            training=replace(ppo_config.training, output_directory=str(directory / "a3c")),
        )
        ppo_environment = _vector_environment(2, 0)
        a3c_environment = _vector_environment(2, 0)
        self.addCleanup(ppo_environment.close)
        self.addCleanup(a3c_environment.close)
        ppo_trainer = AdventurerTrainer(ppo_environment, ppo_config, None)
        a3c_trainer = A3CTrainer(a3c_environment, a3c_config, None)
        self.addCleanup(ppo_trainer.close)
        a3c_trainer.encoder.load_state_dict(ppo_trainer.encoder.state_dict())
        a3c_trainer.generator.load_state_dict(ppo_trainer.generator.state_dict())
        a3c_trainer.discriminator.load_state_dict(ppo_trainer.discriminator.state_dict())
        if transition_variant == "master_l2":
            assert ppo_trainer.master_transition_novelty is not None
            assert a3c_trainer.master_transition_novelty is not None
            a3c_trainer.master_transition_novelty.forward_model.load_state_dict(
                ppo_trainer.master_transition_novelty.forward_model.state_dict()
            )
        return ppo_trainer, a3c_trainer, ppo_environment, a3c_environment

    def _assert_state_novelty_matches(self, trainer: A3CTrainer, ppo_trainer: AdventurerTrainer) -> None:
        """Compare A3C's state-novelty rewards with PPO's pipeline.

        Input: A3C trainer and PPO trainer sharing BiGAN weights.
        Output: No value; assertions fail on any numerical difference.
        Mathematical meaning: Verifies that Equation (5)-normalised BiGAN state
            novelty ``B(s)`` is bit-for-bit the same reward in both trainers.
        """
        torch.manual_seed(7)
        observations = torch.rand(8, *OBSERVATION_SHAPE)
        next_observations = torch.rand(8, *OBSERVATION_SHAPE)
        actions = torch.randint(0, ACTION_DIM, (8,))
        rewards = torch.rand(8)
        segment = _segment_from_arrays(
            observations,
            actions,
            next_observations,
            torch.zeros(8, dtype=torch.bool),
            rewards,
        )
        a3c_reward = trainer.segment_intrinsic_rewards(segment)
        assert ppo_trainer.novelty is not None
        ppo_novelty = ppo_trainer.novelty(
            next_observations, update_statistics=True, extrinsic_reward=rewards
        )
        ppo_reward = ppo_trainer.reward_pipeline(rewards, ppo_novelty)
        self.assertTrue(
            torch.allclose(a3c_reward.intrinsic_reward, ppo_reward.intrinsic_reward)
        )
        self.assertTrue(torch.allclose(a3c_reward.extrinsic_reward, rewards))
        self.assertTrue(
            torch.allclose(
                a3c_reward.normalized_novelty, ppo_reward.normalized_novelty
            )
        )
        self.assertTrue(
            torch.allclose(
                ppo_trainer.novelty.normalized_score.normalizer.statistics.mean,
                trainer.novelty.normalized_score.normalizer.statistics.mean,
            )
        )

    def test_state_novelty_reward_is_identical(self) -> None:
        """BiGAN state novelty must be unchanged by the A3C integration.

        Input: A PPO trainer and an A3C trainer with identical BiGAN weights.
        Output: Identical pixel, feature, and Equation (5) rewards.
        Mathematical meaning: Test 3 -- A3C receives and uses the existing
            state-novelty bonus.
        """
        ppo_trainer, a3c_trainer, _, _ = self._paired_trainers("state", "legacy")
        self._assert_state_novelty_matches(a3c_trainer, ppo_trainer)

    def test_master_transition_novelty_reward_is_identical(self) -> None:
        """The Master's ``N_T`` must be unchanged by the A3C integration.

        Input: PPO and A3C trainers with identical BiGAN and forward-model
            weights, ``novelty_type="transition"``, ``variant="master_l2"``.
        Output: Identical ``||f(E(s_t),a_t)-E(s_(t+1))||_2`` rewards.
        Mathematical meaning: Test 4 -- A3C receives and uses the existing
            transition-novelty bonus with the same Equation (5) mechanism.
        """
        ppo_trainer, a3c_trainer, _, _ = self._paired_trainers(
            "transition", "master_l2"
        )
        torch.manual_seed(11)
        observations = torch.rand(8, *OBSERVATION_SHAPE)
        next_observations = torch.rand(8, *OBSERVATION_SHAPE)
        actions = torch.randint(0, ACTION_DIM, (8,))
        rewards = torch.rand(8)
        segment = _segment_from_arrays(
            observations,
            actions,
            next_observations,
            torch.zeros(8, dtype=torch.bool),
            rewards,
        )
        a3c_reward = a3c_trainer.segment_intrinsic_rewards(segment)
        assert ppo_trainer.master_transition_novelty is not None
        ppo_novelty = ppo_trainer.master_transition_novelty(
            observations, actions, next_observations, rewards
        )
        ppo_reward = ppo_trainer.reward_pipeline(rewards, ppo_novelty)
        self.assertTrue(
            torch.allclose(a3c_reward.intrinsic_reward, ppo_reward.intrinsic_reward)
        )
        self.assertTrue(
            torch.allclose(
                a3c_reward.normalized_novelty, ppo_reward.normalized_novelty
            )
        )
        # The cached latent transition buffer must have received the segment,
        # i.e. A3C feeds the unchanged forward-model training path.
        assert a3c_trainer.master_transition_buffer is not None
        self.assertEqual(len(a3c_trainer.master_transition_buffer), len(segment))

    def test_legacy_transition_novelty_reward_is_identical(self) -> None:
        """The legacy transition novelty must also be preserved for A3C.

        Input: PPO and A3C trainers with identical BiGAN weights and
            ``novelty_type="transition"`` (legacy variant).
        Output: Identical intrinsic rewards.
        Mathematical meaning: Confirms the pre-existing novelty variants remain
            available to A3C without reformulation.
        """
        ppo_trainer, a3c_trainer, _, _ = self._paired_trainers(
            "transition", "legacy"
        )
        torch.manual_seed(13)
        observations = torch.rand(8, *OBSERVATION_SHAPE)
        next_observations = torch.rand(8, *OBSERVATION_SHAPE)
        actions = torch.randint(0, ACTION_DIM, (8,))
        rewards = torch.rand(8)
        segment = _segment_from_arrays(
            observations,
            actions,
            next_observations,
            torch.zeros(8, dtype=torch.bool),
            rewards,
        )
        a3c_reward = a3c_trainer.segment_intrinsic_rewards(segment)
        assert ppo_trainer.transition_novelty is not None
        ppo_novelty = ppo_trainer.transition_novelty(
            observations, actions, next_observations, rewards
        )
        ppo_reward = ppo_trainer.reward_pipeline(rewards, ppo_novelty)
        self.assertTrue(
            torch.allclose(a3c_reward.intrinsic_reward, ppo_reward.intrinsic_reward)
        )


class A3CPpoPreservationTest(unittest.TestCase):
    """The A3C configuration must be inert for PPO runs."""

    def tearDown(self) -> None:
        """Clear the deterministic-algorithms flag set by the runs.

        Input: This test case.
        Output: No value.
        Mathematical meaning: Restores global PyTorch state.
        """
        torch.use_deterministic_algorithms(False)

    def _run_ppo(self, directory: Path, a3c_config: A3CConfig) -> List[Any]:
        """Run one tiny PPO training and return its per-update metrics.

        Input: Output directory and the A3C configuration section.
        Output: Per-update ``TrainingIterationMetrics``.
        Mathematical meaning: Executes the unchanged PPO optimisation path.
        """
        config = _tiny_config(
            directory, algorithm="ppo", total_steps=128, a3c_overrides=None
        )
        config = replace(config, a3c=a3c_config)
        environment = _vector_environment(config.environment.num_parallel_envs, 0)
        with ExperimentLogger(directory, config) as logger:
            trainer = AdventurerTrainer(environment, config, logger)
            results = trainer.train()
            trainer.close()
        return results

    @staticmethod
    def _summary(results: List[Any]) -> List[Tuple[float, ...]]:
        """Reduce PPO metrics to comparable tuples.

        Input: Per-update metrics.
        Output: Tuples of the objective and reward values.
        Mathematical meaning: Extracts the numbers that must not change.
        """
        return [
            (
                item.ppo.policy_loss,
                item.ppo.value_loss,
                item.ppo.total_loss,
                item.ppo.entropy,
                item.ppo.actor_gradient_norm,
                item.extrinsic_reward,
                item.intrinsic_reward,
                item.normalized_novelty,
            )
            for item in results
        ]

    def test_a3c_settings_do_not_change_a_ppo_run(self) -> None:
        """Changing A3C-only settings must leave a PPO run identical.

        Input: The same PPO run with two different ``A3CConfig`` values.
        Output: Bit-identical objective and reward series.
        Mathematical meaning: Proves the A3C integration cannot perturb the
            existing PPO optimisation path.
        """
        first = Path(tempfile.mkdtemp(prefix="a3c_ppo_a_"))
        second = Path(tempfile.mkdtemp(prefix="a3c_ppo_b_"))
        self.addCleanup(shutil.rmtree, first, True)
        self.addCleanup(shutil.rmtree, second, True)
        default_run = self._run_ppo(first, A3CConfig())
        perturbed_run = self._run_ppo(
            second, A3CConfig(n_steps=3, bigan_update_interval=5)
        )
        self.assertEqual(len(default_run), len(perturbed_run))
        self.assertEqual(self._summary(default_run), self._summary(perturbed_run))
        for metrics in default_run:
            self.assertTrue(math.isfinite(metrics.ppo.total_loss))
            self.assertTrue(math.isfinite(metrics.ppo.actor_gradient_norm))

    def test_ppo_run_records_the_ppo_algorithm(self) -> None:
        """A PPO run must still record its algorithm in ``config.json``.

        Input: A tiny PPO run.
        Output: ``config.json`` with ``algorithm == "ppo"``.
        Mathematical meaning: Keeps the experiment record explicit without
            changing the optimisation.
        """
        directory = Path(tempfile.mkdtemp(prefix="a3c_ppo_cfg_"))
        self.addCleanup(shutil.rmtree, directory, True)
        self._run_ppo(directory, A3CConfig())
        configuration = json.loads((directory / "config.json").read_text())
        self.assertEqual(configuration["algorithm"], "ppo")
        self.assertIn("a3c", configuration)


class A3CEndToEndTest(unittest.TestCase):
    """Full A3C runs: standard, state novelty, transition novelty, stability."""

    def tearDown(self) -> None:
        """Clear the deterministic-algorithms flag set by the runs.

        Input: This test case.
        Output: No value.
        Mathematical meaning: Restores global PyTorch state.
        """
        torch.use_deterministic_algorithms(False)

    def _run(
        self,
        directory: Path,
        novelty_type: str = "state",
        transition_variant: str = "legacy",
        num_envs: int = 2,
        n_steps: int = 8,
        total_steps: int = 256,
        seed: int = 0,
        discrete: bool = True,
        observation_shape: Tuple[int, ...] = OBSERVATION_SHAPE,
        action_dim: int = ACTION_DIM,
        ppo_overrides: Optional[Dict[str, Any]] = None,
        training_overrides: Optional[Dict[str, Any]] = None,
        logger_enabled: bool = True,
    ) -> Tuple[List[Any], A3CTrainer, Path]:
        """Execute one tiny A3C run and return its diagnostics.

        Input: Output directory and run settings.
        Output: ``(metrics, trainer, directory)``.
        Mathematical meaning: Runs the complete asynchronous optimisation loop
            for a few segments and exposes its diagnostics.
        """
        config = _tiny_config(
            directory,
            algorithm="a3c",
            novelty_type=novelty_type,
            transition_variant=transition_variant,
            seed=seed,
            num_envs=num_envs,
            n_steps=n_steps,
            total_steps=total_steps,
            discrete=discrete,
            observation_shape=observation_shape,
            action_dim=action_dim,
            ppo_overrides=ppo_overrides,
            training_overrides=training_overrides,
        )
        environment = _vector_environment(
            config.environment.num_parallel_envs,
            config.seed.seed,
            observation_shape,
            action_dim,
            discrete,
        )
        if logger_enabled:
            logger = ExperimentLogger(directory, config)
            write_run_info(
                directory,
                config,
                environment_id="TinyA3C-v0",
                device_label="cpu",
                extra={"algorithm": config.algorithm},
            )
        else:
            logger = None
        trainer = A3CTrainer(environment, config, logger)
        results = trainer.train()
        if logger is not None:
            logger.close()
        return results, trainer, directory

    def _assert_healthy_run(self, results: List[Any], trainer: A3CTrainer) -> None:
        """Assert the stability properties every A3C run must satisfy.

        Input: Per-record metrics and the finished trainer.
        Output: No value; assertions fail the test on instability.
        Mathematical meaning: Checks that policy and value objectives, gradient
            norms, and reward streams stay finite, that the shared parameters
            actually moved, and that every actor-learner synchronised.
        """
        self.assertGreater(len(results), 0)
        for metrics in results:
            for name in (
                "policy_loss",
                "extrinsic_value_loss",
                "intrinsic_value_loss",
                "value_loss",
                "entropy",
                "total_loss",
                "gradient_norm",
                "advantage_mean",
                "advantage_std",
            ):
                self.assertTrue(
                    math.isfinite(getattr(metrics.a3c, name)),
                    f"{name} is not finite",
                )
            self.assertTrue(math.isfinite(metrics.extrinsic_reward))
            self.assertTrue(math.isfinite(metrics.intrinsic_reward))
            self.assertTrue(math.isfinite(metrics.total_reward))
            self.assertTrue(math.isfinite(metrics.pixel_novelty))
            self.assertTrue(math.isfinite(metrics.feature_novelty))
            self.assertGreaterEqual(metrics.a3c.entropy, 0.0)
        self.assertTrue(
            any(metrics.a3c.gradient_norm > 0.0 for metrics in results),
            "no A3C update produced a non-zero gradient norm",
        )
        self.assertGreater(trainer.agent.applied_updates, 0)
        for worker in trainer._workers:
            self.assertIsNone(worker.error)
            self.assertGreater(worker.segments_completed, 0)
            self.assertFalse(worker.is_alive)
        self.assertEqual(
            trainer.agent.applied_updates,
            sum(worker.segments_completed for worker in trainer._workers),
        )
        # Every actor-learner's environment was actually stepped.
        self.assertEqual(
            trainer.environment_steps,
            trainer.config.training.total_environment_steps,
        )

    def test_standard_a3c_without_intrinsic_advantage(self) -> None:
        """Test 2: standard A3C with no intrinsic exploration advantage.

        Input: ``intrinsic_advantage_coefficient=0.0`` (the project's existing
            switch for the standard agent configuration).
        Output: A completed, stable run with a non-zero extrinsic reward.
        Mathematical meaning: Trains the ordinary advantage actor-critic
            objective while the exploration pieces stay available.
        """
        directory = Path(tempfile.mkdtemp(prefix="a3c_standard_"))
        self.addCleanup(shutil.rmtree, directory, True)
        results, trainer, _ = self._run(
            directory, ppo_overrides={"intrinsic_advantage_coefficient": 0.0}
        )
        self._assert_healthy_run(results, trainer)
        self.assertEqual(trainer.agent.beta, 0.0)
        self.assertTrue(any(metrics.extrinsic_reward > 0.0 for metrics in results))

    def test_state_novelty_run(self) -> None:
        """Test 3 end to end: BiGAN state novelty drives the A3C intrinsic
        reward and is logged.

        Input: ``novelty_type="state"``.
        Output: A stable run whose logged novelty and intrinsic reward are
            finite and non-trivial.
        Mathematical meaning: Confirms the state-novelty bonus reaches the
            actor-learner's advantage.
        """
        directory = Path(tempfile.mkdtemp(prefix="a3c_state_"))
        self.addCleanup(shutil.rmtree, directory, True)
        results, trainer, path = self._run(directory, novelty_type="state")
        self._assert_healthy_run(results, trainer)
        self.assertTrue(any(abs(metrics.intrinsic_reward) > 0.0 for metrics in results))
        self.assertTrue(any(metrics.pixel_novelty > 0.0 for metrics in results))
        records = [
            json.loads(line)
            for line in (path / "metrics.jsonl").read_text().splitlines()
        ]
        self.assertTrue(any("novelty/pixel" in record["metrics"] for record in records))
        self.assertTrue(any("a3c" in record["metrics"] for record in records))
        for name in ("config.json", "metrics.jsonl", "run.log", "run_info.json"):
            self.assertTrue((path / name).is_file(), f"missing {name}")

    def test_transition_novelty_run(self) -> None:
        """Test 4 end to end: the Master's transition novelty drives A3C.

        Input: ``--method transition`` equivalent
            (``novelty_type="transition"``, ``transition_variant="master_l2"``).
        Output: A stable run that also optimises the latent forward model.
        Mathematical meaning: Confirms ``N_T`` reaches the actor-learner and the
            forward-model objective keeps improving.
        """
        directory = Path(tempfile.mkdtemp(prefix="a3c_transition_"))
        self.addCleanup(shutil.rmtree, directory, True)
        results, trainer, _ = self._run(
            directory, novelty_type="transition", transition_variant="master_l2"
        )
        self._assert_healthy_run(results, trainer)
        forward_model_losses = [
            metrics.transition.forward_model_mse
            for metrics in results
            if metrics.transition is not None and metrics.transition.updated
        ]
        self.assertTrue(forward_model_losses)
        for value in forward_model_losses:
            self.assertTrue(math.isfinite(value))
        self.assertTrue(any(metrics.intrinsic_reward != 0.0 for metrics in results))

    def test_metric_bonus_run_is_supported(self) -> None:
        """The legacy latent-discrepancy (metric/EME) bonus also works with A3C.

        Input: ``metric_eme.enabled=True`` with the ensemble scaling off.
        Output: A stable run that logs the latent-distance bonus.
        Mathematical meaning: Confirms A3C can use the project's existing
            combined exploration configuration without a new formulation.
        """
        directory = Path(tempfile.mkdtemp(prefix="a3c_metric_"))
        self.addCleanup(shutil.rmtree, directory, True)
        config = _tiny_config(directory, algorithm="a3c")
        config = replace(
            config,
            metric_eme=replace(config.metric_eme, enabled=True, ensemble_scaling=False),
        )
        environment = _vector_environment(config.environment.num_parallel_envs, 0)
        trainer = A3CTrainer(environment, config, None)
        results = trainer.train()
        trainer.close()
        self._assert_healthy_run(results, trainer)
        self.assertTrue(
            any(metrics.metric_eme is not None for metrics in results)
        )

    def test_continuous_action_run(self) -> None:
        """A3C must train with a continuous action space.

        Input: A tiny environment with ``spaces.Box`` actions.
        Output: A stable run.
        Mathematical meaning: Confirms the Gaussian policy path is exercised
            end to end, so the integration does not assume discrete actions.
        """
        directory = Path(tempfile.mkdtemp(prefix="a3c_continuous_"))
        self.addCleanup(shutil.rmtree, directory, True)
        results, trainer, _ = self._run(directory, discrete=False)
        self._assert_healthy_run(results, trainer)

    def test_image_observation_run(self) -> None:
        """A3C must work with the convolutional observation path.

        Input: 40x40x3 observations (the smallest frame size the shared
            convolutional stack accepts).
        Output: A stable run.
        Mathematical meaning: Confirms the shared image preprocessing and
            convolutional encoder are used unchanged by A3C.
        """
        directory = Path(tempfile.mkdtemp(prefix="a3c_image_"))
        self.addCleanup(shutil.rmtree, directory, True)
        results, trainer, _ = self._run(
            directory,
            observation_shape=(40, 40, 3),
            total_steps=128,
        )
        self._assert_healthy_run(results, trainer)

    def test_three_workers_synchronise_without_deadlock(self) -> None:
        """Test 5: several actor-learners share one global network.

        Input: Three actor-learners with a 384-step budget.
        Output: Every worker contributes, the shared optimiser step count
            matches the worker update count, and no worker failed.
        Mathematical meaning: Confirms the asynchronous shared-parameter
            update is correctly serialised.
        """
        directory = Path(tempfile.mkdtemp(prefix="a3c_workers_"))
        self.addCleanup(shutil.rmtree, directory, True)
        results, trainer, _ = self._run(
            directory, num_envs=3, total_steps=384
        )
        self._assert_healthy_run(results, trainer)
        self.assertEqual(len(trainer._workers), 3)
        self.assertTrue(all(worker.episodes_completed > 0 for worker in trainer._workers))

    def test_episodes_terminate_and_are_logged(self) -> None:
        """Episodes must terminate and be recorded in the metrics stream.

        Input: A tiny episode limit of five transitions.
        Output: Recorded ``episode/game_score`` entries and finite scores.
        Mathematical meaning: Confirms episode termination handling and the
            unchanged logging convention.
        """
        directory = Path(tempfile.mkdtemp(prefix="a3c_episodes_"))
        self.addCleanup(shutil.rmtree, directory, True)
        _, trainer, path = self._run(directory, total_steps=128)
        records = [
            json.loads(line)
            for line in (path / "metrics.jsonl").read_text().splitlines()
        ]
        episodes = [
            record["metrics"]
            for record in records
            if "episode/game_score" in record["metrics"]
        ]
        self.assertGreater(len(episodes), 0)
        for episode in episodes:
            self.assertTrue(math.isfinite(episode["episode/game_score"]))
            self.assertGreater(episode["episode/length"], 0)
        self.assertTrue(all(worker.episodes_completed > 0 for worker in trainer._workers))


class A3CReproducibilityTest(unittest.TestCase):
    """Seeding across actor-learners and run-to-run reproducibility."""

    def tearDown(self) -> None:
        """Clear the deterministic-algorithms flag set by the runs.

        Input: This test case.
        Output: No value.
        Mathematical meaning: Restores global PyTorch state.
        """
        torch.use_deterministic_algorithms(False)

    def _run_single_worker(self, directory: Path, seed: int) -> List[Any]:
        """Run a one-actor-learner A3C experiment.

        Input: Output directory and experiment seed.
        Output: Per-record metrics.
        Mathematical meaning: With one actor-learner there is no interleaving,
            so the run must be bit-for-bit repeatable.
        """
        config = _tiny_config(
            directory,
            algorithm="a3c",
            seed=seed,
            num_envs=1,
            n_steps=8,
            total_steps=128,
            training_overrides={"log_interval_updates": 1},
        )
        environment = _vector_environment(1, config.seed.seed)
        trainer = A3CTrainer(environment, config, None)
        results = trainer.train()
        trainer.close()
        return results

    @staticmethod
    def _summary(results: List[Any]) -> List[Tuple[float, ...]]:
        """Reduce A3C metrics to comparable tuples.

        Input: Per-record metrics.
        Output: Tuples of losses, gradient norms, and reward streams.
        Mathematical meaning: Extracts the numbers that reproducibility must
            preserve.
        """
        return [
            (
                item.a3c.policy_loss,
                item.a3c.value_loss,
                item.a3c.total_loss,
                item.a3c.gradient_norm,
                item.extrinsic_reward,
                item.intrinsic_reward,
                item.pixel_novelty,
            )
            for item in results
        ]

    def test_same_seed_repeats_a_single_worker_run(self) -> None:
        """Test 6: a fixed seed reproduces a single-actor-learner run.

        Input: Two runs with ``--seed 3``.
        Output: Identical objective and reward series.
        Mathematical meaning: Confirms seed handling through environment
            resets, policy sampling, and the shared optimiser.
        """
        first = Path(tempfile.mkdtemp(prefix="a3c_rep_1_"))
        second = Path(tempfile.mkdtemp(prefix="a3c_rep_2_"))
        self.addCleanup(shutil.rmtree, first, True)
        self.addCleanup(shutil.rmtree, second, True)
        first_run = self._run_single_worker(first, seed=3)
        second_run = self._run_single_worker(second, seed=3)
        self.assertGreater(len(first_run), 0)
        self.assertEqual(len(first_run), len(second_run))
        self.assertEqual(self._summary(first_run), self._summary(second_run))

    def test_different_seeds_diverge(self) -> None:
        """Different seeds must produce different trajectories.

        Input: Seed 3 versus seed 4.
        Output: Different reward or objective series.
        Mathematical meaning: Confirms the seed actually drives the sampled
            environment and policy streams.
        """
        first = Path(tempfile.mkdtemp(prefix="a3c_rep_3_"))
        second = Path(tempfile.mkdtemp(prefix="a3c_rep_4_"))
        self.addCleanup(shutil.rmtree, first, True)
        self.addCleanup(shutil.rmtree, second, True)
        first_run = self._run_single_worker(first, seed=3)
        other_run = self._run_single_worker(second, seed=4)
        self.assertNotEqual(self._summary(first_run), self._summary(other_run))

    def test_worker_streams_are_distinct(self) -> None:
        """Each actor-learner must derive distinct random streams.

        Input: Base seed and worker indices.
        Output: Pairwise distinct environment and action-sampling seeds.
        Mathematical meaning: Guarantees independent sampling streams across
            the asynchronous actor-learners.
        """
        seed = 5
        environment_seeds = [
            derive_seed(derive_seed(seed, 0xA3C), index) for index in range(4)
        ]
        action_seeds = [
            derive_seed(derive_seed(seed, 0x5A3), index) for index in range(4)
        ]
        self.assertEqual(len(set(environment_seeds)), 4)
        self.assertEqual(len(set(action_seeds)), 4)
        self.assertEqual(len(set(environment_seeds) & set(action_seeds)), 0)


class A3CExperimentScriptTest(unittest.TestCase):
    """``scripts/run_master_a3c.sh``: the A3C driver of the Master's experiment.

    The driver must (a) launch both conditions with ``--algorithm a3c``, (b) keep
    the results layout and aggregation artefacts of
    ``scripts/run_master_experiments.sh``, and (c) refuse an inconsistent
    schedule before spending any compute.
    """

    SCRIPT = Path("scripts") / "run_master_a3c.sh"

    def tearDown(self) -> None:
        """Clear the deterministic-algorithms flag set by the runs.

        Input: This test case.
        Output: No value.
        Mathematical meaning: Restores global PyTorch state.
        """
        torch.use_deterministic_algorithms(False)

    def _tiny_script_environment(self) -> Dict[str, str]:
        """Return the environment variables of a seconds-long driver run.

        Input: None.
        Output: Mapping of the driver's small-scale settings.
        Mathematical meaning: Executes the same code path as the thesis run at a
            scale where two conditions finish in seconds on CPU.
        """
        return {
            "PYTHON": sys.executable,
            "ENV_NAME": "CartPole-v1",
            "DEVICE": "cpu",
            "NUM_WORKERS": "2",
            "N_STEPS": "8",
            "TOTAL_STEPS": "128",
            "ROLLOUT_STEPS": "8",
            "MINIBATCH_SIZE": "4",
            "BIGAN_BATCH_SIZE": "8",
            "SEEDS": "0",
            "METHODS": "state transition",
            "DISABLE_TENSORBOARD": "1",
            "DISABLE_PLOTS": "1",
        }

    def test_script_is_syntactically_valid(self) -> None:
        """The driver must be valid bash.

        Input: ``bash -n scripts/run_master_a3c.sh``.
        Output: Exit status ``0``.
        Mathematical meaning: None; a syntax error would only surface after a
            long experiment had been prepared.
        """
        completed = subprocess.run(
            ["bash", "-n", str(self.SCRIPT)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_inconsistent_schedule_fails_before_training(self) -> None:
        """An A3C schedule that does not divide the budget must be rejected.

        Input: ``TOTAL_STEPS=130`` with ``NUM_WORKERS*N_STEPS=16``.
        Output: Non-zero exit status and an explanatory message; no run
            directory is created.
        Mathematical meaning: Mirrors ``ExperimentConfig``'s A3C divisibility
            rule in the shell layer so a broken sweep fails immediately.
        """
        directory = Path(tempfile.mkdtemp(prefix="a3c_script_bad_"))
        self.addCleanup(shutil.rmtree, directory, True)
        environment = dict(self._tiny_script_environment())
        environment.update(
            {
                "TOTAL_STEPS": "130",
                "METHODS": "state",
                "OUTPUT_DIR": str(directory / "results"),
                "RUN_AGGREGATION": "0",
            }
        )
        completed = subprocess.run(
            ["bash", str(self.SCRIPT)],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, **environment},
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("must be divisible by", completed.stdout + completed.stderr)
        self.assertFalse((directory / "results" / "state" / "seed_0").exists())

    def test_driver_trains_both_conditions_and_aggregates(self) -> None:
        """The driver runs both Master's conditions with A3C end to end.

        Input: A tiny two-condition driver run on CartPole.
        Output: Exit status ``0``; per-condition runs, CSVs, and figures that
            record the A3C algorithm.
        Mathematical meaning: Confirms the A3C experiment pipeline -- including
            the shared BiGAN/novelty/Equation (5) reward path -- is launchable
            through the same interface as the PPO experiment.
        """
        directory = Path(tempfile.mkdtemp(prefix="a3c_script_"))
        self.addCleanup(shutil.rmtree, directory, True)
        results = directory / "results"
        environment = dict(self._tiny_script_environment())
        environment["OUTPUT_DIR"] = str(results)
        completed = subprocess.run(
            ["bash", str(self.SCRIPT)],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, **environment},
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

        for method in ("state", "transition"):
            run_directory = results / method / "seed_0"
            for name in ("config.json", "run_info.json", "metrics.jsonl", "run.log", "training.log"):
                self.assertTrue((run_directory / name).is_file(), f"missing {method}/{name}")
            configuration = json.loads((run_directory / "config.json").read_text())
            self.assertEqual(configuration["algorithm"], "a3c")
            self.assertEqual(configuration["a3c"]["n_steps"], 8)
            run_info = json.loads((run_directory / "run_info.json").read_text())
            self.assertEqual(run_info["algorithm"], "a3c")
            records = [
                json.loads(line)
                for line in (run_directory / "metrics.jsonl").read_text().splitlines()
            ]
            self.assertTrue(any("a3c" in record["metrics"] for record in records))
            self.assertFalse(
                any("ppo" in record["metrics"] for record in records),
                "an A3C run must not log PPO optimisation metrics",
            )

        summary = results / "master_summary.csv"
        self.assertTrue(summary.is_file())
        self.assertTrue((results / "master_summary_stats.csv").is_file())
        rows = list(csv.DictReader(summary.open(encoding="utf-8")))
        self.assertEqual([row["method"] for row in rows], ["state", "transition"])

        # The A3C sample geometry must be used: one logged record covers
        # NUM_WORKERS * N_STEPS transitions, not the PPO rollout geometry.
        for row in rows:
            run_directory = results / row["method"] / "seed_0"
            records = [
                json.loads(line)
                for line in (run_directory / "metrics.jsonl").read_text().splitlines()
            ]
            means = [
                record["metrics"]["reward/extrinsic"]
                for record in records
                if "reward/extrinsic" in record["metrics"]
            ]
            expected = sum(means) * 2 * 8
            self.assertAlmostEqual(
                float(row["total_extrinsic_return"]), expected, places=3
            )

        figures = results / "figures"
        self.assertTrue((figures / "plot1_game_score_vs_steps.png").is_file())

        # The per-seed metric table must carry the A3C diagnostics.
        for method in ("state", "transition"):
            header = (results / method / "seed_0" / "metrics.csv").read_text(
                encoding="utf-8"
            ).splitlines()[0]
            self.assertIn("a3c/policy_loss", header)
            self.assertIn("a3c/gradient_norm", header)
            self.assertNotIn("ppo/policy_loss", header)

    def test_driver_can_run_the_matched_ppo_control(self) -> None:
        """``ALGORITHM=ppo`` gives the matched control through the same driver.

        Input: A tiny one-condition driver run with ``ALGORITHM=ppo``.
        Output: A PPO run that records ``algorithm="ppo"`` and keeps the PPO
            metric columns.
        Mathematical meaning: Confirms the driver's algorithm knob does not
            leak A3C settings into a PPO run.
        """
        directory = Path(tempfile.mkdtemp(prefix="a3c_script_ppo_"))
        self.addCleanup(shutil.rmtree, directory, True)
        results = directory / "results"
        environment = dict(self._tiny_script_environment())
        environment.update(
            {
                "ALGORITHM": "ppo",
                "METHODS": "state",
                "OUTPUT_DIR": str(results),
                "RUN_AGGREGATION": "0",
            }
        )
        completed = subprocess.run(
            ["bash", str(self.SCRIPT)],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, **environment},
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        configuration = json.loads((results / "state" / "seed_0" / "config.json").read_text())
        self.assertEqual(configuration["algorithm"], "ppo")
        records = [
            json.loads(line)
            for line in (results / "state" / "seed_0" / "metrics.jsonl").read_text().splitlines()
        ]
        self.assertTrue(any("ppo" in record["metrics"] for record in records))
        # Aggregation was skipped, so the CSV summary must be absent.
        self.assertFalse((results / "master_summary.csv").exists())


class MasterAggregationAlgorithmTest(unittest.TestCase):
    """The aggregation scripts must handle A3C runs without changing PPO ones."""

    @staticmethod
    def _script_module() -> Any:
        """Import ``scripts/summarize_master_results.py`` as a module.

        Input: None.
        Output: The imported module object.
        Mathematical meaning: None; exposes the aggregation helpers to tests.
        """
        import importlib.util

        path = Path("scripts") / "summarize_master_results.py"
        specification = importlib.util.spec_from_file_location(
            "summarize_master_results_under_test", path
        )
        module = importlib.util.module_from_spec(specification)
        assert specification.loader is not None
        # ``from __future__ import annotations`` turns the dataclass field types
        # into strings, and dataclasses resolves them through sys.modules, so
        # the module must be registered before it is executed.
        sys.modules[specification.name] = module
        try:
            specification.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(specification.name, None)
            raise
        return module

    def test_sample_geometry_matches_each_algorithm(self) -> None:
        """One logged record must cover the right number of transitions.

        Input: PPO and A3C ``config.json``-shaped mappings.
        Output: ``rollout_steps*num_envs`` for PPO and
            ``n_steps*num_envs*log_interval`` for A3C.
        Mathematical meaning: Makes ``total_extrinsic_return`` comparable across
            algorithms instead of silently scaling A3C means by a PPO rollout.
        """
        module = self._script_module()
        ppo_configuration = {
            "environment": {"num_parallel_envs": 4},
            "ppo": {"rollout_steps": 32},
        }
        a3c_configuration = {
            "algorithm": "a3c",
            "environment": {"num_parallel_envs": 4},
            "ppo": {"rollout_steps": 32},
            "a3c": {"n_steps": 8},
            "training": {"log_interval_updates": 1},
        }
        self.assertEqual(module.samples_per_logged_update(ppo_configuration), 128)
        self.assertEqual(module.samples_per_logged_update(a3c_configuration), 32)
        a3c_configuration["training"]["log_interval_updates"] = 3
        self.assertEqual(module.samples_per_logged_update(a3c_configuration), 96)

    def test_update_metric_columns_follow_the_algorithm(self) -> None:
        """The exported CSV columns must match the optimised objective.

        Input: PPO and A3C configurations.
        Output: ``ppo/*`` columns for PPO, ``a3c/*`` columns for A3C.
        Mathematical meaning: None; keeps a PPO table unchanged and gives an
            A3C table its own loss diagnostics.
        """
        module = self._script_module()
        ppo_columns = module.update_metrics_for({})
        a3c_columns = module.update_metrics_for({"algorithm": "a3c"})
        # The PPO column order must stay exactly as it was before A3C existed,
        # so existing PPO tables diff cleanly against new ones.
        self.assertEqual(
            ppo_columns,
            [
                "update",
                "reward/extrinsic",
                "reward/intrinsic",
                "reward/total",
                "novelty/pixel",
                "novelty/feature",
                "novelty/normalized",
                "ppo/policy_loss",
                "ppo/value_loss",
                "ppo/entropy",
                "ppo/approximate_kl",
                "ppo/clip_fraction",
                "ppo/learning_rate",
                "bigan/discriminator_loss",
                "bigan/encoder_generator_loss",
                "bigan/reconstruction_loss",
                "transition/forward_model_mse",
                "transition/mean_prediction_error",
                "transition/forward_prediction_loss",
                "transition/feature_matching_loss",
            ],
        )
        self.assertIn("a3c/policy_loss", a3c_columns)
        self.assertIn("a3c/gradient_norm", a3c_columns)
        self.assertNotIn("ppo/policy_loss", a3c_columns)
        # A3C has no importance ratio, so it must not claim PPO-only columns.
        for column in ("ppo/approximate_kl", "ppo/clip_fraction"):
            self.assertNotIn(column, a3c_columns)
        # Both algorithms report the same reward, novelty, BiGAN, and
        # transition diagnostics, in the same order.
        for column in module.SHARED_UPDATE_METRICS:
            self.assertIn(column, ppo_columns)
            self.assertIn(column, a3c_columns)

    def test_summary_rows_of_a3c_and_ppo_runs(self) -> None:
        """``summarize_run`` must total A3C and PPO rewards correctly.

        Input: Two synthetic run directories that differ only in ``algorithm``
            and the recorded geometry.
        Output: ``total_extrinsic_return`` scaled by the matching geometry and
            the recorded algorithm reported per run.
        Mathematical meaning: Verifies the cumulative extrinsic return is the
            same physical quantity for both algorithms.
        """
        module = self._script_module()
        results = Path(tempfile.mkdtemp(prefix="agg_"))
        self.addCleanup(shutil.rmtree, results, True)
        for name, configuration in (
            (
                "ppo",
                {
                    "seed": {"seed": 0},
                    "environment": {"num_parallel_envs": 4},
                    "ppo": {"rollout_steps": 8},
                    "training": {"total_environment_steps": 64, "log_interval_updates": 1},
                },
            ),
            (
                "a3c",
                {
                    "algorithm": "a3c",
                    "environment": {"num_parallel_envs": 4},
                    "ppo": {"rollout_steps": 8},
                    "a3c": {"n_steps": 4},
                    "training": {"total_environment_steps": 64, "log_interval_updates": 1},
                },
            ),
        ):
            run_directory = results / name / "seed_0"
            run_directory.mkdir(parents=True)
            (run_directory / "config.json").write_text(json.dumps(configuration))
            (run_directory / "run_info.json").write_text(
                json.dumps({"environment_id": "Tiny-v0", "git_commit": "deadbeef"})
            )
            lines = []
            for update in range(2):
                lines.append(
                    json.dumps(
                        {
                            "step": (update + 1) * 4 * int(
                                configuration.get("a3c", {}).get("n_steps", 8)
                            ),
                            "metrics": {
                                "update": update,
                                "reward/extrinsic": 0.5,
                                "reward/intrinsic": 0.25,
                            },
                        }
                    )
                )
            (run_directory / "metrics.jsonl").write_text("\n".join(lines) + "\n")

        ppo_summary = module.summarize_run("state", 0, results / "ppo" / "seed_0", 10)
        a3c_summary = module.summarize_run("state", 0, results / "a3c" / "seed_0", 10)
        self.assertEqual(ppo_summary.algorithm, "ppo")
        self.assertEqual(a3c_summary.algorithm, "a3c")
        # Two records with mean 0.5 each: PPO covers 2 * (8 * 4) transitions,
        # A3C covers 2 * (4 * 4) transitions.
        self.assertAlmostEqual(ppo_summary.total_extrinsic_return, 0.5 * 2 * 32)
        self.assertAlmostEqual(a3c_summary.total_extrinsic_return, 0.5 * 2 * 16)


class A3CCommandLineTest(unittest.TestCase):
    """The CLI must select the algorithm without touching PPO's commands."""

    def tearDown(self) -> None:
        """Clear the deterministic-algorithms flag set by the runs.

        Input: This test case.
        Output: No value.
        Mathematical meaning: Restores global PyTorch state.
        """
        torch.use_deterministic_algorithms(False)

    def _run_main(self, arguments: List[str]) -> int:
        """Invoke ``main.main`` with an explicit argument vector.

        Input: Command-line argument list (without the program name).
        Output: Process-style exit code.
        Mathematical meaning: Executes one complete experiment launch path.
        """
        from main import main as run_experiment

        with mock.patch.object(sys, "argv", ["main.py", *arguments]):
            return run_experiment()

    def test_a3c_cli_run_completes_and_records_the_algorithm(self) -> None:
        """Test 1/2 at command-line level: ``--algorithm`` selects A3C.

        Input: A tiny CartPole A3C command.
        Output: Exit code ``0`` and ``config.json`` recording ``"a3c"``.
        Mathematical meaning: Confirms the end-to-end launcher and the
            experiment record.
        """
        directory = Path(tempfile.mkdtemp(prefix="a3c_cli_"))
        self.addCleanup(shutil.rmtree, directory, True)
        exit_code = self._run_main(
            [
                "--algorithm",
                "a3c",
                "--env",
                "CartPole-v1",
                "--num-parallel-envs",
                "2",
                "--total-environment-steps",
                "128",
                "--rollout-steps",
                "8",
                "--minibatch-size",
                "4",
                "--a3c-n-steps",
                "8",
                "--bigan-batch-size",
                "8",
                "--disable-tensorboard",
                "--disable-plots",
                "--output-directory",
                str(directory),
            ]
        )
        self.assertEqual(exit_code, 0)
        configuration = json.loads((directory / "config.json").read_text())
        self.assertEqual(configuration["algorithm"], "a3c")
        self.assertEqual(configuration["a3c"]["n_steps"], 8)
        run_info = json.loads((directory / "run_info.json").read_text())
        self.assertEqual(run_info["algorithm"], "a3c")
        records = [
            json.loads(line)
            for line in (directory / "metrics.jsonl").read_text().splitlines()
        ]
        self.assertTrue(any("a3c" in record["metrics"] for record in records))

    def test_default_cli_run_still_selects_ppo(self) -> None:
        """PPO regression at command-line level: the default is unchanged.

        Input: A tiny CartPole command with no ``--algorithm`` flag.
        Output: Exit code ``0`` and ``config.json`` recording ``"ppo"``.
        Mathematical meaning: Confirms existing commands still launch PPO.
        """
        directory = Path(tempfile.mkdtemp(prefix="ppo_cli_"))
        self.addCleanup(shutil.rmtree, directory, True)
        exit_code = self._run_main(
            [
                "--env",
                "CartPole-v1",
                "--num-parallel-envs",
                "2",
                "--total-environment-steps",
                "128",
                "--rollout-steps",
                "8",
                "--minibatch-size",
                "4",
                "--bigan-batch-size",
                "8",
                "--disable-tensorboard",
                "--disable-plots",
                "--output-directory",
                str(directory),
            ]
        )
        self.assertEqual(exit_code, 0)
        configuration = json.loads((directory / "config.json").read_text())
        self.assertEqual(configuration["algorithm"], "ppo")
        records = [
            json.loads(line)
            for line in (directory / "metrics.jsonl").read_text().splitlines()
        ]
        self.assertTrue(any("ppo" in record["metrics"] for record in records))
        self.assertFalse(any("a3c" in record["metrics"] for record in records))


if __name__ == "__main__":
    unittest.main()
