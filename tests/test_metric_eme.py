"""Unit tests for the metric-based exploration bonus (Contribution 2).

Every expected value is computed analytically or by an independent reference
computation so the tests validate mathematics rather than restating code.
"""

from __future__ import annotations

import math
import unittest
from typing import Optional

import torch

from config import ExperimentConfig, MetricEMEConfig
from bigan.encoder import BiGANEncoder
from exploration.eme_metric import EMEMetricHead, EMEMetricLearner
from exploration.episodic_count import EpisodicLatentCountScaling
from exploration.ensemble_scaling import EnsembleRewardVariance, RewardNet
from exploration.intrinsic_reward import MetricIntrinsicReward
from exploration.state_discrepancy import LatentStateDiscrepancy


class IdentityEncoder(torch.nn.Module):
    """Encoder stub whose latent code equals the observation vector."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.observation_shape = (latent_dim,)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return observation.float()


class LatentStateDiscrepancyTest(unittest.TestCase):
    """Validate the latent metric term ``||E(s_t)-E(s_{t+1})||_p``."""

    def test_l2_distance_matches_analytic_value_raw_mode(self) -> None:
        discrepancy = LatentStateDiscrepancy(
            IdentityEncoder(3), norm="L2", normalize_latent=False
        )
        obs_t = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]])
        obs_tp1 = torch.tensor([[3.0, 4.0, 0.0], [1.0, 2.0, 2.0]])
        distances = discrepancy(obs_t, obs_tp1)
        # ||(3,4,0)||_2 = 5 exactly; identical states have zero distance.
        self.assertTrue(torch.allclose(distances, torch.tensor([5.0, 0.0])))

    def test_l1_distance_matches_analytic_value_raw_mode(self) -> None:
        discrepancy = LatentStateDiscrepancy(
            IdentityEncoder(3), norm="L1", normalize_latent=False
        )
        obs_t = torch.zeros(1, 3)
        obs_tp1 = torch.tensor([[3.0, -4.0, 1.0]])
        self.assertTrue(torch.allclose(discrepancy(obs_t, obs_tp1), torch.tensor([8.0])))

    def test_normalized_distance_is_scale_invariant_and_bounded(self) -> None:
        """Encoder weight growth must not leak into the metric."""
        discrepancy = LatentStateDiscrepancy(IdentityEncoder(3), norm="L2")
        obs_t = torch.tensor([[1.0, 0.0, 0.0]])
        small = discrepancy(obs_t, torch.tensor([[1.0, 1.0, 0.0]]))
        grown = discrepancy(obs_t, torch.tensor([[100.0, 100.0, 0.0]]))
        # Same direction, different magnitude: identical normalized distance,
        # bounded by 2*sqrt(latent_dim) whatever the code scale is.
        self.assertTrue(torch.allclose(small, grown))
        self.assertLessEqual(float(grown), 2.0 * math.sqrt(3.0) + 1e-6)
        self.assertGreater(float(grown), 0.0)

    def test_normalized_identical_directions_have_zero_distance(self) -> None:
        discrepancy = LatentStateDiscrepancy(IdentityEncoder(3), norm="L2")
        codes = torch.randn(4, 3)
        distances = discrepancy(codes, codes * 7.0)
        self.assertTrue(torch.allclose(distances, torch.zeros(4), atol=1e-6))

    def test_rejects_unknown_norm_and_shape_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            LatentStateDiscrepancy(IdentityEncoder(2), norm="L3")
        discrepancy = LatentStateDiscrepancy(IdentityEncoder(2))
        with self.assertRaises(ValueError):
            discrepancy(torch.zeros(2, 2), torch.zeros(3, 2))

    def test_works_with_the_real_bigan_encoder(self) -> None:
        encoder = BiGANEncoder((6,), latent_dim=4, feature_dim=8, hidden_dim=8)
        discrepancy = LatentStateDiscrepancy(encoder, latent_dim=4)
        distances = discrepancy(torch.randn(5, 6), torch.randn(5, 6))
        self.assertEqual(distances.shape, (5,))
        self.assertTrue(bool((distances >= 0).all()))

    def test_freeze_disables_encoder_gradients(self) -> None:
        encoder = BiGANEncoder((6,), latent_dim=4, feature_dim=8, hidden_dim=8)
        discrepancy = LatentStateDiscrepancy(encoder)
        discrepancy.freeze()
        self.assertFalse(any(p.requires_grad for p in encoder.parameters()))


class EnsembleRewardVarianceTest(unittest.TestCase):
    """Validate the diversity-enhanced scaling factor ``zeta(r)``."""

    def test_variance_matches_manual_population_variance(self) -> None:
        ensemble = EnsembleRewardVariance(input_dim=3, ensemble_size=4, hidden_dim=8)
        observations = torch.randn(7, 3)
        features = ensemble.features(observations)
        with torch.no_grad():
            predictions = torch.stack([m(features) for m in ensemble.models], dim=-1)
        expected = predictions.var(dim=-1, unbiased=False)
        self.assertTrue(torch.allclose(ensemble.get_variance(observations), expected, atol=1e-6))

    def test_identical_members_produce_zero_variance(self) -> None:
        ensemble = EnsembleRewardVariance(input_dim=3, ensemble_size=3, hidden_dim=8)
        reference = ensemble.models[0].state_dict()
        for model in ensemble.models[1:]:
            model.load_state_dict(reference)
        variance = ensemble.get_variance(torch.randn(4, 3))
        self.assertTrue(torch.allclose(variance, torch.zeros(4), atol=1e-7))

    def test_update_fits_a_constant_reward(self) -> None:
        torch.manual_seed(0)
        ensemble = EnsembleRewardVariance(
            input_dim=3,
            ensemble_size=3,
            hidden_dim=32,
            learning_rate=1e-2,
            batch_size=16,
            min_buffer_size=16,
            buffer_capacity=256,
        )
        observations = torch.randn(64, 3)
        rewards = torch.ones(64)
        first = ensemble.update(observations, rewards)
        self.assertTrue(first.updated)
        for _ in range(200):
            metrics = ensemble.update()
        self.assertLess(metrics.mean_loss, first.mean_loss)
        with torch.no_grad():
            predictions = ensemble.models[0](ensemble.features(observations))
        self.assertLess(float((predictions - 1.0).abs().mean()), 0.2)

    def test_bootstrap_masking_keeps_buffers_distinct(self) -> None:
        torch.manual_seed(0)
        ensemble = EnsembleRewardVariance(
            input_dim=2,
            ensemble_size=4,
            bootstrap_probability=0.5,
            min_buffer_size=8,
            batch_size=8,
        )
        ensemble.add(torch.randn(200, 2), torch.randn(200))
        sizes = {len(buffer) for buffer in ensemble.buffers}
        self.assertGreater(len(sizes), 1)

    def test_rejects_degenerate_ensemble_size(self) -> None:
        with self.assertRaises(ValueError):
            EnsembleRewardVariance(input_dim=2, ensemble_size=1)

    def test_reward_net_rejects_wrong_feature_width(self) -> None:
        with self.assertRaises(ValueError):
            RewardNet(4)(torch.zeros(2, 5))


class ConstantVarianceEnsemble:
    """Ensemble stub returning a fixed variance, for clamp verification."""

    def __init__(self, value: float) -> None:
        self.value = value

    def get_variance(
        self, observations: torch.Tensor, actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return torch.full((observations.shape[0],), self.value)


class ScriptedVarianceEnsemble:
    """Ensemble stub returning a caller-supplied variance vector."""

    def __init__(self, values: torch.Tensor) -> None:
        self.values = values

    def get_variance(
        self, observations: torch.Tensor, actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.values


class FixedMetricHead(torch.nn.Module):
    """Metric-head stub returning a constant learned distance of 7."""

    def forward(self, codes_i: torch.Tensor, codes_j: torch.Tensor) -> torch.Tensor:
        return torch.full((codes_i.shape[0],), 7.0)


class MetricIntrinsicRewardTest(unittest.TestCase):
    """Validate ``b_t = d_t * min(max(zeta, 1), M)`` and its clamping."""

    def _bonus(self, zeta: float, distance: float = 5.0, maximum: float = 5.0) -> float:
        reward = MetricIntrinsicReward(
            encoder=IdentityEncoder(3),
            ensemble=ConstantVarianceEnsemble(zeta),
            max_reward_scaling=maximum,
            normalize=False,
            normalize_latent=False,
        )
        obs_t = torch.zeros(1, 3)
        obs_tp1 = torch.tensor([[distance, 0.0, 0.0]])
        return float(reward(obs_t, obs_tp1).combined_score.item())

    def test_variance_below_one_is_clamped_up(self) -> None:
        self.assertAlmostEqual(self._bonus(0.25), 5.0, places=5)

    def test_variance_inside_the_interval_scales_linearly(self) -> None:
        self.assertAlmostEqual(self._bonus(2.0), 10.0, places=5)

    def test_variance_above_the_maximum_is_clamped_down(self) -> None:
        self.assertAlmostEqual(self._bonus(50.0), 25.0, places=5)

    def test_without_an_ensemble_the_bonus_is_the_metric_distance(self) -> None:
        reward = MetricIntrinsicReward(
            encoder=IdentityEncoder(3), ensemble=None, normalize=False,
            normalize_latent=False,
        )
        components = reward(torch.zeros(2, 3), torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 1.0]]))
        self.assertTrue(torch.allclose(components.combined_score, torch.tensor([5.0, 1.0])))
        self.assertTrue(torch.allclose(components.bonus_scale, torch.ones(2)))

    def test_normalization_centres_the_bonus_on_the_extrinsic_scale(self) -> None:
        reward = MetricIntrinsicReward(
            encoder=IdentityEncoder(3), ensemble=None, normalize=True,
            normalize_latent=False,
        )
        obs_t = torch.zeros(64, 3)
        obs_tp1 = torch.randn(64, 3)
        extrinsic = torch.zeros(64)
        components = reward(obs_t, obs_tp1, extrinsic_reward=extrinsic)
        # Eq. (5) with mu(r^e)=0 standardizes the bonus: the batch mean of the
        # normalized score is near zero while raw distances are strictly positive.
        self.assertLess(abs(float(components.normalized_score.mean())), 0.5)
        self.assertTrue(bool((components.combined_score > 0).all()))

    def test_components_expose_generic_novelty_aliases(self) -> None:
        reward = MetricIntrinsicReward(encoder=IdentityEncoder(3), normalize=False)
        components = reward(torch.zeros(3, 3), torch.randn(3, 3))
        self.assertTrue(torch.equal(components.pixel_error, components.latent_distance))
        self.assertTrue(torch.equal(components.feature_error, components.bonus_scale))

    def test_rejects_inverted_clamp_interval(self) -> None:
        with self.assertRaises(ValueError):
            MetricIntrinsicReward(
                encoder=IdentityEncoder(3),
                min_reward_scaling=4.0,
                max_reward_scaling=2.0,
            )

    def test_done_transitions_pay_no_bonus(self) -> None:
        """Death/respawn frame jumps must never register as exploration."""
        reward = MetricIntrinsicReward(
            encoder=IdentityEncoder(3), ensemble=None, normalize=False,
            normalize_latent=False,
        )
        obs_t = torch.zeros(2, 3)
        obs_tp1 = torch.tensor([[3.0, 4.0, 0.0], [3.0, 4.0, 0.0]])
        done = torch.tensor([True, False])
        components = reward(obs_t, obs_tp1, done=done)
        self.assertEqual(float(components.combined_score[0]), 0.0)
        self.assertAlmostEqual(float(components.combined_score[1]), 5.0, places=5)
        self.assertEqual(float(components.bonus_distance[0]), 0.0)

    def test_episodic_counts_habituate_the_bonus(self) -> None:
        reward = MetricIntrinsicReward(
            encoder=IdentityEncoder(3),
            ensemble=None,
            normalize=False,
            episodic_counter=EpisodicLatentCountScaling(num_envs=2, resolution=32),
        )
        obs_t = torch.zeros(2, 3)
        obs_tp1 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        first = reward(obs_t, obs_tp1).combined_score
        second = reward(obs_t, obs_tp1).combined_score
        # First visit pays 1/sqrt(1)=1; the immediate revisit pays 1/sqrt(2).
        unit_distance = 3.0 ** 0.5  # both unit directions at latent_dim=3
        self.assertTrue(torch.allclose(first, torch.full((2,), unit_distance), atol=1e-5))
        self.assertTrue(
            torch.allclose(second, torch.full((2,), unit_distance / math.sqrt(2.0)), atol=1e-5)
        )

    def test_episodic_counts_reset_between_episodes(self) -> None:
        counter = EpisodicLatentCountScaling(num_envs=1, resolution=32)
        codes = torch.ones(1, 4)
        counter.scale(codes)
        counter.scale(codes)
        counter.reset(torch.tensor([True]))
        first_after_reset = counter.scale(codes)
        self.assertAlmostEqual(float(first_after_reset[0]), 1.0, places=6)

    def test_learned_metric_head_replaces_the_geometric_distance(self) -> None:
        reward = MetricIntrinsicReward(
            encoder=IdentityEncoder(3),
            ensemble=None,
            normalize=False,
            normalize_latent=False,
            metric_head=FixedMetricHead(),
        )
        components = reward(torch.zeros(3, 3), torch.randn(3, 3))
        self.assertTrue(torch.allclose(components.bonus_distance, torch.full((3,), 7.0)))
        self.assertTrue(torch.allclose(components.combined_score, torch.full((3,), 7.0)))
        # The geometric diagnostic distance is still reported for the logs.
        self.assertFalse(torch.allclose(components.latent_distance, torch.full((3,), 7.0)))


class NormalisedEMEModeTest(unittest.TestCase):
    """Validate the scale-free mode ``zeta / running_mean(zeta)``."""

    def _reward(
        self,
        ensemble,
        maximum: float = 5.0,
        momentum: float = 0.0,
    ) -> MetricIntrinsicReward:
        return MetricIntrinsicReward(
            encoder=IdentityEncoder(3),
            ensemble=ensemble,
            eme_mode="normalised",
            zeta_momentum=momentum,
            max_reward_scaling=maximum,
            normalize=False,
        )

    def test_scale_is_the_ratio_to_the_running_mean(self) -> None:
        variances = torch.tensor([1.0e-6, 3.0e-6])  # far below the clamp floor
        reward = self._reward(ScriptedVarianceEnsemble(variances))
        obs_t = torch.zeros(2, 3)
        obs_tp1 = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        components = reward(obs_t, obs_tp1)
        # The batch mean is 2e-6, so the two factors are 0.5 and 1.5 even though
        # both raw variances are six orders of magnitude below the clamp floor.
        self.assertTrue(
            torch.allclose(components.bonus_scale, torch.tensor([0.5, 1.5]), atol=1e-5)
        )

    def test_sparse_rewards_do_not_collapse_the_scale_unlike_clamped_mode(self) -> None:
        variances = torch.tensor([1.0e-8, 4.0e-8])
        obs_t = torch.zeros(2, 3)
        obs_tp1 = torch.tensor([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

        clamped = MetricIntrinsicReward(
            encoder=IdentityEncoder(3),
            ensemble=ScriptedVarianceEnsemble(variances),
            eme_mode="clamped",
            normalize=False,
        )
        clamped_scale = clamped(obs_t, obs_tp1).bonus_scale
        # The lower clamp binds for both transitions: V3 cannot separate them.
        self.assertTrue(torch.allclose(clamped_scale, torch.ones(2)))

        normalised_scale = self._reward(ScriptedVarianceEnsemble(variances))(
            obs_t, obs_tp1
        ).bonus_scale
        # The normalised mode still ranks the more surprising transition higher.
        self.assertGreater(float(normalised_scale[1]), float(normalised_scale[0]))
        self.assertAlmostEqual(float(normalised_scale.mean()), 1.0, places=3)

    def test_degenerate_ensemble_falls_back_to_the_metric_bonus(self) -> None:
        reward = self._reward(ConstantVarianceEnsemble(0.0))
        components = reward(torch.zeros(4, 3), torch.randn(4, 3))
        # A zero-variance ensemble reproduces V2 rather than zeroing the
        # exploration signal.
        self.assertTrue(torch.allclose(components.bonus_scale, torch.ones(4)))
        self.assertTrue(
            torch.allclose(components.combined_score, components.latent_distance)
        )

    def test_scale_is_capped_at_the_maximum(self) -> None:
        reward = self._reward(ScriptedVarianceEnsemble(torch.tensor([0.0, 100.0])), maximum=5.0)
        scale = reward(torch.zeros(2, 3), torch.randn(2, 3)).bonus_scale
        self.assertLessEqual(float(scale.max()), 5.0)

    def test_reference_level_is_an_exponential_moving_average(self) -> None:
        reward = self._reward(
            ScriptedVarianceEnsemble(torch.tensor([2.0, 4.0])), momentum=0.9
        )
        obs_t, obs_tp1 = torch.zeros(2, 3), torch.randn(2, 3)
        reward(obs_t, obs_tp1)
        # The EMA is seeded with the first batch mean, (2+4)/2 = 3.
        self.assertAlmostEqual(reward.mean_zeta, 3.0, places=5)
        reward.ensemble = ScriptedVarianceEnsemble(torch.tensor([6.0, 12.0]))
        reward(obs_t, obs_tp1)
        # 0.9*3 + 0.1*9 = 3.6, versus 6.0 for a cumulative mean.
        self.assertAlmostEqual(reward.mean_zeta, 3.6, places=5)

    def test_moving_average_tracks_a_rising_variance(self) -> None:
        """A drifting zeta must not saturate the factor at the cap."""
        ensemble = ScriptedVarianceEnsemble(torch.tensor([1.0, 1.0]))
        reward = self._reward(ensemble, momentum=0.9)
        obs_t, obs_tp1 = torch.zeros(2, 3), torch.randn(2, 3)
        scales = []
        for exponent in range(40):
            # Geometrically increasing disagreement, as in a run that keeps
            # discovering reward structure.
            value = 1.5 ** exponent
            reward.ensemble = ScriptedVarianceEnsemble(torch.tensor([value, value]))
            scales.append(float(reward(obs_t, obs_tp1).bonus_scale.mean()))
        # The EMA keeps up, so late factors stay well below the cap of five.
        self.assertLess(max(scales[-10:]), 5.0)
        self.assertGreater(min(scales[-10:]), 1.0)

    def test_statistics_are_not_updated_when_disabled(self) -> None:
        reward = self._reward(ScriptedVarianceEnsemble(torch.tensor([2.0, 4.0])))
        reward(torch.zeros(2, 3), torch.randn(2, 3), update_statistics=False)
        self.assertEqual(reward.mean_zeta, 0.0)

    def test_rejects_unknown_mode(self) -> None:
        with self.assertRaises(ValueError):
            MetricIntrinsicReward(encoder=IdentityEncoder(3), eme_mode="scaled")

    def test_rejects_invalid_momentum(self) -> None:
        with self.assertRaises(ValueError):
            MetricIntrinsicReward(encoder=IdentityEncoder(3), zeta_momentum=1.0)


class MetricEMEConfigTest(unittest.TestCase):
    """Validate configuration guards for the new experiment variants."""

    def test_defaults_are_disabled_and_valid(self) -> None:
        config = MetricEMEConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.ensemble_size, 5)
        self.assertEqual(config.max_reward_scaling, 5.0)
        self.assertEqual(config.latent_norm, "L2")
        self.assertEqual(config.eme_mode, "clamped")
        # Post-fix defaults: bounded metric, trained ensemble, no legacy paths.
        self.assertTrue(config.normalize_latent_embeddings)
        self.assertEqual(config.ensemble_updates_per_rollout, 64)
        self.assertFalse(config.episodic_count_scaling)
        self.assertEqual(config.metric_learning, "none")

    def test_invalid_settings_raise(self) -> None:
        for kwargs in (
            {"latent_norm": "L3"},
            {"ensemble_size": 1},
            {"max_reward_scaling": 0.5, "min_reward_scaling": 1.0},
            {"ensemble_bootstrap_probability": 0.0},
            {"ensemble_input": "pixels"},
            {"eme_mode": "scaled"},
            {"zeta_epsilon": 0.0},
            {"zeta_momentum": 1.0},
            {"zeta_momentum": -0.1},
            {"metric_learning": "bisimulation"},
            {"episodic_count_resolution": 0},
            {"metric_learning_batch_size": 0},
            {"ensemble_validation_fraction": 1.0},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    MetricEMEConfig(**kwargs)

    def test_metric_bonus_conflicts_with_transition_novelty(self) -> None:
        from config import NoveltyConfig

        with self.assertRaises(ValueError):
            ExperimentConfig(
                novelty=NoveltyConfig(novelty_type="transition"),
                metric_eme=MetricEMEConfig(enabled=True),
            )


class EnsembleActionInputTest(unittest.TestCase):
    """EME Eq. (8): the reward models must see the action g(s, a)."""

    def test_features_include_one_hot_actions(self) -> None:
        ensemble = EnsembleRewardVariance(
            input_dim=3, action_dim=4, discrete_actions=True, hidden_dim=8
        )
        observations = torch.randn(5, 3)
        actions = torch.tensor([0, 1, 2, 3, 0])
        features = ensemble.features(observations, actions)
        self.assertEqual(features.shape, (5, 7))
        # One-hot block is exact.
        self.assertTrue(
            torch.allclose(features[:, 3:], torch.nn.functional.one_hot(actions, 4).float())
        )

    def test_actions_are_required_when_configured(self) -> None:
        ensemble = EnsembleRewardVariance(input_dim=3, action_dim=2, hidden_dim=8)
        with self.assertRaises(ValueError):
            ensemble.features(torch.randn(3, 3))

    def test_variance_runs_over_state_action_pairs(self) -> None:
        ensemble = EnsembleRewardVariance(
            input_dim=3, action_dim=4, discrete_actions=True, hidden_dim=8
        )
        observations = torch.randn(6, 3)
        actions = torch.randint(0, 4, (6,))
        variance = ensemble.get_variance(observations, actions)
        self.assertEqual(variance.shape, (6,))
        self.assertTrue(bool((variance >= 0).all()))

    def test_validation_loss_is_reported_after_updates(self) -> None:
        torch.manual_seed(0)
        ensemble = EnsembleRewardVariance(
            input_dim=3,
            ensemble_size=3,
            hidden_dim=16,
            batch_size=16,
            min_buffer_size=16,
            buffer_capacity=512,
            validation_fraction=0.25,
        )
        for _ in range(30):
            metrics = ensemble.update(torch.randn(64, 3), torch.randn(64))
        self.assertIsNotNone(metrics.validation_loss)
        self.assertGreaterEqual(metrics.validation_loss, 0.0)

    def test_validation_loss_is_none_when_disabled(self) -> None:
        ensemble = EnsembleRewardVariance(
            input_dim=2, hidden_dim=8, validation_fraction=0.0
        )
        ensemble.update(torch.randn(64, 2), torch.randn(64))
        self.assertIsNone(ensemble.validation_loss())


class EpisodicLatentCountScalingTest(unittest.TestCase):
    """Validate the RIDE/NovelD habituation factor 1/sqrt(N_ep)."""

    def test_first_visit_pays_full_bonus_and_revisits_decay(self) -> None:
        counter = EpisodicLatentCountScaling(num_envs=2, resolution=32)
        codes = torch.tensor([[0.5, -0.5], [1.0, 1.0]])
        first = counter.scale(codes)
        second = counter.scale(codes)
        third = counter.scale(codes)
        self.assertTrue(torch.allclose(first, torch.ones(2)))
        self.assertTrue(torch.allclose(second, torch.full((2,), 1.0 / math.sqrt(2.0))))
        self.assertTrue(torch.allclose(third, torch.full((2,), 1.0 / math.sqrt(3.0))))

    def test_distinct_states_are_counted_independently(self) -> None:
        counter = EpisodicLatentCountScaling(num_envs=1, resolution=32)
        counter.scale(torch.tensor([[0.9, 0.9]]))
        fresh = counter.scale(torch.tensor([[-0.9, 0.9]]))
        self.assertAlmostEqual(float(fresh[0]), 1.0, places=6)

    def test_reset_clears_the_slot(self) -> None:
        counter = EpisodicLatentCountScaling(num_envs=2, resolution=32)
        codes = torch.ones(2, 3)
        counter.scale(codes)
        counter.reset(torch.tensor([True, False]))
        env0 = counter.scale(codes)
        env1 = counter.scale(codes)
        self.assertAlmostEqual(float(env0[0]), 1.0, places=6)
        self.assertAlmostEqual(float(env1[0]), 1.0 / math.sqrt(2.0), places=6)

    def test_rejects_bad_configuration(self) -> None:
        with self.assertRaises(ValueError):
            EpisodicLatentCountScaling(num_envs=0)
        with self.assertRaises(ValueError):
            EpisodicLatentCountScaling(num_envs=2, resolution=0)


class LinearActorStub(torch.nn.Module):
    """Tiny discrete policy stand-in returning linear logits."""

    def __init__(self, observation_dim: int, actions: int) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(observation_dim, actions)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.layer(observation.float())


class EMEMetricLearnerTest(unittest.TestCase):
    """Validate EME Eq. (9) metric learning on identity-coded states."""

    def _learner(self) -> EMEMetricLearner:
        torch.manual_seed(0)
        discrepancy = LatentStateDiscrepancy(IdentityEncoder(2), norm="L2")
        actor = LinearActorStub(2, 3)
        return EMEMetricLearner(
            discrepancy=discrepancy,
            actor=actor,
            action_dim=3,
            discrete_actions=True,
            gamma=0.9,
            hidden_dim=32,
            learning_rate=1.0e-2,
            batch_size=128,
        )

    @staticmethod
    def _rollout(steps: int = 8, envs: int = 4):
        observations = torch.randn(steps + 1, envs, 2)
        actions = torch.randint(0, 3, (steps, envs))
        rewards = torch.rand(steps, envs)
        terminated = torch.zeros(steps, envs, dtype=torch.bool)
        terminated[-1, 0] = True
        return observations, actions, rewards, terminated

    def test_head_outputs_are_non_negative(self) -> None:
        head = EMEMetricHead(latent_dim=4, hidden_dim=8)
        distances = head(torch.randn(6, 4), torch.randn(6, 4))
        self.assertEqual(distances.shape, (6,))
        self.assertTrue(bool((distances >= 0).all()))

    def test_regression_loss_decreases_below_the_trivial_floor(self) -> None:
        """With gamma=0 the target is static, so the regression must converge."""
        torch.manual_seed(0)
        discrepancy = LatentStateDiscrepancy(IdentityEncoder(2), norm="L2")
        learner = EMEMetricLearner(
            discrepancy=discrepancy,
            actor=LinearActorStub(2, 3),
            action_dim=3,
            discrete_actions=True,
            gamma=0.0,  # target = |r_i - r_j| exactly, independent of the head
            hidden_dim=32,
            learning_rate=1.0e-2,
            batch_size=128,
        )
        observations, actions, rewards, terminated = self._rollout()
        initial = learner.update(observations, actions, rewards, terminated)
        final = initial
        for _ in range(120):
            final = learner.update(observations, actions, rewards, terminated)
        # At init the head outputs ~0, so the loss starts at E[target^2]; the
        # regression can only reduce the variance left around the (fixed)
        # target, so it must fall well below its initial value.
        self.assertLess(final, initial * 0.6)
        self.assertTrue(math.isfinite(final))

    def test_terminal_bootstrap_is_masked(self) -> None:
        """The update must run cleanly even when every transition ends."""
        learner = self._learner()
        observations, actions, rewards, _ = self._rollout()
        terminated = torch.ones_like(rewards, dtype=torch.bool)
        loss = learner.update(observations, actions, rewards, terminated)
        self.assertTrue(math.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
