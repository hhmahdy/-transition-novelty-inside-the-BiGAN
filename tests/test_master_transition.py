"""Focused tests for the Master's thesis transition novelty.

The Master's experiment compares exactly two intrinsic-reward signals under an
otherwise identical configuration:

* baseline   ``--method state``      Adventurer BiGAN state novelty ``B(s)``
* proposed   ``--method transition`` ``N_T = ||f(E(s_t),a_t) - E(s_{t+1})||_2``

These tests validate the mathematics of the proposed signal (L2 prediction
error, MSE forward-model objective, one-hot action encoding, encoder exclusion
from the forward-model optimizer, finite intrinsic rewards), the cached latent
transition buffer, the command-line selection rules, and that all three
end-to-end paths (Master's baseline, Master's transition, legacy transition)
still complete a tiny run.

Expected values are computed analytically or by an independent reference
computation, matching the convention of the existing test suite.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

import numpy as np
import torch
from torch import Tensor, nn

import main as main_module
from bigan.encoder import BiGANEncoder
from config import DeviceType, ExperimentConfig, NoveltyConfig
from environments.adapter import SingleEnvironmentAdapter
from environments.vector_adapter import VectorEnvironmentAdapter
from exploration.master_transition import (
    LatentTransitionBuffer,
    LatentTransitionNovelty,
    LatentTransitionTrainer,
    MasterTransitionNoveltyComponents,
)
from exploration.transition_novelty import LatentForwardModel
from trainer import AdventurerTrainer
from utils.logger import ExperimentLogger
from utils.normalization import RunningMeanVariance
from utils.run_metadata import write_run_info


LATENT_DIM = 4
OBSERVATION_DIM = 6
ACTION_DIM = 3


class IdentityEncoder(nn.Module):
    """Encoder stub whose latent code equals the observation vector."""

    def __init__(self, latent_dim: int) -> None:
        """Store the latent width and expose the encoder interface.

        Input: Latent dimension equal to the observation width.
        Output: Stub encoder with ``observation_shape`` and ``latent_dim``.
        Mathematical meaning: Realizes ``E(s) = s`` so that latent arithmetic
            can be checked analytically.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.observation_shape = (latent_dim,)

    def forward(self, observation: Tensor) -> Tensor:
        """Return the observation as its own latent code.

        Input: Observation tensor ``[B, latent_dim]``.
        Output: The same values as a latent tensor ``[B, latent_dim]``.
        Mathematical meaning: ``E`` is the identity map.
        """
        return observation.float()


class ConstantForwardModel(nn.Module):
    """Forward-model stub that always predicts one fixed latent vector."""

    def __init__(self, value: Tensor) -> None:
        """Register the constant prediction.

        Input: Latent vector returned for every ``(z, a)`` pair.
        Output: Stub forward model with the ``LatentForwardModel`` interface.
        Mathematical meaning: Realizes ``f(z,a) = c``, so the prediction error
            reduces to the analytic distance ``||c - z'||_2``.
        """
        super().__init__()
        self.register_buffer("value", value.float().reshape(-1))
        self.latent_dim = int(self.value.numel())
        self.action_dim = ACTION_DIM
        self.discrete_actions = True

    def forward(self, latent: Tensor, action: Tensor) -> Tensor:
        """Return the constant prediction for a whole batch.

        Input: Latent batch ``[B,Z]`` and an ignored action batch.
        Output: Tensor ``[B,Z]`` whose rows all equal ``c``.
        Mathematical meaning: ``hat_z_(t+1) = c``.
        """
        return self.value.expand(latent.shape[0], self.latent_dim)


class TransitionNoveltyMathTest(unittest.TestCase):
    """Validate ``N_T = ||f(E(s_t),a_t) - E(s_{t+1})||_2`` analytically."""

    def test_l2_prediction_error_matches_the_analytic_value(self) -> None:
        """A constant predictor makes ``N_T`` an exactly computable distance."""
        encoder = IdentityEncoder(3)
        forward_model = ConstantForwardModel(torch.zeros(3))
        novelty = LatentTransitionNovelty(encoder, forward_model)
        observations = torch.tensor([[1.0, 1.0, 1.0], [-2.0, 0.5, 4.0]])
        next_observations = torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
        actions = torch.tensor([0, 2])
        components = novelty(observations, actions, next_observations, torch.zeros(2))
        # ||(3,4,0) - 0||_2 = 5 and ||0 - 0||_2 = 0 exactly.
        expected = torch.tensor([5.0, 0.0])
        self.assertTrue(
            torch.allclose(components.latent_prediction_error, expected, atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(components.combined_score, expected, atol=1e-6)
        )

    def test_prediction_error_with_a_nonzero_constant_predictor(self) -> None:
        """The error is measured against the prediction, not against zero."""
        encoder = IdentityEncoder(3)
        forward_model = ConstantForwardModel(torch.tensor([1.0, 2.0, 2.0]))
        novelty = LatentTransitionNovelty(encoder, forward_model)
        components = novelty(
            torch.zeros(2, 3),
            torch.tensor([0, 1]),
            torch.tensor([[1.0, 2.0, 2.0], [4.0, 2.0, 2.0]]),
            torch.zeros(2),
        )
        # First target equals the prediction -> 0; second differs by (3,0,0) -> 3.
        self.assertTrue(
            torch.allclose(
                components.latent_prediction_error, torch.tensor([0.0, 3.0]), atol=1e-6
            )
        )

    def test_estimator_agrees_with_an_independent_recomputation(self) -> None:
        """Real encoder + real forward model, checked against manual math."""
        torch.manual_seed(0)
        encoder = BiGANEncoder((OBSERVATION_DIM,), latent_dim=LATENT_DIM, feature_dim=8, hidden_dim=8)
        forward_model = LatentForwardModel(LATENT_DIM, ACTION_DIM, hidden_dim=8, discrete_actions=True)
        novelty = LatentTransitionNovelty(encoder, forward_model)
        observations = torch.randn(5, OBSERVATION_DIM)
        next_observations = torch.randn(5, OBSERVATION_DIM)
        actions = torch.tensor([0, 1, 2, 1, 0])
        with torch.no_grad():
            latents = encoder(observations)
            next_latents = encoder(next_observations)
            predicted = forward_model(latents, actions)
            expected = torch.sqrt(((next_latents - predicted) ** 2).sum(dim=-1))
            expected_l1 = (next_latents - predicted).abs().sum(dim=-1)
        components = novelty(observations, actions, next_observations, torch.zeros(5))
        self.assertTrue(
            torch.allclose(components.latent_prediction_error, expected, atol=1e-6)
        )
        # The Master's score is the L2 norm, explicitly not the legacy L1 sum.
        self.assertFalse(
            torch.allclose(components.latent_prediction_error, expected_l1, atol=1e-5)
        )
        self.assertTrue(bool((components.latent_prediction_error >= 0).all()))

    def test_no_generator_or_discriminator_term_contributes(self) -> None:
        """``feature_error`` is identically zero: the method has one term only."""
        encoder = IdentityEncoder(3)
        novelty = LatentTransitionNovelty(encoder, ConstantForwardModel(torch.zeros(3)))
        components = novelty(
            torch.randn(4, 3), torch.tensor([0, 1, 2, 0]), torch.randn(4, 3), torch.zeros(4)
        )
        self.assertTrue(torch.equal(components.feature_error, torch.zeros(4)))
        self.assertTrue(
            torch.equal(components.pixel_error, components.latent_prediction_error)
        )

    def test_cached_latents_are_returned_for_the_forward_model_buffer(self) -> None:
        """The estimator exposes the exact latents that get cached."""
        torch.manual_seed(1)
        encoder = BiGANEncoder((OBSERVATION_DIM,), latent_dim=LATENT_DIM, feature_dim=8, hidden_dim=8)
        novelty = LatentTransitionNovelty(
            encoder, LatentForwardModel(LATENT_DIM, ACTION_DIM, hidden_dim=8, discrete_actions=True)
        )
        observations = torch.randn(3, OBSERVATION_DIM)
        next_observations = torch.randn(3, OBSERVATION_DIM)
        components = novelty(observations, torch.tensor([0, 1, 2]), next_observations, torch.zeros(3))
        with torch.no_grad():
            self.assertTrue(torch.allclose(components.latent, encoder(observations)))
            self.assertTrue(torch.allclose(components.next_latent, encoder(next_observations)))

    def test_intrinsic_reward_uses_the_baseline_equation_five_normalization(self) -> None:
        """``r^int = (N_T - mu(N_T) + mu(r^e)) / sigma(N_T)`` with running stats."""
        epsilon = 1.0e-8
        encoder = IdentityEncoder(3)
        novelty = LatentTransitionNovelty(encoder, ConstantForwardModel(torch.zeros(3)), epsilon)
        scores = torch.tensor([5.0, 0.0, 3.0, 4.0])
        extrinsic = torch.tensor([0.0, 0.0, 1.0, 0.0])
        # Drive the estimator with targets whose distance to the zero prediction
        # reproduces `scores`: place each target at (score, 0, 0).
        targets = torch.zeros(4, 3)
        targets[:, 0] = scores
        components = novelty(torch.zeros(4, 3), torch.tensor([0, 1, 2, 0]), targets, extrinsic)
        # Independent reference computation of Equation (5).
        score_statistics = RunningMeanVariance((), epsilon)
        score_statistics.update(scores)
        reward_statistics = RunningMeanVariance((), epsilon)
        reward_statistics.update(extrinsic)
        expected = (scores - score_statistics.mean + reward_statistics.mean) / (
            score_statistics.standard_deviation
        )
        self.assertTrue(torch.allclose(components.normalized_score, expected, atol=1e-5))

    def test_intrinsic_reward_is_finite_for_sparse_and_degenerate_inputs(self) -> None:
        """Zero variance and all-zero rewards must not produce NaN/Inf."""
        encoder = IdentityEncoder(3)
        novelty = LatentTransitionNovelty(encoder, ConstantForwardModel(torch.zeros(3)))
        # Degenerate batch: every prediction error is identical, so sigma(N_T)=0
        # and only the epsilon floor keeps the ratio finite.
        constant_targets = torch.tensor([[3.0, 4.0, 0.0]]).repeat(6, 1)
        components = novelty(
            torch.zeros(6, 3), torch.zeros(6, dtype=torch.long), constant_targets, torch.zeros(6)
        )
        self.assertTrue(bool(torch.isfinite(components.latent_prediction_error).all()))
        self.assertTrue(bool(torch.isfinite(components.normalized_score).all()))
        # A second, non-degenerate batch with sparse extrinsic rewards.
        components = novelty(
            torch.randn(6, 3), torch.randint(0, ACTION_DIM, (6,)), torch.randn(6, 3), torch.zeros(6)
        )
        self.assertTrue(bool(torch.isfinite(components.normalized_score).all()))

    def test_normalization_statistics_are_not_updated_by_scoring_gradients(self) -> None:
        """Scoring is a no-grad RL signal: it must not build a graph."""
        encoder = BiGANEncoder((OBSERVATION_DIM,), latent_dim=LATENT_DIM, feature_dim=8, hidden_dim=8)
        novelty = LatentTransitionNovelty(
            encoder, LatentForwardModel(LATENT_DIM, ACTION_DIM, hidden_dim=8, discrete_actions=True)
        )
        observations = torch.randn(4, OBSERVATION_DIM, requires_grad=True)
        components = novelty(observations, torch.tensor([0, 1, 2, 0]), torch.randn(4, OBSERVATION_DIM), torch.zeros(4))
        self.assertFalse(components.latent_prediction_error.requires_grad)
        self.assertFalse(components.normalized_score.requires_grad)


class OneHotActionEncodingTest(unittest.TestCase):
    """Validate the action encoding of the reused ``LatentForwardModel``."""

    def test_discrete_actions_are_one_hot_encoded(self) -> None:
        """Each action index selects exactly one unit coordinate."""
        model = LatentForwardModel(LATENT_DIM, ACTION_DIM, hidden_dim=8, discrete_actions=True)
        actions = torch.tensor([0, 1, 2, 1])
        features = model._encode_action(actions)
        expected = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
        )
        self.assertTrue(torch.equal(features, expected))
        self.assertEqual(tuple(features.shape), (4, ACTION_DIM))

    def test_forward_model_input_width_is_latent_plus_action(self) -> None:
        """The first layer consumes ``z`` concatenated with the one-hot action."""
        model = LatentForwardModel(LATENT_DIM, ACTION_DIM, hidden_dim=8, discrete_actions=True)
        first_layer = model.network[0]
        self.assertEqual(first_layer.in_features, LATENT_DIM + ACTION_DIM)
        self.assertEqual(model(torch.randn(5, LATENT_DIM), torch.randint(0, ACTION_DIM, (5,))).shape, (5, LATENT_DIM))

    def test_out_of_range_discrete_actions_are_rejected(self) -> None:
        """An invalid action index is an error, never a silent mis-encoding."""
        model = LatentForwardModel(LATENT_DIM, ACTION_DIM, hidden_dim=8, discrete_actions=True)
        with self.assertRaises(ValueError):
            model(torch.randn(2, LATENT_DIM), torch.tensor([0, ACTION_DIM]))
        with self.assertRaises(ValueError):
            model(torch.randn(2, LATENT_DIM), torch.tensor([-1, 0]))


class ForwardModelObjectiveTest(unittest.TestCase):
    """Validate the MSE forward-model objective ``L_f``."""

    def _buffer_with(self, latents: Tensor, actions: Tensor, next_latents: Tensor) -> LatentTransitionBuffer:
        """Build a latent buffer holding exactly the supplied transitions.

        Input: Latent/action/next-latent tensors with ``B`` rows.
        Output: Buffer whose size equals ``B`` so sampling returns every row.
        Mathematical meaning: Makes the minibatch expectation equal the full
            empirical mean, so ``L_f`` can be compared with manual arithmetic.
        """
        buffer = LatentTransitionBuffer(
            capacity=max(latents.shape[0], 1), latent_dim=latents.shape[1]
        )
        buffer.add_batch(latents, actions, next_latents)
        return buffer

    def test_loss_is_the_mean_squared_l2_norm_over_the_latent_dimension(self) -> None:
        """``L_f = (1/B) * sum_i ||pred_i - target_i||_2^2``."""
        predicted = torch.tensor([[1.0, 2.0, 2.0], [0.0, 0.0, 0.0]])
        target = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        loss = LatentTransitionTrainer.loss(predicted, target)
        # Sample 1: 1+4+4 = 9; sample 2: 1+1+1 = 3; mean = 6.
        self.assertAlmostEqual(float(loss), 6.0, places=6)
        # Explicitly not the elementwise MSE (which would be 12/6 = 2) and not L1.
        self.assertNotAlmostEqual(float(loss), float((predicted - target).square().mean()), places=6)
        self.assertNotAlmostEqual(float(loss), float((predicted - target).abs().mean()), places=6)

    def test_reported_update_metric_equals_the_manual_mse(self) -> None:
        """The logged ``forward_model_mse`` is the objective actually minimized."""
        torch.manual_seed(0)
        latents = torch.randn(8, LATENT_DIM)
        next_latents = torch.randn(8, LATENT_DIM)
        actions = torch.randint(0, ACTION_DIM, (8,))
        forward_model = LatentForwardModel(LATENT_DIM, ACTION_DIM, hidden_dim=8, discrete_actions=True)
        trainer = LatentTransitionTrainer(forward_model, learning_rate=1.0e-4, batch_size=8)
        with torch.no_grad():
            expected = (
                (forward_model(latents, actions) - next_latents).square().sum(dim=-1).mean()
            )
        metrics = trainer.update(self._buffer_with(latents, actions, next_latents))
        self.assertTrue(metrics.updated)
        self.assertAlmostEqual(metrics.forward_model_mse, float(expected), places=6)
        self.assertEqual(metrics.update_count, 1)
        self.assertTrue(math.isfinite(metrics.mean_prediction_error))

    def test_update_is_skipped_when_the_buffer_is_too_small(self) -> None:
        """No gradient step happens before ``batch_size`` transitions exist."""
        forward_model = LatentForwardModel(LATENT_DIM, ACTION_DIM, hidden_dim=8, discrete_actions=True)
        trainer = LatentTransitionTrainer(forward_model, batch_size=8)
        metrics = trainer.update(self._buffer_with(torch.randn(4, LATENT_DIM), torch.zeros(4, dtype=torch.long), torch.randn(4, LATENT_DIM)))
        self.assertFalse(metrics.updated)
        self.assertEqual(metrics.forward_model_mse, 0.0)
        self.assertEqual(metrics.update_count, 0)

    def test_training_reduces_the_mse_on_a_deterministic_mapping(self) -> None:
        """Fitting ``z_{t+1} = z_t`` must drive the squared error down."""
        torch.manual_seed(0)
        latents = torch.randn(64, LATENT_DIM)
        actions = torch.zeros(64, dtype=torch.long)
        forward_model = LatentForwardModel(LATENT_DIM, ACTION_DIM, hidden_dim=32, discrete_actions=True)
        trainer = LatentTransitionTrainer(forward_model, learning_rate=1.0e-3, batch_size=32)
        buffer = LatentTransitionBuffer(capacity=64, latent_dim=LATENT_DIM)
        buffer.add_batch(latents, actions, latents.clone())
        first = trainer.update(buffer).forward_model_mse
        last = first
        for _ in range(300):
            last = trainer.update(buffer).forward_model_mse
        self.assertLess(last, first / 10.0)
        self.assertTrue(math.isfinite(last))

    def test_loss_rejects_mismatched_shapes(self) -> None:
        """A shape mismatch is an error rather than a broadcast accident."""
        with self.assertRaises(ValueError):
            LatentTransitionTrainer.loss(torch.randn(3, LATENT_DIM), torch.randn(4, LATENT_DIM))


class EncoderExclusionTest(unittest.TestCase):
    """The BiGAN encoder must never be optimized by the forward-model loss."""

    def test_optimizer_holds_only_forward_model_parameters(self) -> None:
        """Parameter identity check on the Adam optimizer."""
        encoder = BiGANEncoder((OBSERVATION_DIM,), latent_dim=LATENT_DIM, feature_dim=8, hidden_dim=8)
        forward_model = LatentForwardModel(LATENT_DIM, ACTION_DIM, hidden_dim=8, discrete_actions=True)
        trainer = LatentTransitionTrainer(forward_model, batch_size=4)
        optimized = {id(parameter) for group in trainer.optimizer.param_groups for parameter in group["params"]}
        forward_ids = {id(parameter) for parameter in forward_model.parameters()}
        encoder_ids = {id(parameter) for parameter in encoder.parameters()}
        self.assertEqual(optimized, forward_ids)
        self.assertEqual(optimized & encoder_ids, set())
        self.assertGreater(len(optimized), 0)

    def test_encoder_parameters_and_gradients_are_untouched_by_an_update(self) -> None:
        """Bitwise parameter equality and ``None`` encoder gradients after updates."""
        torch.manual_seed(0)
        encoder = BiGANEncoder((OBSERVATION_DIM,), latent_dim=LATENT_DIM, feature_dim=8, hidden_dim=8)
        forward_model = LatentForwardModel(LATENT_DIM, ACTION_DIM, hidden_dim=8, discrete_actions=True)
        trainer = LatentTransitionTrainer(forward_model, learning_rate=1.0e-2, batch_size=8)
        with torch.no_grad():
            latents = encoder(torch.randn(8, OBSERVATION_DIM))
            next_latents = encoder(torch.randn(8, OBSERVATION_DIM))
        before = {name: parameter.detach().clone() for name, parameter in encoder.named_parameters()}
        before_forward = {
            name: parameter.detach().clone() for name, parameter in forward_model.named_parameters()
        }
        buffer = LatentTransitionBuffer(capacity=8, latent_dim=LATENT_DIM)
        buffer.add_batch(latents, torch.randint(0, ACTION_DIM, (8,)), next_latents)
        for _ in range(3):
            metrics = trainer.update(buffer)
        self.assertTrue(metrics.updated)
        for name, parameter in encoder.named_parameters():
            self.assertTrue(
                torch.equal(parameter.detach(), before[name]),
                f"encoder parameter {name} was modified by the forward-model update",
            )
            self.assertIsNone(parameter.grad, f"encoder parameter {name} received a gradient")
        # The forward model itself must have moved, otherwise the test is vacuous.
        moved = any(
            not torch.equal(parameter.detach(), before_forward[name])
            for name, parameter in forward_model.named_parameters()
        )
        self.assertTrue(moved)

    def test_latents_are_detached_before_they_enter_the_buffer(self) -> None:
        """Cached targets cannot carry a graph back into the encoder."""
        encoder = BiGANEncoder((OBSERVATION_DIM,), latent_dim=LATENT_DIM, feature_dim=8, hidden_dim=8)
        novelty = LatentTransitionNovelty(
            encoder, LatentForwardModel(LATENT_DIM, ACTION_DIM, hidden_dim=8, discrete_actions=True)
        )
        components = novelty(
            torch.randn(4, OBSERVATION_DIM),
            torch.randint(0, ACTION_DIM, (4,)),
            torch.randn(4, OBSERVATION_DIM),
            torch.zeros(4),
        )
        buffer = LatentTransitionBuffer(capacity=4, latent_dim=LATENT_DIM)
        buffer.add_batch(components.latent, torch.randint(0, ACTION_DIM, (4,)), components.next_latent)
        self.assertFalse(buffer.latents.requires_grad)
        self.assertFalse(buffer.next_latents.requires_grad)


class LatentTransitionBufferTest(unittest.TestCase):
    """Validate shapes, dtypes, circular behaviour, and validation errors."""

    def test_shapes_and_dtypes(self) -> None:
        """Discrete actions are stored as scalars, latents as ``[capacity,Z]``."""
        buffer = LatentTransitionBuffer(capacity=16, latent_dim=LATENT_DIM)
        self.assertEqual(tuple(buffer.latents.shape), (16, LATENT_DIM))
        self.assertEqual(tuple(buffer.next_latents.shape), (16, LATENT_DIM))
        self.assertEqual(tuple(buffer.actions.shape), (16,))
        self.assertEqual(buffer.latents.dtype, torch.float32)
        self.assertEqual(buffer.actions.dtype, torch.long)
        self.assertEqual(len(buffer), 0)

    def test_continuous_action_shape(self) -> None:
        """Continuous actions keep their ``(action_dim,)`` per-sample shape."""
        buffer = LatentTransitionBuffer(capacity=8, latent_dim=LATENT_DIM, action_shape=(2,), action_dtype=torch.float32)
        buffer.add_batch(torch.randn(3, LATENT_DIM), torch.randn(3, 2), torch.randn(3, LATENT_DIM))
        self.assertEqual(tuple(buffer.actions.shape), (8, 2))
        self.assertEqual(len(buffer), 3)
        batch = buffer.sample(2)
        self.assertEqual(tuple(batch.actions.shape), (2, 2))

    def test_add_batch_and_sample_shapes(self) -> None:
        """Sampling returns a ``(z, a, z')`` batch of the requested size."""
        buffer = LatentTransitionBuffer(capacity=32, latent_dim=LATENT_DIM)
        latents = torch.randn(10, LATENT_DIM)
        next_latents = torch.randn(10, LATENT_DIM)
        actions = torch.randint(0, ACTION_DIM, (10,))
        buffer.add_batch(latents, actions, next_latents)
        self.assertEqual(len(buffer), 10)
        batch = buffer.sample(4)
        self.assertEqual(tuple(batch.latents.shape), (4, LATENT_DIM))
        self.assertEqual(tuple(batch.next_latents.shape), (4, LATENT_DIM))
        self.assertEqual(tuple(batch.actions.shape), (4,))
        self.assertEqual(len(batch), 4)
        with self.assertRaises(ValueError):
            buffer.sample(11)
        with self.assertRaises(ValueError):
            buffer.sample(0)

    def test_circular_overwrite_keeps_the_most_recent_transitions(self) -> None:
        """Wrapping replaces the oldest rows and caps the size at capacity."""
        buffer = LatentTransitionBuffer(capacity=4, latent_dim=1)
        first = torch.tensor([[0.0], [1.0], [2.0]])
        buffer.add_batch(first, torch.zeros(3, dtype=torch.long), first.clone())
        self.assertEqual(len(buffer), 3)
        second = torch.tensor([[3.0], [4.0], [5.0]])
        buffer.add_batch(second, torch.zeros(3, dtype=torch.long), second.clone())
        self.assertEqual(len(buffer), 4)
        stored = sorted(float(value) for value in buffer.latents[: len(buffer), 0].tolist())
        # Rows 0 and 1 were overwritten by 4 and 5; rows 2 and 3 hold 2 and 3.
        self.assertEqual(stored, [2.0, 3.0, 4.0, 5.0])

    def test_a_batch_larger_than_capacity_keeps_its_last_rows(self) -> None:
        """An oversized write is truncated deterministically, not scattered."""
        buffer = LatentTransitionBuffer(capacity=3, latent_dim=1)
        values = torch.arange(7, dtype=torch.float32).reshape(-1, 1)
        buffer.add_batch(values, torch.zeros(7, dtype=torch.long), values.clone())
        self.assertEqual(len(buffer), 3)
        stored = sorted(float(value) for value in buffer.latents[:, 0].tolist())
        self.assertEqual(stored, [4.0, 5.0, 6.0])

    def test_invalid_inputs_are_rejected(self) -> None:
        """Wrong widths, mismatched targets, and bad capacities raise."""
        with self.assertRaises(ValueError):
            LatentTransitionBuffer(capacity=0, latent_dim=LATENT_DIM)
        with self.assertRaises(ValueError):
            LatentTransitionBuffer(capacity=4, latent_dim=0)
        buffer = LatentTransitionBuffer(capacity=4, latent_dim=LATENT_DIM)
        with self.assertRaises(ValueError):
            buffer.add_batch(torch.randn(2, LATENT_DIM + 1), torch.zeros(2, dtype=torch.long), torch.randn(2, LATENT_DIM + 1))
        with self.assertRaises(ValueError):
            buffer.add_batch(torch.randn(2, LATENT_DIM), torch.zeros(2, dtype=torch.long), torch.randn(2, LATENT_DIM + 1))
        with self.assertRaises(ValueError):
            buffer.add_batch(torch.randn(2, LATENT_DIM), torch.zeros(3, dtype=torch.long), torch.randn(2, LATENT_DIM))


class _TinyDiscreteEnvironment:
    """Minimal Gymnasium-style discrete environment used by the smoke runs.

    Rewards are sparse but attainable (``action == 1`` pays ``+1``), episodes
    terminate after ``episode_limit`` steps, and observations are small random
    vectors. This exercises the full trainer path -- rollout, novelty, GAE,
    PPO, BiGAN, logging -- in a fraction of a second on CPU.
    """

    def __init__(self, observation_dim: int = OBSERVATION_DIM, actions: int = ACTION_DIM, episode_limit: int = 5) -> None:
        """Declare the observation/action spaces and episode limit.

        Input: Observation width, number of discrete actions, episode length.
        Output: An uninitialized environment (``reset`` starts an episode).
        Mathematical meaning: Defines a tiny finite-horizon MDP with a sparse
            extrinsic reward.
        """
        import gymnasium as gym

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(actions)
        self.episode_limit = episode_limit
        self._rng = np.random.default_rng(0)
        self._steps = 0
        self.unwrapped = self

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Start a new episode.

        Input: Optional episode seed and ignored options.
        Output: ``(observation, info)`` following the Gymnasium five-value API.
        Mathematical meaning: Samples the initial state ``s_0``.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._steps = 0
        return self._rng.normal(size=self.observation_space.shape).astype(np.float32), {}

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance one step of the tiny MDP.

        Input: Discrete action.
        Output: ``(observation, reward, terminated, truncated, info)``.
        Mathematical meaning: ``r = 1`` for action one, else ``0``; the episode
            terminates after ``episode_limit`` transitions.
        """
        self._steps += 1
        observation = self._rng.normal(size=self.observation_space.shape).astype(np.float32)
        reward = 1.0 if int(action) == 1 else 0.0
        terminated = self._steps >= self.episode_limit
        return observation, reward, terminated, False, {}

    def close(self) -> None:
        """Release nothing; the stub owns no resources.

        Input: This environment.
        Output: No value.
        Mathematical meaning: None.
        """
        return None


def _tiny_vector_environment(num_envs: int, seed: int) -> VectorEnvironmentAdapter:
    """Build and seed a vector of tiny environments.

    Input: Number of parallel environments and base seed.
    Output: Seeded ``VectorEnvironmentAdapter``.
    Mathematical meaning: Constructs the product MDP for the smoke runs.
    """
    environments = [
        SingleEnvironmentAdapter(_TinyDiscreteEnvironment()) for _ in range(num_envs)
    ]
    vector_environment = VectorEnvironmentAdapter(environments)
    vector_environment.seed(seed)
    return vector_environment


def _tiny_config(
    output_directory: Path,
    novelty_type: str,
    transition_variant: str = "legacy",
    seed: int = 0,
    num_envs: int = 2,
    rollout_steps: int = 8,
    minibatch_size: int = 4,
    total_steps: int = 64,
    bigan_batch_size: int = 8,
) -> ExperimentConfig:
    """Build a validated tiny configuration for an end-to-end smoke run.

    Input: Output directory, novelty selection, and tiny schedule settings.
    Output: ``ExperimentConfig`` with CPU device, no TensorBoard, and no plots.
    Mathematical meaning: Keeps every PPO/BiGAN relationship valid
        (``total_steps`` divisible by ``num_envs*rollout_steps``, ``rollout_steps``
        divisible by ``minibatch_size``) at a scale that runs in seconds.
    """
    base = ExperimentConfig()
    return replace(
        base,
        seed=replace(base.seed, seed=seed),
        environment=replace(
            base.environment,
            observation_shape=(OBSERVATION_DIM,),
            action_dim=ACTION_DIM,
            discrete_actions=True,
            num_parallel_envs=num_envs,
        ),
        ppo=replace(base.ppo, rollout_steps=rollout_steps, minibatch_size=minibatch_size),
        bigan=replace(base.bigan, batch_size=bigan_batch_size),
        novelty=replace(
            base.novelty,
            novelty_type=novelty_type,
            transition_variant=transition_variant,
            transition_batch_size=bigan_batch_size,
        ),
        training=replace(
            base.training,
            total_environment_steps=total_steps,
            device=DeviceType.CPU,
            output_directory=str(output_directory),
            tensorboard_directory=None,
            plot_directory=None,
            enable_tensorboard=False,
            save_plots=False,
        ),
    )


class EndToEndSmokeRunTest(unittest.TestCase):
    """Tiny CPU runs of both Master's methods and of the legacy transition path."""

    def tearDown(self) -> None:
        """Restore non-deterministic algorithm selection for later tests.

        Input: This test case.
        Output: No value; the global PyTorch determinism flag is cleared.
        Mathematical meaning: ``seed_everything`` enables deterministic
            algorithms for the run under test; other test modules do not expect
            that global state.
        """
        torch.use_deterministic_algorithms(False)

    def _run(
        self,
        method: Optional[str],
        novelty_type: str,
        transition_variant: str = "legacy",
    ) -> Tuple[Path, List[Any]]:
        """Execute one tiny training run and return its directory and metrics.

        Input: Master's method label (or ``None`` for legacy CLI usage), the
            novelty type, and the transition variant.
        Output: ``(output_directory, per_update_metrics)``.
        Mathematical meaning: Runs the complete stochastic optimization loop
            for a few updates and exposes its diagnostics for assertions.
        """
        directory = Path(tempfile.mkdtemp(prefix="master_smoke_"))
        self.addCleanup(shutil.rmtree, directory, True)
        config = _tiny_config(directory, novelty_type, transition_variant)
        environment = _tiny_vector_environment(config.environment.num_parallel_envs, config.seed.seed)
        with ExperimentLogger(directory, config) as logger:
            write_run_info(
                directory,
                config,
                method=method,
                environment_id="TinyDiscrete-v0",
                device_label="cpu",
            )
            trainer = AdventurerTrainer(environment, config, logger)
            results = trainer.train()
            trainer.close()
        return directory, results

    def _assert_common_outputs(
        self, directory: Path, results: List[Any], expected_updates: int
    ) -> List[Dict[str, Any]]:
        """Check the artifacts and numerical health every run must produce.

        Input: Run directory, per-update metrics, and expected update count.
        Output: The update-level JSONL records of the run; assertions fail the
            test on any violation.
        Mathematical meaning: Confirms a complete rollout/GAE/PPO/BiGAN cycle
            with finite rewards and the expected number of updates.
        """
        self.assertEqual(len(results), expected_updates)
        for metrics in results:
            self.assertTrue(math.isfinite(metrics.extrinsic_reward))
            self.assertTrue(math.isfinite(metrics.intrinsic_reward))
            self.assertTrue(math.isfinite(metrics.total_reward))
            self.assertTrue(math.isfinite(metrics.ppo.total_loss))
            self.assertTrue(math.isfinite(metrics.ppo.policy_loss))
        for name in ("config.json", "metrics.jsonl", "run.log", "run_info.json"):
            self.assertTrue((directory / name).is_file(), f"missing {name}")
        configuration = json.loads((directory / "config.json").read_text())
        self.assertIn("novelty", configuration)
        run_info = json.loads((directory / "run_info.json").read_text())
        for key in ("git_commit", "git_dirty", "timestamp", "python_version", "torch_version",
                    "gymnasium_version", "ale_version", "numpy_version", "device", "method",
                    "environment_id", "seed", "num_parallel_envs", "rollout_steps",
                    "total_environment_steps", "key_settings"):
            self.assertIn(key, run_info)
        records = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
        update_records = [record for record in records if "update" in record["metrics"]]
        self.assertEqual(len(update_records), expected_updates)
        return update_records

    def test_master_state_baseline_completes(self) -> None:
        """Experiment A: the unchanged BiGAN state-novelty baseline runs."""
        directory, results = self._run("state", "state")
        update_records = self._assert_common_outputs(directory, results, expected_updates=4)
        for metrics in results:
            self.assertIsNone(metrics.transition)
            self.assertIsNone(metrics.metric_eme)
            # State novelty reports a real reconstruction and feature term.
            self.assertTrue(math.isfinite(metrics.pixel_novelty))
            self.assertTrue(math.isfinite(metrics.feature_novelty))
        self.assertTrue(all("transition" not in record["metrics"] or record["metrics"]["transition"] is None
                            for record in update_records))
        configuration = json.loads((directory / "config.json").read_text())
        self.assertEqual(configuration["novelty"]["novelty_type"], "state")
        self.assertFalse(configuration["metric_eme"]["enabled"])

    def test_master_transition_method_completes(self) -> None:
        """Experiment B: the L2 transition novelty runs and trains ``f_phi``."""
        directory, results = self._run("transition", "transition", "master_l2")
        update_records = self._assert_common_outputs(directory, results, expected_updates=4)
        for metrics in results:
            self.assertIsNotNone(metrics.transition)
            self.assertTrue(metrics.transition.updated)
            self.assertTrue(math.isfinite(metrics.transition.forward_model_mse))
            self.assertTrue(math.isfinite(metrics.transition.mean_prediction_error))
            self.assertGreaterEqual(metrics.transition.mean_prediction_error, 0.0)
            # No feature-matching term exists in the Master's method.
            self.assertEqual(metrics.feature_novelty, 0.0)
            self.assertIsNone(metrics.metric_eme)
        logged = update_records[-1]["metrics"]["transition"]
        self.assertIn("forward_model_mse", logged)
        self.assertIn("mean_prediction_error", logged)
        configuration = json.loads((directory / "config.json").read_text())
        self.assertEqual(configuration["novelty"]["novelty_type"], "transition")
        self.assertEqual(configuration["novelty"]["transition_variant"], "master_l2")
        self.assertFalse(configuration["metric_eme"]["enabled"])

    def test_legacy_transition_path_still_completes(self) -> None:
        """Backward compatibility: ``--novelty transition`` behaves as before."""
        directory, results = self._run(None, "transition", "legacy")
        update_records = self._assert_common_outputs(directory, results, expected_updates=4)
        for metrics in results:
            self.assertIsNotNone(metrics.transition)
            self.assertTrue(metrics.transition.updated)
            # Legacy diagnostics: L1 prediction loss and a feature-matching term.
            self.assertTrue(math.isfinite(metrics.transition.forward_prediction_loss))
            self.assertTrue(math.isfinite(metrics.transition.feature_matching_loss))
        self.assertIn("forward_prediction_loss", update_records[-1]["metrics"]["transition"])
        configuration = json.loads((directory / "config.json").read_text())
        self.assertEqual(configuration["novelty"]["transition_variant"], "legacy")

    def test_both_methods_produce_finite_intrinsic_rewards_in_the_logs(self) -> None:
        """The compared quantity -- ``reward/intrinsic`` -- is finite for both."""
        for variant, novelty_type in (("legacy", "state"), ("master_l2", "transition")):
            directory, _ = self._run(None, novelty_type, variant)
            records = [
                json.loads(line)
                for line in (directory / "metrics.jsonl").read_text().splitlines()
            ]
            intrinsic = [
                record["metrics"]["reward/intrinsic"]
                for record in records
                if "reward/intrinsic" in record["metrics"]
            ]
            self.assertTrue(intrinsic)
            self.assertTrue(all(math.isfinite(value) for value in intrinsic))


class CommandLineSelectionTest(unittest.TestCase):
    """Validate ``--method`` selection and the rejection of ambiguity."""

    def _parse(self, argv: List[str]) -> Any:
        """Parse arguments through the real command-line entry point.

        Input: Argument list without the program name.
        Output: Parsed namespace.
        Mathematical meaning: None; this exercises the experiment selector.
        """
        with mock.patch.object(sys, "argv", ["main.py", *argv]):
            return main_module.parse_args()

    def test_method_state_selects_the_baseline(self) -> None:
        """``--method state`` maps to Adventurer state novelty, no EME."""
        args = self._parse(["--method", "state"])
        config = main_module.build_config(args, _tiny_vector_environment(2, 0))
        self.assertEqual(config.novelty.novelty_type, "state")
        self.assertEqual(config.novelty.transition_variant, "legacy")
        self.assertFalse(config.metric_eme.enabled)

    def test_method_transition_selects_the_master_l2_variant(self) -> None:
        """``--method transition`` maps to the pure L2 forward-model novelty."""
        args = self._parse(["--method", "transition"])
        config = main_module.build_config(args, _tiny_vector_environment(2, 0))
        self.assertEqual(config.novelty.novelty_type, "transition")
        self.assertEqual(config.novelty.transition_variant, "master_l2")
        self.assertFalse(config.metric_eme.enabled)
        self.assertEqual(config.novelty.transition_batch_size, 64)
        self.assertEqual(config.novelty.transition_learning_rate, 1.0e-4)
        self.assertEqual(config.novelty.transition_max_grad_norm, 0.5)
        self.assertEqual(config.novelty.transition_hidden_dim, 256)
        self.assertEqual(config.novelty.transition_update_epochs, 1)

    def test_transition_batch_size_defaults_to_the_documented_value(self) -> None:
        """The forward-model minibatch is 64 unless a smoke run shrinks it."""
        args = self._parse(["--method", "transition"])
        config = main_module.build_config(args, _tiny_vector_environment(2, 0))
        self.assertEqual(config.novelty.transition_batch_size, 64)
        args = self._parse(["--method", "transition", "--transition-batch-size", "8"])
        config = main_module.build_config(args, _tiny_vector_environment(2, 0))
        self.assertEqual(config.novelty.transition_batch_size, 8)
        # The state baseline keeps the same default, so the two compared runs
        # never differ in a field that one of them actually uses.
        args = self._parse(["--method", "state"])
        config = main_module.build_config(args, _tiny_vector_environment(2, 0))
        self.assertEqual(config.novelty.transition_batch_size, 64)

    def test_legacy_cli_selection_is_unchanged(self) -> None:
        """Existing flags keep their historical meaning."""
        for argv, expected_type, expected_variant, expected_metric in (
            ([], "state", "legacy", False),
            (["--novelty", "bigan"], "state", "legacy", False),
            (["--novelty", "state"], "state", "legacy", False),
            (["--novelty", "transition"], "transition", "legacy", False),
            (["--novelty-type", "transition"], "transition", "legacy", False),
            (["--novelty", "latent_discrepancy"], "state", "legacy", True),
        ):
            with self.subTest(argv=argv):
                args = self._parse(argv)
                config = main_module.build_config(args, _tiny_vector_environment(2, 0))
                self.assertEqual(config.novelty.novelty_type, expected_type)
                self.assertEqual(config.novelty.transition_variant, expected_variant)
                self.assertEqual(config.metric_eme.enabled, expected_metric)

    def test_method_and_novelty_combinations_fail_clearly(self) -> None:
        """Ambiguous selections exit with an informative message."""
        cases = [
            (["--method", "state", "--novelty", "transition"], "--novelty transition"),
            (["--method", "transition", "--novelty", "bigan"], "--novelty bigan"),
            (["--method", "transition", "--novelty-type", "state"], "--novelty-type"),
        ]
        for argv, expected_fragment in cases:
            with self.subTest(argv=argv):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as context:
                        self._parse(argv)
                self.assertEqual(context.exception.code, 2)
                self.assertIn(expected_fragment, stderr.getvalue())
                self.assertIn("cannot be combined", stderr.getvalue())

    def test_method_with_legacy_bonus_flags_fails_clearly(self) -> None:
        """EME/ensemble/visit-count bonuses cannot enter the Master's comparison."""
        for argv in (
            ["--method", "transition", "--eme", "True"],
            ["--method", "state", "--metric-learning", "eme"],
            ["--method", "transition", "--episodic-count-scaling", "True"],
        ):
            with self.subTest(argv=argv):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as context:
                        self._parse(argv)
                self.assertEqual(context.exception.code, 2)
                self.assertIn("no EME", stderr.getvalue())

    def test_invalid_transition_variant_is_rejected_by_configuration(self) -> None:
        """The configuration validates the new field."""
        with self.assertRaises(ValueError):
            NoveltyConfig(transition_variant="mse")
        with self.assertRaises(ValueError):
            NoveltyConfig(transition_batch_size=0)
        self.assertEqual(NoveltyConfig().transition_variant, "legacy")
        self.assertEqual(NoveltyConfig().transition_batch_size, 64)


if __name__ == "__main__":
    unittest.main()
