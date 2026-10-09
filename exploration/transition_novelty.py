"""Transition-level novelty using BiGAN latent dynamics.

This module implements the Section 2 transition variant. It reuses the BiGAN
encoder, generator, and discriminator, and adds only a latent forward model
that predicts the next encoded state from the current encoded state and action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import Tensor, nn

from bigan.discriminator import BiGANDiscriminator
from bigan.encoder import BiGANEncoder
from bigan.generator import BiGANGenerator
from utils.normalization import RunningMeanVariance, RunningNormalizer
from utils.transition_replay import TransitionReplayBuffer


class LatentForwardModel(nn.Module):
    """Predict the next BiGAN latent state from current latent and action."""

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int, discrete_actions: bool) -> None:
        """Initialize a two-layer latent forward dynamics MLP.

        Input: Latent/action dimensions, hidden width, and action-space type.
        Output: Initialized model ``f_phi(z,a)``.
        Mathematical meaning: Defines the latent transition predictor used in
            the first term of transition novelty.
        """
        super().__init__()
        if min(latent_dim, action_dim, hidden_dim) <= 0:
            raise ValueError("forward-model dimensions must be positive")
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.discrete_actions = discrete_actions
        self.network = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.zeros_(module.bias)

    def _encode_action(self, action: Tensor) -> Tensor:
        """Convert discrete or continuous actions to model input features.

        Input: Discrete actions ``[B]`` or continuous actions ``[B,A]``.
        Output: Action feature tensor ``[B,action_dim]``.
        Mathematical meaning: Represents the action component of
            ``f_phi(z_s,a)`` without using raw observations.
        """
        if self.discrete_actions:
            values = action.long().reshape(-1)
            if values.min().item() < 0 or values.max().item() >= self.action_dim:
                raise ValueError("discrete action outside action space")
            return torch.nn.functional.one_hot(values, self.action_dim).float()
        values = action.float()
        if values.ndim != 2 or values.shape[-1] != self.action_dim:
            raise ValueError("continuous action has incorrect shape")
        return values

    def forward(self, latent: Tensor, action: Tensor) -> Tensor:
        """Predict next latent state.

        Input: Current latent tensor ``[B,Z]`` and matching actions.
        Output: Predicted next latent tensor ``[B,Z]``.
        Mathematical meaning: Computes ``hat_z_(t+1)=f_phi(z_t,a_t)``.
        """
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError("latent has incorrect shape")
        action_features = self._encode_action(action).to(latent.device)
        if action_features.shape[0] != latent.shape[0]:
            raise ValueError("latent/action batch sizes must match")
        return self.network(torch.cat((latent, action_features), dim=-1))


@dataclass(frozen=True)
class TransitionNoveltyComponents:
    """Transition prediction, feature, combined, and normalized scores."""

    latent_prediction_error: Tensor
    feature_matching_error: Tensor
    combined_score: Tensor
    normalized_score: Tensor

    @property
    def pixel_error(self) -> Tensor:
        """Return latent prediction error for generic novelty logging.

        Input: This transition novelty result.
        Output: Latent prediction-error tensor.
        Mathematical meaning: Provides a compatibility alias for reward logs.
        """
        return self.latent_prediction_error

    @property
    def feature_error(self) -> Tensor:
        """Return transition feature-matching error.

        Input: This transition novelty result.
        Output: Feature-matching-error tensor.
        Mathematical meaning: Exposes the second term of ``T(s,a,s')``.
        """
        return self.feature_matching_error


class TransitionNoveltyEstimator:
    """Compute normalized transition novelty using shared BiGAN features."""

    def __init__(
        self,
        encoder: BiGANEncoder,
        generator: BiGANGenerator,
        discriminator: BiGANDiscriminator,
        forward_model: LatentForwardModel,
        alpha: float = 0.9,
        normalization_epsilon: float = 1.0e-8,
        normalization_clip: Optional[float] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        """Initialize transition scoring and Eq. (5) statistics.

        Input: Shared BiGAN modules, forward model, transition alpha, and
            normalization/device settings.
        Output: Initialized transition novelty estimator.
        Mathematical meaning: Defines ``T(s,a,s')`` and its reward-scale
            normalization using running ``mu(B)``/``sigma(B)``/``mu(r^e)``.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("transition alpha must be in [0,1]")
        self.encoder = encoder.to(device)
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.forward_model = forward_model.to(device)
        self.alpha = alpha
        self.device = torch.device(device)
        self.normalizer = RunningNormalizer((), normalization_epsilon, normalization_clip, self.device)
        self.extrinsic_statistics = RunningMeanVariance((), normalization_epsilon, self.device)

    @torch.no_grad()
    def score_components(self, observations: Tensor, actions: Tensor, next_observations: Tensor) -> TransitionNoveltyComponents:
        """Compute raw transition errors and normalized transition novelty.

        Input: ``s_t``, ``a_t``, and ``s_(t+1)`` batches.
        Output: Transition novelty components, including normalized score.
        Mathematical meaning: Computes the PDF's transition equation using
            latent prediction and discriminator feature matching.
        """
        observations = observations.to(self.device).float()
        next_observations = next_observations.to(self.device).float()
        actions = actions.to(self.device)
        latent = self.encoder(observations)
        next_latent = self.encoder(next_observations)
        predicted_latent = self.forward_model(latent, actions)
        latent_error = (next_latent - predicted_latent).abs().sum(dim=1)
        _, actual_features = self.discriminator.forward_with_features(next_observations, next_latent)
        predicted_observations = self.generator(predicted_latent)
        _, predicted_features = self.discriminator.forward_with_features(
            predicted_observations, predicted_latent
        )
        feature_error = (actual_features - predicted_features).abs().sum(dim=1)
        combined = self.alpha * latent_error + (1.0 - self.alpha) * feature_error
        if self.extrinsic_statistics.count == 0.0:
            extrinsic_mean = torch.zeros((), device=self.device)
        else:
            extrinsic_mean = self.extrinsic_statistics.mean
        normalized = self.normalizer.normalize(
            combined,
            update=True,
            mean_offset=extrinsic_mean,
        )
        return TransitionNoveltyComponents(latent_error, feature_error, combined, normalized)

    def __call__(
        self,
        observations: Tensor,
        actions: Tensor,
        next_observations: Tensor,
        extrinsic_reward: Tensor,
    ) -> TransitionNoveltyComponents:
        """Compute transition novelty using the current extrinsic reward scale.

        Input: Transition batch and matching extrinsic rewards.
        Output: Normalized transition novelty components.
        Mathematical meaning: Updates ``mu(r^e)`` and evaluates Eq. (5) for
            the transition score.
        """
        self.extrinsic_statistics.update(extrinsic_reward.reshape(-1))
        return self.score_components(observations, actions, next_observations)


@dataclass(frozen=True)
class TransitionUpdateMetrics:
    """Forward-model training diagnostics."""

    updated: bool
    forward_prediction_loss: float
    feature_matching_loss: float
    total_loss: float
    update_count: int


class TransitionNoveltyTrainer:
    """Fit the latent forward model from transition replay."""

    def __init__(self, estimator: TransitionNoveltyEstimator, learning_rate: float, update_epochs: int, max_grad_norm: float) -> None:
        """Initialize the forward-model optimizer.

        Input: Transition estimator and optimizer schedule settings.
        Output: Trainer that updates only ``f_phi``; BiGAN parameters are reused
            and not modified by this transition-model update.
        Mathematical meaning: Fits ``f_phi(E(s),a)`` toward ``E(s')``.
        """
        self.estimator = estimator
        self.optimizer = torch.optim.Adam(estimator.forward_model.parameters(), lr=learning_rate)
        self.update_epochs = update_epochs
        self.max_grad_norm = max_grad_norm
        self.update_count = 0

    def update(self, replay_buffer: TransitionReplayBuffer, batch_size: int) -> TransitionUpdateMetrics:
        """Train the forward model on replayed latent transitions.

        Input: Transition replay buffer and positive minibatch size.
        Output: Prediction/feature/total diagnostics.
        Mathematical meaning: Minimizes L1 latent prediction error; feature
            matching is reported as a detached transition novelty diagnostic.
        """
        if len(replay_buffer) < batch_size:
            return TransitionUpdateMetrics(False, 0.0, 0.0, 0.0, self.update_count)
        prediction_losses = []
        feature_losses = []
        total_losses = []
        for _ in range(self.update_epochs):
            batch = replay_buffer.sample(batch_size)
            with torch.no_grad():
                states = batch.observations.to(self.estimator.device).float()
                next_states = batch.next_observations.to(self.estimator.device).float()
                states = states / 255.0 if states.numel() and states.amax().item() > 1.0 and states.ndim >= 3 else states
                next_states = next_states / 255.0 if next_states.numel() and next_states.amax().item() > 1.0 and next_states.ndim >= 3 else next_states
                target = self.estimator.encoder(next_states)
                source = self.estimator.encoder(states)
            predicted = self.estimator.forward_model(source, batch.actions.to(self.estimator.device))
            prediction_loss = (predicted - target).abs().mean()
            self.optimizer.zero_grad(set_to_none=True)
            prediction_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.estimator.forward_model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            with torch.no_grad():
                components = self.estimator.score_components(states, batch.actions, next_states)
                feature_loss = components.feature_matching_error.mean()
            prediction_losses.append(prediction_loss.detach())
            feature_losses.append(feature_loss.detach())
            total_losses.append(prediction_loss.detach())
        self.update_count += 1
        return TransitionUpdateMetrics(
            True,
            float(torch.stack(prediction_losses).mean().cpu()),
            float(torch.stack(feature_losses).mean().cpu()),
            float(torch.stack(total_losses).mean().cpu()),
            self.update_count,
        )
