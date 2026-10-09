"""EME diversity-enhanced scaling factor from an ensemble of reward models.

The metric term ``d_t = ||E(s_t)-E(s_{t+1})||_p`` treats every latent movement
equally. EME rescales it by the epistemic disagreement of an ensemble of
reward predictors,

.. math:: \\zeta(r) = \\operatorname{Var}\\big(\\hat r_1(s), \\dots, \\hat r_K(s)\\big),

so that transitions entering reward-relevant but poorly modelled regions are
amplified while well-understood regions are not. Each ensemble member owns an
independent bootstrap replay buffer, which keeps the members diverse: without
independent data the variance collapses and ``zeta`` degenerates to zero.

Members are small MLPs over a feature vector. Following EME (Wang et al.,
2024, Eq. 8), every member predicts the extrinsic reward from the
**state-action pair** ``g(s, a; eta_k)`` -- the action is one-hot encoded
(discrete) or appended raw (continuous) after the state features -- because a
reward model that cannot see the action confuses "unexplored state" with
"untried action in a known state". For pixel observations the feature vector
should be the BiGAN latent code ``E(s)``; the trainer passes an encoder-based
feature extractor for that purpose.

Members are also trained enough to fit their bootstrap samples. The EME
ensemble is trained repeatedly as part of metric learning; one gradient step
per rollout against tens of thousands of newly collected transitions leaves
the members unfitted, and ``zeta`` then tracks input drift instead of reward
structure. A small held-out fraction of each batch is routed to a shared
validation set, and ``update`` reports the out-of-sample MSE so an
under-trained ensemble is visible in the logs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class EnsembleUpdateMetrics:
    """Diagnostics from one ensemble reward-model optimization round."""

    updated: bool
    mean_loss: float
    member_losses: tuple
    buffer_size: int
    update_count: int
    validation_loss: Optional[float] = None


class RewardNet(nn.Module):
    """Predict scalar extrinsic reward from an observation feature vector."""

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        """Build a two-hidden-layer reward regressor ``hat r_k``.

        Input: Feature dimension and hidden width.
        Output: Initialized reward network.
        Mathematical meaning: Defines one member of the ensemble whose spread
            estimates epistemic uncertainty about ``r(s)``.
        """
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        self.input_dim = int(input_dim)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: Tensor) -> Tensor:
        """Predict one reward per row of the feature batch.

        Input: Feature tensor ``[B, input_dim]``.
        Output: Prediction tensor ``[B]``.
        Mathematical meaning: Computes ``hat r_k(s)``.
        """
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError("features must be shaped [batch, input_dim]")
        return self.network(features).squeeze(-1)


class FeatureRewardBuffer:
    """Circular storage of (feature, reward) pairs for one ensemble member."""

    def __init__(
        self,
        capacity: int,
        input_dim: int,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        """Allocate empty bootstrap storage.

        Input: Positive capacity, feature dimension, and storage device.
        Output: Empty buffer.
        Mathematical meaning: Defines the empirical distribution from which one
            ensemble member estimates ``r(s)``.
        """
        if capacity <= 0 or input_dim <= 0:
            raise ValueError("capacity and input_dim must be positive")
        self.capacity = int(capacity)
        self.input_dim = int(input_dim)
        self.device = torch.device(device)
        self.features = torch.zeros((self.capacity, self.input_dim), device=self.device)
        self.rewards = torch.zeros(self.capacity, device=self.device)
        self._size = 0
        self._next_index = 0

    def __len__(self) -> int:
        """Return the number of stored pairs.

        Input: This buffer.
        Output: Integer in ``[0, capacity]``.
        Mathematical meaning: Gives the support size of the member's data set.
        """
        return self._size

    def add(self, features: Tensor, rewards: Tensor) -> None:
        """Append a batch of feature/reward pairs, overwriting oldest rows.

        Input: Features ``[B, input_dim]`` and rewards ``[B]``.
        Output: No value; rows are copied into circular storage.
        Mathematical meaning: Adds Monte Carlo samples of ``(s, r^e)``.
        """
        features = features.detach().to(self.device).float().reshape(-1, self.input_dim)
        rewards = rewards.detach().to(self.device).float().reshape(-1)
        if features.shape[0] != rewards.shape[0]:
            raise ValueError("feature and reward batch sizes must match")
        for index in range(features.shape[0]):
            self.features[self._next_index].copy_(features[index])
            self.rewards[self._next_index] = rewards[index]
            self._next_index = (self._next_index + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> tuple:
        """Uniformly sample a minibatch without replacement.

        Input: Positive batch size no larger than the current size and an
            optional seeded generator.
        Output: ``(features, rewards)`` tensors.
        Mathematical meaning: Draws the empirical expectation used by the
            member's regression loss.
        """
        if batch_size <= 0 or batch_size > self._size:
            raise ValueError("invalid ensemble batch size")
        indices = torch.randperm(self._size, generator=generator, device=self.device)[:batch_size]
        return self.features[: self._size][indices], self.rewards[: self._size][indices]


class EnsembleRewardVariance:
    """Maintain K bootstrapped reward models and expose their variance.

    Args:
        input_dim: Dimension of the feature vector fed to every member. When a
            ``feature_extractor`` is supplied this is the extractor's output
            dimension, otherwise it is the flattened observation dimension.
        ensemble_size: Number of members ``K``.
        hidden_dim: Hidden width of each member.
        learning_rate: Adam learning rate for every member.
        buffer_capacity: Capacity of each member's bootstrap buffer.
        batch_size: Minibatch size used for member updates.
        min_buffer_size: Number of stored pairs required before a member is
            optimized at all.
        bootstrap_probability: Probability that a collected sample is inserted
            into a given member's buffer. Values below one create bootstrap
            diversity; one keeps all data in every member.
        max_grad_norm: Gradient-norm clip applied per member.
        feature_extractor: Optional callable mapping observations to features,
            typically the frozen BiGAN encoder.
        device: Device used for members, buffers, and predictions.
        generator: Optional seeded generator for bootstrap masking and
            minibatch sampling.
    """

    def __init__(
        self,
        input_dim: int,
        ensemble_size: int = 5,
        hidden_dim: int = 256,
        learning_rate: float = 1.0e-3,
        buffer_capacity: int = 10_000,
        batch_size: int = 64,
        min_buffer_size: int = 128,
        bootstrap_probability: float = 1.0,
        max_grad_norm: float = 0.5,
        action_dim: int = 0,
        discrete_actions: bool = True,
        validation_fraction: float = 0.05,
        validation_buffer_capacity: int = 4_096,
        feature_extractor: Optional[Callable[[Tensor], Tensor]] = None,
        device: Union[torch.device, str] = "cpu",
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """Create the members, their optimizers, and their bootstrap buffers.

        Input: Ensemble architecture, optimization, bootstrap, and action
            settings.
        Output: Initialized ensemble able to report ``zeta(r)``.
        Mathematical meaning: Instantiates ``K`` independent estimators of
            ``r(s, a)`` whose disagreement measures epistemic uncertainty.
            ``input_dim`` counts the *state* feature width; the action terms
            (``action_dim`` wide, one-hot when discrete) are appended
            internally, so every member maps
            ``[state features, action] -> reward`` as in EME Eq. (8).
        """
        if ensemble_size < 2:
            raise ValueError("ensemble_size must be at least two for a variance")
        if learning_rate <= 0.0 or max_grad_norm <= 0.0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if batch_size <= 0 or min_buffer_size <= 0 or buffer_capacity <= 0:
            raise ValueError("batch_size, min_buffer_size, and buffer_capacity must be positive")
        if min_buffer_size < batch_size:
            raise ValueError("min_buffer_size must be at least batch_size")
        if not 0.0 < bootstrap_probability <= 1.0:
            raise ValueError("bootstrap_probability must be in (0, 1]")
        if action_dim < 0:
            raise ValueError("action_dim must be non-negative")
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        self.device = torch.device(device)
        self.ensemble_size = int(ensemble_size)
        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)
        self.discrete_actions = bool(discrete_actions)
        self.batch_size = int(batch_size)
        self.min_buffer_size = int(min_buffer_size)
        self.bootstrap_probability = float(bootstrap_probability)
        self.validation_fraction = float(validation_fraction)
        self.max_grad_norm = float(max_grad_norm)
        self.feature_extractor = feature_extractor
        self.generator = generator
        self.update_count = 0
        self.models: List[RewardNet] = [
            RewardNet(self.input_dim + self.action_dim, hidden_dim).to(self.device)
            for _ in range(self.ensemble_size)
        ]
        self.optimizers = [
            torch.optim.Adam(model.parameters(), lr=learning_rate)
            for model in self.models
        ]
        self.buffers = [
            FeatureRewardBuffer(buffer_capacity, self.input_dim + self.action_dim, self.device)
            for _ in range(self.ensemble_size)
        ]
        # Shared held-out set for the out-of-sample loss reported per update.
        self.validation_buffer = FeatureRewardBuffer(
            validation_buffer_capacity, self.input_dim + self.action_dim, self.device
        )

    def _append_actions(self, features: Tensor, actions: Optional[Tensor]) -> Tensor:
        """Concatenate the action terms onto the state feature batch.

        Input: State feature tensor ``[B, state_dim]`` and optional actions.
        Output: Feature tensor ``[B, state_dim + action_dim]``.
        Mathematical meaning: Builds the ``g(s, a)`` regression input of EME
            Eq. (8); discrete actions enter one-hot, continuous actions raw.
        """
        if self.action_dim <= 0:
            return features
        if actions is None:
            raise ValueError(
                "this ensemble was configured with action_dim > 0 and requires actions"
            )
        actions = actions.detach().to(self.device)
        if self.discrete_actions:
            encoded = torch.nn.functional.one_hot(
                actions.reshape(-1).long(), self.action_dim
            ).float()
        else:
            encoded = actions.reshape(features.shape[0], -1).float()
            if encoded.shape[-1] != self.action_dim:
                raise ValueError("continuous action width does not match action_dim")
        return torch.cat([features, encoded], dim=-1)

    @torch.no_grad()
    def features(self, observations: Tensor, actions: Optional[Tensor] = None) -> Tensor:
        """Convert observations and actions to the member input representation.

        Input: Observation batch ``[B, *observation_shape]`` and optional
            actions ``[B]`` (discrete) or ``[B, action_dim]`` (continuous).
        Output: Feature tensor ``[B, input_dim + action_dim]``.
        Mathematical meaning: Applies the frozen embedding ``E(s)`` when one is
            configured, otherwise flattens the observation, then appends the
            action encoding.
        """
        observations = observations.to(self.device).float()
        if self.feature_extractor is not None:
            features = self.feature_extractor(observations)
        else:
            features = observations.flatten(1)
        features = features.detach().reshape(features.shape[0], -1)
        features = self._append_actions(features, actions)
        if features.shape[-1] != self.input_dim + self.action_dim:
            raise ValueError(
                f"feature dimension {features.shape[-1]} does not match "
                f"{self.input_dim + self.action_dim}"
            )
        return features

    def add(
        self,
        observations: Tensor,
        rewards: Tensor,
        actions: Optional[Tensor] = None,
    ) -> None:
        """Insert a batch into every member's bootstrap buffer.

        Input: Observations ``[B, *observation_shape]``, rewards ``[B]``, and
            optional actions. Each member receives an independently masked
            subset; a small random fraction is also routed to the shared
            held-out validation set.
        Output: No value.
        Mathematical meaning: Builds ``K`` bootstrap resamples of the reward
            data set so member predictions stay statistically distinguishable,
            plus an out-of-sample slice for the validation loss.
        """
        features = self.features(observations, actions)
        rewards = rewards.detach().to(self.device).float().reshape(-1)
        if rewards.shape[0] != features.shape[0]:
            raise ValueError("observation and reward batch sizes must match")
        if self.validation_fraction > 0.0:
            keep_mask = (
                torch.rand(features.shape[0], generator=self.generator, device=self.device)
                >= self.validation_fraction
            )
            validation_mask = ~keep_mask
            if bool(validation_mask.any()):
                self.validation_buffer.add(features[validation_mask], rewards[validation_mask])
            if not bool(keep_mask.any()):
                return
            features = features[keep_mask]
            rewards = rewards[keep_mask]
        for buffer in self.buffers:
            if self.bootstrap_probability >= 1.0:
                buffer.add(features, rewards)
                continue
            mask = (
                torch.rand(features.shape[0], generator=self.generator, device=self.device)
                < self.bootstrap_probability
            )
            if bool(mask.any()):
                buffer.add(features[mask], rewards[mask])

    def update(
        self,
        observations: Optional[Tensor] = None,
        rewards: Optional[Tensor] = None,
        actions: Optional[Tensor] = None,
    ) -> EnsembleUpdateMetrics:
        """Optionally store a batch, then fit every member for one step.

        Input: Optional observation/reward/action batch to store before
            optimizing.
        Output: ``EnsembleUpdateMetrics`` with per-member regression losses and
            the mean held-out MSE when the validation set is populated.
        Mathematical meaning: Performs one stochastic gradient step of
            ``min_k E[(hat r_k(s,a)-r)^2]`` for each member on its own bootstrap
            sample, which sharpens the ensemble mean and shrinks ``zeta`` in
            well-explored regions.
        """
        if (observations is None) != (rewards is None):
            raise ValueError("observations and rewards must be provided together")
        if observations is not None and rewards is not None:
            self.add(observations, rewards, actions)
        losses: List[float] = []
        for model, optimizer, buffer in zip(self.models, self.optimizers, self.buffers):
            if len(buffer) < self.min_buffer_size:
                continue
            features, targets = buffer.sample(self.batch_size, self.generator)
            predictions = model(features)
            loss = torch.nn.functional.mse_loss(predictions, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        updated = len(losses) > 0
        if updated:
            self.update_count += 1
        return EnsembleUpdateMetrics(
            updated=updated,
            mean_loss=float(sum(losses) / len(losses)) if updated else 0.0,
            member_losses=tuple(losses),
            buffer_size=int(min(len(buffer) for buffer in self.buffers)),
            update_count=self.update_count,
            validation_loss=self.validation_loss(),
        )

    @torch.no_grad()
    def validation_loss(self) -> Optional[float]:
        """Return the mean held-out MSE across members, or ``None``.

        Input: This ensemble.
        Output: Float when the validation set holds at least one batch worth of
            samples, else ``None``.
        Mathematical meaning: Estimates ``E[(hat r_k(s,a)-r)^2]`` out of
            sample; a value stuck near the target variance floor (near zero for
            sparse rewards) exposes an under-trained ensemble whose ``zeta``
            reflects input drift rather than learned reward structure.
        """
        if self.validation_fraction <= 0.0:
            return None
        if len(self.validation_buffer) < self.batch_size:
            return None
        sample_size = min(len(self.validation_buffer), 2_048)
        features, targets = self.validation_buffer.sample(sample_size, self.generator)
        member_losses = []
        for model in self.models:
            predictions = model(features)
            member_losses.append(
                float(torch.nn.functional.mse_loss(predictions, targets).detach().cpu())
            )
        return float(sum(member_losses) / len(member_losses))

    @torch.no_grad()
    def get_variance(self, observations: Tensor, actions: Optional[Tensor] = None) -> Tensor:
        """Return the ensemble prediction variance for each observation.

        Input: Observation batch ``[B, *observation_shape]`` and the actions
            taken in those states.
        Output: Non-negative tensor ``[B]``.
        Mathematical meaning: Computes ``zeta(r)=Var_k(hat r_k(s,a))`` using the
            population variance over the ``K`` members of the state-action
            reward models.
        """
        features = self.features(observations, actions)
        predictions = torch.stack([model(features) for model in self.models], dim=-1)
        return predictions.var(dim=-1, unbiased=False)

    def state_dict(self) -> dict:
        """Return checkpointable member and optimizer parameters.

        Input: This ensemble.
        Output: Dictionary of member and optimizer state dictionaries.
        Mathematical meaning: Preserves the uncertainty estimator so a resumed
            run reproduces the same scaling factor.
        """
        return {
            "models": [model.state_dict() for model in self.models],
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "update_count": self.update_count,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore member and optimizer parameters from a checkpoint.

        Input: State produced by ``state_dict`` with matching ensemble size.
        Output: No value; members and optimizers are restored in place.
        Mathematical meaning: Resumes the same epistemic-uncertainty estimator.
        """
        models = state.get("models", [])
        optimizers = state.get("optimizers", [])
        if len(models) != self.ensemble_size or len(optimizers) != self.ensemble_size:
            raise ValueError("checkpoint ensemble size does not match")
        for model, model_state in zip(self.models, models):
            model.load_state_dict(model_state)
        for optimizer, optimizer_state in zip(self.optimizers, optimizers):
            optimizer.load_state_dict(optimizer_state)
        self.update_count = int(state.get("update_count", 0))
