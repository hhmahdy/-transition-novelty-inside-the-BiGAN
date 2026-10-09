"""Master's thesis transition novelty: pure L2 latent forward-prediction error.

This module is the only exploration code used by the Master's experiment
``main.py --method transition``. It intentionally contains no generator
reconstruction term, no discriminator feature-matching term, no EME ensemble
scaling, no adaptive reward scaling, and no visit-count/episodic-count bonus.
It also does not import anything from the legacy EME/metric modules.

The method reuses the BiGAN encoder ``E`` that the baseline already trains and
adds one small latent forward model ``f_phi``:

.. math::

    z_t = E(s_t),\\qquad
    \\hat z_{t+1} = f_\\phi(z_t, a_t),\\qquad
    z_{t+1} = E(s_{t+1}),

.. math::

    N_T(s_t, a_t, s_{t+1}) = \\left\\|\\hat z_{t+1} - z_{t+1}\\right\\|_2.

``N_T`` becomes the intrinsic reward through the project's existing Equation (5)
reward-scale normalization -- exactly the mechanism used by the state-novelty
baseline, with the same epsilon and no additional clipping or scaling:

.. math::

    r_t^{int} = \\frac{N_T - \\mu(N_T) + \\mu(r^e)}{\\sigma(N_T)}.

The forward model is fitted by minimizing the mean squared latent prediction
error over a minibatch of ``B`` cached transitions,

.. math::

    L_f = \\frac{1}{B}\\sum_{i=1}^{B}
          \\left\\|f_\\phi(E(s_t), a_t) - E(s_{t+1})\\right\\|_2^2,

with Adam, one update per PPO update, and gradient-norm clipping. Only
``f_phi``'s parameters belong to this optimizer: the BiGAN encoder never
receives gradients from ``L_f`` and continues to be trained by the BiGAN
adversarial objective alone.

Replay storage holds the cached latent triples ``(z_t, a_t, z_{t+1})`` rather
than raw observations. A raw ``(s_t, a_t, s_{t+1})`` buffer of the
``rollout_steps * num_envs`` transitions collected per update costs roughly
10 GB at the 96x128 Montezuma configuration, while the latent buffer costs a
few tens of megabytes. The consequence, documented explicitly: the regression
target ``z_{t+1}``` is the encoder representation **at collection time** (the
encoder state before that update's BiGAN step). Observations are never
re-encoded later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import Tensor

from bigan.encoder import BiGANEncoder
from exploration.novelty import NormalizedNoveltyScore
from exploration.transition_novelty import LatentForwardModel


@dataclass(frozen=True)
class LatentTransitionBatch:
    """A minibatch of cached latent transitions ``(z_t, a_t, z_{t+1})``."""

    latents: Tensor
    actions: Tensor
    next_latents: Tensor

    def __len__(self) -> int:
        """Return the number of transitions in this batch.

        Input: This batch.
        Output: Number of rows in the latent tensors.
        Mathematical meaning: Gives the minibatch size ``B`` of the
        forward-model objective ``L_f``.
        """
        return int(self.latents.shape[0])


class LatentTransitionBuffer:
    """Fixed-capacity circular storage for cached latent transitions.

    Args:
        capacity: Maximum number of ``(z, a, z')`` triples retained.
        latent_dim: Width of the BiGAN latent code ``z = E(s)``.
        action_shape: Shape of one action excluding the batch dimension; ``()``
            for discrete actions and ``(action_dim,)`` for continuous ones.
        device: Storage device.
        action_dtype: Storage dtype of the actions (``torch.long`` for discrete
            actions, ``torch.float32`` for continuous ones).
    """

    def __init__(
        self,
        capacity: int,
        latent_dim: int,
        action_shape: tuple[int, ...] = (),
        device: Union[torch.device, str] = "cpu",
        action_dtype: torch.dtype = torch.long,
    ) -> None:
        """Allocate latent transition storage.

        Input: Capacity, latent width, action shape, device, and action dtype.
        Output: Empty latent transition buffer.
        Mathematical meaning: Defines the empirical transition support used to
            fit ``f_phi(z_t, a_t)`` to ``z_{t+1}``.
        """
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if any(dimension <= 0 for dimension in action_shape):
            raise ValueError("action_shape dimensions must be positive")
        self.capacity = int(capacity)
        self.latent_dim = int(latent_dim)
        self.action_shape = tuple(action_shape)
        self.device = torch.device(device)
        self.action_dtype = action_dtype
        self.latents = torch.empty(
            (self.capacity, self.latent_dim), dtype=torch.float32, device=self.device
        )
        self.next_latents = torch.empty_like(self.latents)
        self.actions = torch.empty(
            (self.capacity, *self.action_shape), dtype=action_dtype, device=self.device
        )
        self._size = 0
        self._next_index = 0

    def __len__(self) -> int:
        """Return the number of valid cached transitions.

        Input: This buffer.
        Output: Integer in ``[0, capacity]``.
        Mathematical meaning: Reports the sample size available for the
            forward-model minibatch estimate.
        """
        return self._size

    def add_batch(self, latents: Tensor, actions: Tensor, next_latents: Tensor) -> None:
        """Append one batch of latent transitions in collection order.

        Input: Batched ``z_t`` ``[n,Z]``, ``a_t`` ``[n]`` or ``[n,A]``, and
            ``z_{t+1}`` ``[n,Z]`` tensors, all detached from any graph.
        Output: No value; the triples are copied into circular storage.
        Mathematical meaning: Adds samples from the latent transition
            distribution used to regress ``f_phi``.
        """
        latents = latents.detach().to(device=self.device, dtype=torch.float32)
        next_latents = next_latents.detach().to(device=self.device, dtype=torch.float32)
        actions = actions.detach().to(device=self.device, dtype=self.action_dtype)
        if latents.ndim != 2 or latents.shape[1] != self.latent_dim:
            raise ValueError("latents must have shape [batch, latent_dim]")
        if next_latents.shape != latents.shape:
            raise ValueError("latents and next_latents must have identical shapes")
        count = latents.shape[0]
        if count == 0:
            return
        if actions.shape[0] != count or tuple(actions.shape[1:]) != self.action_shape:
            raise ValueError(
                f"actions must have shape [batch, {self.action_shape}], got {tuple(actions.shape)}"
            )
        if count > self.capacity:
            # Keep the most recent ``capacity`` transitions so every write slot
            # stays unique and the circular copy is deterministic.
            latents = latents[-self.capacity :]
            next_latents = next_latents[-self.capacity :]
            actions = actions[-self.capacity :]
            count = self.capacity
        slots = (self._next_index + torch.arange(count, device=self.device)) % self.capacity
        self.latents[slots] = latents
        self.next_latents[slots] = next_latents
        self.actions[slots] = actions
        self._next_index = int((int(slots[-1].item()) + 1) % self.capacity)
        self._size = min(self._size + count, self.capacity)

    def sample(
        self, batch_size: int, generator: Optional[torch.Generator] = None
    ) -> LatentTransitionBatch:
        """Uniformly sample cached latent transitions without replacement.

        Input: Positive batch size no greater than the current buffer size and
            an optional seeded generator.
        Output: ``LatentTransitionBatch`` on the configured device.
        Mathematical meaning: Draws the Monte Carlo minibatch over which
            ``L_f`` is estimated.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > self._size:
            raise ValueError(
                f"cannot sample {batch_size} transitions from buffer of size {self._size}"
            )
        indices = torch.randperm(self._size, generator=generator, device=self.device)[:batch_size]
        return LatentTransitionBatch(
            latents=self.latents[: self._size][indices],
            actions=self.actions[: self._size][indices],
            next_latents=self.next_latents[: self._size][indices],
        )


@dataclass(frozen=True)
class MasterTransitionNoveltyComponents:
    """Per-transition components of the Master's transition novelty.

    ``latent_prediction_error`` is ``N_T``. ``combined_score`` equals ``N_T``
    because the Master's method has exactly one novelty term. The
    ``pixel_error`` and ``feature_error`` properties are compatibility aliases
    so the unchanged trainer/logging code can consume this object exactly like
    :class:`exploration.novelty.NoveltyComponents`: ``novelty/pixel`` reports
    ``N_T`` and ``novelty/feature`` is identically zero because no
    discriminator feature term exists in this method.
    """

    latent_prediction_error: Tensor
    combined_score: Tensor
    normalized_score: Tensor
    latent: Optional[Tensor] = None
    next_latent: Optional[Tensor] = None

    @property
    def pixel_error(self) -> Tensor:
        """Return ``N_T`` under the generic novelty name.

        Input: This components object.
        Output: Latent prediction-error tensor ``[B]``.
        Mathematical meaning: Exposes ``||f(E(s_t),a_t) - E(s_{t+1})||_2``
            where the reconstruction pipeline would report its pixel term.
        """
        return self.latent_prediction_error

    @property
    def feature_error(self) -> Tensor:
        """Return zeros: the Master's method has no feature-matching term.

        Input: This components object.
        Output: Zero tensor ``[B]``.
        Mathematical meaning: Keeps the trainer's logging contract intact while
            making it explicit that no discriminator feature discrepancy
            contributes to ``N_T``.
        """
        return torch.zeros_like(self.latent_prediction_error)


class LatentTransitionNovelty:
    """Score transitions by L2 latent forward-prediction error.

    Args:
        encoder: BiGAN encoder ``E`` producing ``z = E(s)``. Shared with the
            baseline; its parameters are never touched by this module.
        forward_model: Latent forward model ``f_phi(z, a)``.
        normalization_epsilon: Stability constant of the Equation (5) running
            normalization (the baseline's ``NoveltyConfig.normalization_epsilon``).
        normalization_clip: Optional symmetric bound on the normalized score.
            Left ``None`` for the Master's experiment so the mechanism is
            identical to the baseline's unclipped Equation (5) normalization.
        device: Device used for encoding and statistics.
    """

    def __init__(
        self,
        encoder: BiGANEncoder,
        forward_model: LatentForwardModel,
        normalization_epsilon: float = 1.0e-8,
        normalization_clip: Optional[float] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        """Initialize transition scoring and Equation (5) statistics.

        Input: Shared BiGAN encoder, latent forward model, and normalization
            settings.
        Output: Initialized transition novelty estimator.
        Mathematical meaning: Defines ``N_T`` and its reward-scale
            normalization using running ``mu(N_T)``, ``sigma(N_T)``, and
            ``mu(r^e)``.
        """
        if normalization_epsilon <= 0.0:
            raise ValueError("normalization_epsilon must be positive")
        self.device = torch.device(device)
        self.encoder = encoder.to(self.device)
        self.forward_model = forward_model.to(self.device)
        self.normalized_score = NormalizedNoveltyScore(
            epsilon=normalization_epsilon,
            clip_range=normalization_clip,
            device=self.device,
        )

    @torch.no_grad()
    def encode(self, observations: Tensor) -> Tensor:
        """Encode observations with the shared BiGAN encoder.

        Input: Observation batch ``[B, *observation_shape]``.
        Output: Latent codes ``[B, latent_dim]``.
        Mathematical meaning: Computes ``z = E(s)`` using the same encoder
            preprocessing path as the baseline (image inputs are scaled to
            ``[0,1]`` inside the encoder).
        """
        return self.encoder(observations.to(self.device).float())

    @torch.no_grad()
    def prediction_error(
        self, latents: Tensor, actions: Tensor, next_latents: Tensor
    ) -> Tensor:
        """Return one L2 forward-prediction error per transition.

        Input: Cached ``z_t`` ``[B,Z]``, ``a_t``, and target ``z_{t+1}`` ``[B,Z]``.
        Output: Tensor ``[B]`` of Euclidean prediction errors.
        Mathematical meaning: Computes
            ``N_T = ||f_phi(z_t, a_t) - z_{t+1}||_2`` -- the L2 norm over the
            latent dimension, not a squared or L1 error.
        """
        if latents.shape != next_latents.shape:
            raise ValueError("latents and next_latents must have identical shapes")
        predicted = self.forward_model(latents, actions)
        return torch.linalg.vector_norm(next_latents - predicted, ord=2, dim=-1)

    @torch.no_grad()
    def __call__(
        self,
        observations: Tensor,
        actions: Tensor,
        next_observations: Tensor,
        extrinsic_reward: Tensor,
    ) -> MasterTransitionNoveltyComponents:
        """Compute ``N_T`` and its Equation (5) intrinsic reward.

        Input: ``s_t``, ``a_t``, ``s_{t+1}``, and the matching extrinsic reward
            used to centre the normalization on the reward scale.
        Output: Components containing ``N_T``, the cached latents, and the
            normalized intrinsic reward.
        Mathematical meaning: Computes ``z_t = E(s_t)``, ``z_{t+1} = E(s_{t+1})``,
            ``N_T = ||f_phi(z_t,a_t) - z_{t+1}||_2``, updates ``mu(N_T)``,
            ``sigma(N_T)``, ``mu(r^e)``, and returns
            ``r^int = (N_T - mu(N_T) + mu(r^e)) / sigma(N_T)``.
        """
        latents = self.encode(observations)
        next_latents = self.encode(next_observations)
        prediction_error = self.prediction_error(latents, actions, next_latents)
        normalized = self.normalized_score(
            prediction_error,
            update=True,
            extrinsic_reward=extrinsic_reward,
        )
        return MasterTransitionNoveltyComponents(
            latent_prediction_error=prediction_error,
            combined_score=prediction_error,
            normalized_score=normalized,
            latent=latents,
            next_latent=next_latents,
        )


@dataclass(frozen=True)
class MasterTransitionUpdateMetrics:
    """Forward-model training diagnostics of the Master's method."""

    updated: bool
    forward_model_mse: float
    mean_prediction_error: float
    update_count: int


class LatentTransitionTrainer:
    """Fit the latent forward model on cached latent transitions.

    Args:
        forward_model: The model ``f_phi`` to optimize. Only its parameters are
            registered with the optimizer.
        learning_rate: Adam step size (``1e-4`` for the Master's experiment).
        batch_size: Minibatch size ``B`` of ``L_f`` (``64``).
        max_grad_norm: Gradient-norm clipping bound (``0.5``).
        update_epochs: Number of minibatch steps per PPO update (``1``).
        device: Device of the model and buffers.
    """

    def __init__(
        self,
        forward_model: LatentForwardModel,
        learning_rate: float = 1.0e-4,
        batch_size: int = 64,
        max_grad_norm: float = 0.5,
        update_epochs: int = 1,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        """Initialize the forward-model optimizer.

        Input: Latent forward model and optimization schedule settings.
        Output: Trainer whose optimizer contains only ``f_phi`` parameters.
        Mathematical meaning: Defines the stochastic minimization of
            ``L_f = (1/B) * sum_i ||f_phi(z_t,a_t) - z_{t+1}||_2^2``. The BiGAN
            encoder is excluded by construction and therefore keeps being
            trained only by the BiGAN adversarial objective.
        """
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if update_epochs <= 0:
            raise ValueError("update_epochs must be positive")
        self.forward_model = forward_model.to(device)
        self.device = torch.device(device)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.max_grad_norm = float(max_grad_norm)
        self.update_epochs = int(update_epochs)
        self.optimizer = torch.optim.Adam(
            self.forward_model.parameters(), lr=self.learning_rate
        )
        self.update_count = 0

    @staticmethod
    def loss(predicted: Tensor, target: Tensor) -> Tensor:
        """Return the mean squared latent prediction error of one minibatch.

        Input: Predicted and target latent tensors, both ``[B,Z]``.
        Output: Scalar loss tensor.
        Mathematical meaning: Computes
            ``L_f = (1/B) * sum_i ||predicted_i - target_i||_2^2``, i.e. the
            squared L2 norm summed over the latent dimension and averaged over
            the minibatch.
        """
        if predicted.shape != target.shape:
            raise ValueError("predicted and target latents must have identical shapes")
        return (predicted - target).square().sum(dim=-1).mean()

    def update(self, buffer: LatentTransitionBuffer) -> MasterTransitionUpdateMetrics:
        """Perform the configured forward-model steps for one PPO update.

        Input: Latent transition buffer holding at least ``batch_size`` cached
            triples (otherwise no step is taken).
        Output: Diagnostics containing the mean MSE loss and the mean L2
            prediction error ``sqrt``-scale of the sampled minibatches.
        Mathematical meaning: Applies ``update_epochs`` Adam steps on ``L_f``
            with gradient-norm clipping, using cached latent targets only.
        """
        if len(buffer) < self.batch_size:
            return MasterTransitionUpdateMetrics(False, 0.0, 0.0, self.update_count)
        losses = []
        errors = []
        for _ in range(self.update_epochs):
            batch = buffer.sample(self.batch_size)
            predicted = self.forward_model(batch.latents, batch.actions)
            loss = self.loss(predicted, batch.next_latents)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.forward_model.parameters(), self.max_grad_norm
            )
            self.optimizer.step()
            losses.append(loss.detach())
            errors.append(
                torch.linalg.vector_norm(
                    batch.next_latents - predicted.detach(), ord=2, dim=-1
                ).mean()
            )
        self.update_count += 1
        return MasterTransitionUpdateMetrics(
            True,
            float(torch.stack(losses).mean().cpu()),
            float(torch.stack(errors).mean().cpu()),
            self.update_count,
        )
