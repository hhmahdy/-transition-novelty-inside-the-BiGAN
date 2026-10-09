"""Two-stream Proximal Policy Optimization for Adventurer.

The actor uses the combined advantage

.. math:: A_t = A_t^e + beta A_t^i,

while two independent critics regress against separate extrinsic and
intrinsic return targets. This implements the paper's Section 4.2 design and
does not mix rewards before advantage estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Union

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from config import PPOConfig
from agent.actor import ActorNetwork
from agent.advantage import TwoStreamAdvantageEstimate
from agent.critic import DualCriticNetwork
from agent.rollout_buffer import RolloutBatch, RolloutBuffer


@dataclass(frozen=True)
class PPOUpdateMetrics:
    """Aggregate diagnostics produced by one two-stream PPO update."""

    policy_loss: float
    extrinsic_value_loss: float
    intrinsic_value_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    approximate_kl: float
    clip_fraction: float
    actor_gradient_norm: float
    extrinsic_critic_gradient_norm: float
    intrinsic_critic_gradient_norm: float
    learning_rate: float


class PPOAgent:
    """Optimize one actor and two independent value functions."""

    def __init__(
        self,
        actor: ActorNetwork,
        critic: DualCriticNetwork,
        config: PPOConfig,
        total_updates: int,
        beta: float,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        """Initialize PPO optimizers and learning-rate schedules.

        Input: Actor, dual critic, PPO configuration, planned update count,
            intrinsic advantage coefficient ``beta``, and device.
        Output: Initialized two-stream PPO agent.
        Mathematical meaning: Creates optimization processes for
            ``pi_theta``, ``V^e_phi``, and ``V^i_psi``.
        """
        if total_updates <= 0:
            raise ValueError("total_updates must be positive")
        if beta < 0.0:
            raise ValueError("beta must be non-negative")
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.config = config
        self.device = torch.device(device)
        self.beta = beta
        self.total_updates = total_updates
        self.update_count = 0
        extrinsic_lr = config.critic_learning_rate or config.learning_rate
        intrinsic_lr = config.intrinsic_critic_learning_rate or extrinsic_lr
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.learning_rate, eps=config.adam_epsilon
        )
        self.extrinsic_critic_optimizer = torch.optim.Adam(
            self.critic.extrinsic_critic.parameters(), lr=extrinsic_lr, eps=config.adam_epsilon
        )
        self.intrinsic_critic_optimizer = torch.optim.Adam(
            self.critic.intrinsic_critic.parameters(), lr=intrinsic_lr, eps=config.adam_epsilon
        )
        self.actor_scheduler = self._make_scheduler(self.actor_optimizer)
        self.extrinsic_scheduler = self._make_scheduler(self.extrinsic_critic_optimizer)
        self.intrinsic_scheduler = self._make_scheduler(self.intrinsic_critic_optimizer)

    def _make_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LambdaLR:
        """Create the configured linear learning-rate schedule.

        Input: One PPO optimizer.
        Output: Lambda scheduler decaying to the configured final fraction.
        Mathematical meaning: Controls the gradient-descent step size across
            the finite optimization horizon.
        """
        if not self.config.lr_decay:
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        final_fraction = self.config.lr_final_fraction

        def decay(step: int) -> float:
            """Map scheduler step to a linear learning-rate multiplier.

            Input: Integer scheduler step.
            Output: Multiplier in ``[final_fraction,1]``.
            Mathematical meaning: Implements linear annealing.
            """
            progress = min(max(step, 0) / self.total_updates, 1.0)
            return 1.0 - progress * (1.0 - final_fraction)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, decay)

    @torch.no_grad()
    def select_action(self, observation: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Select an action and evaluate both critics for rollout storage.

        Input: One or more observations and deterministic-action flag.
        Output: ``(action, log_probability, extrinsic_value, intrinsic_value)``.
        Mathematical meaning: Samples from ``pi_theta`` and records
            ``log pi_old``, ``V^e(s_t)``, and ``V^i(s_t)``.
        """
        observation = observation.to(self.device)
        action, log_probability, _ = self.actor.get_action_and_stats(observation, deterministic)
        extrinsic_value, intrinsic_value = self.critic(observation)
        return action, log_probability, extrinsic_value, intrinsic_value

    def _losses(self, batch: RolloutBatch) -> tuple[Tensor, ...]:
        """Compute policy, dual-value, entropy, and clipping terms.

        Input: Two-stream PPO minibatch.
        Output: Total loss, policy loss, separate value losses, entropy, KL,
            and clipping fraction.
        Mathematical meaning: Uses combined ``A^e+beta*A^i`` only for the
            policy objective and separate ``R^e/R^i`` for value regression.
        """
        observations = batch.observations.to(self.device)
        actions = batch.actions.to(self.device)
        old_log_probabilities = batch.old_log_probabilities.to(self.device)
        advantages = batch.advantages.to(self.device)
        extrinsic_returns = batch.extrinsic_returns.to(self.device)
        intrinsic_returns = batch.intrinsic_returns.to(self.device)
        old_extrinsic_values = batch.extrinsic_values.to(self.device)
        old_intrinsic_values = batch.intrinsic_values.to(self.device)
        new_log_probabilities, entropy = self.actor.evaluate_actions(observations, actions)
        new_extrinsic_values, new_intrinsic_values = self.critic(observations)
        log_ratio = new_log_probabilities - old_log_probabilities
        ratio = log_ratio.exp()
        clipped_ratio = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon)
        policy_loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()
        extrinsic_value_loss = self._value_loss(new_extrinsic_values, old_extrinsic_values, extrinsic_returns)
        intrinsic_value_loss = self._value_loss(new_intrinsic_values, old_intrinsic_values, intrinsic_returns)
        entropy_mean = entropy.mean()
        value_loss = (
            self.config.value_loss_coefficient * extrinsic_value_loss
            + self.config.intrinsic_value_loss_coefficient * intrinsic_value_loss
        )
        total_loss = policy_loss + value_loss - self.config.entropy_coefficient * entropy_mean
        approximate_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = ((ratio - clipped_ratio).abs() > 1.0e-8).float().mean()
        return total_loss, policy_loss, extrinsic_value_loss, intrinsic_value_loss, entropy_mean, approximate_kl, clip_fraction

    def _value_loss(self, new_values: Tensor, old_values: Tensor, returns: Tensor) -> Tensor:
        """Compute one critic's regression loss, optionally with value clipping.

        Input: New values, rollout values, and the corresponding stream return
            targets.
        Output: Scalar half mean-squared value loss.
        Mathematical meaning: Computes ``1/2 E[(V-R)^2]`` or its clipped-value
            PPO variant for one reward stream.
        """
        if self.config.value_clip_epsilon is None:
            return 0.5 * (new_values - returns).square().mean()
        clipped_values = old_values + torch.clamp(
            new_values - old_values,
            -self.config.value_clip_epsilon,
            self.config.value_clip_epsilon,
        )
        return 0.5 * torch.maximum(
            (new_values - returns).square(), (clipped_values - returns).square()
        ).mean()

    def _update_minibatch(self, batch: RolloutBatch) -> tuple[float, ...]:
        """Apply one PPO minibatch update to actor and both critics.

        Input: One two-stream rollout minibatch.
        Output: Detached scalar diagnostics and pre-clipping gradient norms.
        Mathematical meaning: Performs gradient descent on the combined policy
            objective and both independent value objectives.
        """
        losses = self._losses(batch)
        total_loss, policy_loss, extrinsic_value_loss, intrinsic_value_loss, entropy, kl, clip_fraction = losses
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.extrinsic_critic_optimizer.zero_grad(set_to_none=True)
        self.intrinsic_critic_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        actor_norm = clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        extrinsic_norm = clip_grad_norm_(self.critic.extrinsic_critic.parameters(), self.config.max_grad_norm)
        intrinsic_norm = clip_grad_norm_(self.critic.intrinsic_critic.parameters(), self.config.max_grad_norm)
        self.actor_optimizer.step()
        self.extrinsic_critic_optimizer.step()
        self.intrinsic_critic_optimizer.step()
        return tuple(float(value.detach().cpu()) for value in (
            total_loss, policy_loss, extrinsic_value_loss, intrinsic_value_loss,
            entropy, kl, clip_fraction, actor_norm, extrinsic_norm, intrinsic_norm,
        ))

    def update(self, rollout_buffer: RolloutBuffer, estimate: TwoStreamAdvantageEstimate, minibatch_size: Optional[int] = None) -> PPOUpdateMetrics:
        """Run all PPO epochs using combined advantages and separate targets.

        Input: Full two-stream rollout buffer and matching advantage estimate.
        Output: Aggregate PPO diagnostics.
        Mathematical meaning: Optimizes policy using ``A^e+beta*A^i`` while
            independently fitting ``V^e`` to ``R^e`` and ``V^i`` to ``R^i``.
        """
        batch_size = minibatch_size or self.config.minibatch_size
        metrics: list[tuple[float, ...]] = []
        for _ in range(self.config.update_epochs):
            minibatches: Iterator[RolloutBatch] = rollout_buffer.iter_minibatches(
                advantages=estimate.combined_advantages,
                extrinsic_advantages=estimate.extrinsic.advantages,
                intrinsic_advantages=estimate.intrinsic.advantages,
                extrinsic_returns=estimate.extrinsic.returns,
                intrinsic_returns=estimate.intrinsic.returns,
                minibatch_size=batch_size,
                beta=self.beta,
            )
            for batch in minibatches:
                metrics.append(self._update_minibatch(batch))
        if not metrics:
            raise RuntimeError("PPO update produced no minibatches")
        averages = torch.tensor(metrics, dtype=torch.float64).mean(dim=0)
        self.actor_scheduler.step()
        self.extrinsic_scheduler.step()
        self.intrinsic_scheduler.step()
        self.update_count += 1
        return PPOUpdateMetrics(
            policy_loss=float(averages[1]),
            extrinsic_value_loss=float(averages[2]),
            intrinsic_value_loss=float(averages[3]),
            value_loss=float(averages[2] * self.config.value_loss_coefficient + averages[3] * self.config.intrinsic_value_loss_coefficient),
            entropy=float(averages[4]),
            total_loss=float(averages[0]),
            approximate_kl=float(averages[5]),
            clip_fraction=float(averages[6]),
            actor_gradient_norm=float(averages[7]),
            extrinsic_critic_gradient_norm=float(averages[8]),
            intrinsic_critic_gradient_norm=float(averages[9]),
            learning_rate=float(self.actor_optimizer.param_groups[0]["lr"]),
        )
