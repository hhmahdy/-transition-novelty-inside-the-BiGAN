"""Generalized Advantage Estimation and two-stream return computation.

This module implements the temporal-difference and GAE equations used by PPO.
The Adventurer-specific two-stream function estimates extrinsic and intrinsic
advantages independently, then combines only the advantage estimates.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class AdvantageEstimate:
    """Results of one-stream GAE and bootstrapped return computation."""

    advantages: Tensor
    returns: Tensor
    td_errors: Tensor


@dataclass(frozen=True)
class TwoStreamAdvantageEstimate:
    """Independent estimates and the combined policy advantage."""

    extrinsic: AdvantageEstimate
    intrinsic: AdvantageEstimate
    combined_advantages: Tensor


def _validate_trajectory_shapes(
    rewards: Tensor,
    values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    last_value: Tensor,
) -> None:
    """Validate tensor shapes and dtypes for one GAE recurrence.

    Input: Time-major rewards, values, masks, and final bootstrap value.
    Output: No value; raises an exception for incompatible tensors.
    Mathematical meaning: Ensures every reward has a corresponding value and
        termination mask in the TD and GAE equations.
    """
    if rewards.ndim < 1:
        raise ValueError("rewards must have at least one dimension")
    expected = rewards.shape
    for name, tensor in (("values", values), ("terminated", terminated), ("truncated", truncated)):
        if tensor.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tensor.shape}")
    if last_value.shape != rewards.shape[1:]:
        raise ValueError(f"last_value must have shape {rewards.shape[1:]}, got {last_value.shape}")
    if not torch.is_floating_point(rewards) or not torch.is_floating_point(values):
        raise TypeError("rewards and values must be floating-point tensors")
    if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
        raise TypeError("terminated and truncated must be boolean tensors")


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    last_value: Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Compute one reward stream's TD residuals and GAE advantages.

    Input: Time-major rewards/values/masks, final value, discount ``gamma``,
        and trace parameter ``gae_lambda``.
    Output: ``(advantages, td_errors)`` with the reward tensor's shape.
    Mathematical meaning: Computes
        ``delta_t=r_t+gamma*(1-terminal_t)*V_(t+1)-V_t`` and
        ``A_t=delta_t+gamma*lambda*(1-terminal_t)*A_(t+1)``.
        Truncations remain bootstrap-able.
    """
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must satisfy 0 <= gamma < 1")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must satisfy 0 <= gae_lambda <= 1")
    _validate_trajectory_shapes(rewards, values, terminated, truncated, last_value)
    advantages = torch.zeros_like(rewards)
    td_errors = torch.zeros_like(rewards)
    next_value = last_value
    next_advantage = torch.zeros_like(last_value)
    for time_index in range(rewards.shape[0] - 1, -1, -1):
        bootstrap_mask = (~terminated[time_index]).to(values.dtype)
        td_error = rewards[time_index] + gamma * bootstrap_mask * next_value - values[time_index]
        next_advantage = td_error + gamma * gae_lambda * bootstrap_mask * next_advantage
        td_errors[time_index] = td_error
        advantages[time_index] = next_advantage
        next_value = values[time_index]
    return advantages, td_errors


def normalize_advantages(advantages: Tensor, epsilon: float = 1.0e-8) -> Tensor:
    """Normalize one advantage stream to improve PPO conditioning.

    Input: Non-empty advantage tensor and positive numerical epsilon.
    Output: Same-shaped approximately zero-mean, unit-population-variance tensor.
    Mathematical meaning: Applies ``(A-mean(A))/(std(A)+epsilon)``.
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if advantages.numel() == 0:
        raise ValueError("advantages cannot be empty")
    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + epsilon)


def compute_returns_and_advantages(
    rewards: Tensor,
    values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    last_value: Tensor,
    gamma: float,
    gae_lambda: float,
    normalize: bool = True,
    normalization_epsilon: float = 1.0e-8,
) -> AdvantageEstimate:
    """Compute one stream's GAE and critic return targets.

    Input: One reward/value stream, masks, final value, GAE coefficients, and
        normalization setting.
    Output: ``AdvantageEstimate`` with policy advantages, unnormalized returns,
        and TD errors.
    Mathematical meaning: Computes ``R_t=V_t+A_t``; normalization affects only
        policy advantages, never critic targets.
    """
    advantages, td_errors = compute_gae(
        rewards, values, terminated, truncated, last_value, gamma, gae_lambda
    )
    returns = advantages + values
    return AdvantageEstimate(
        advantages=normalize_advantages(advantages, normalization_epsilon) if normalize else advantages,
        returns=returns,
        td_errors=td_errors,
    )


def compute_two_stream_returns_and_advantages(
    extrinsic_rewards: Tensor,
    intrinsic_rewards: Tensor,
    extrinsic_values: Tensor,
    intrinsic_values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    extrinsic_last_value: Tensor,
    intrinsic_last_value: Tensor,
    extrinsic_gamma: float,
    extrinsic_gae_lambda: float,
    intrinsic_gamma: float,
    intrinsic_gae_lambda: float,
    intrinsic_coefficient: float,
    normalize_each_stream: bool = True,
    normalization_epsilon: float = 1.0e-8,
) -> TwoStreamAdvantageEstimate:
    """Compute independent extrinsic/intrinsic GAE and combine advantages only.

    Input: Separate reward streams, separate value streams, common masks,
        separate final values, separate discount/GAE coefficients, and
        ``intrinsic_coefficient=beta``.
    Output: ``TwoStreamAdvantageEstimate`` containing both stream estimates and
        ``combined_advantages = A_e + beta*A_i``.
    Mathematical meaning: Implements the paper's Section 4.2 construction;
        no reward mixing occurs before either GAE recurrence or either critic
        return target.
    """
    if intrinsic_coefficient < 0.0:
        raise ValueError("intrinsic_coefficient must be non-negative")
    extrinsic = compute_returns_and_advantages(
        extrinsic_rewards, extrinsic_values, terminated, truncated,
        extrinsic_last_value, extrinsic_gamma, extrinsic_gae_lambda,
        normalize_each_stream, normalization_epsilon,
    )
    intrinsic = compute_returns_and_advantages(
        intrinsic_rewards, intrinsic_values, terminated, truncated,
        intrinsic_last_value, intrinsic_gamma, intrinsic_gae_lambda,
        normalize_each_stream, normalization_epsilon,
    )
    combined_advantages = extrinsic.advantages + intrinsic_coefficient * intrinsic.advantages
    return TwoStreamAdvantageEstimate(
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        combined_advantages=combined_advantages,
    )
