"""Numerically stable online normalization utilities.

The novelty model produces non-stationary reconstruction and feature errors.
This module tracks their first and second moments online using the parallel
variance algorithm, avoiding storage of all historical values and avoiding
backpropagation through statistics.
"""

from __future__ import annotations

from typing import Optional, Union

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class NormalizationState:
    """Serializable snapshot of running normalization statistics."""

    count: float
    mean: Tensor
    variance: Tensor


class RunningMeanVariance:
    """Track element-wise running mean and variance with parallel updates.

    Args:
        shape: Shape of one statistic item. ``()`` tracks scalar values.
        epsilon: Positive floor used only when reading standard deviation.
        device: Storage device.
        dtype: Floating-point storage dtype.
    """

    def __init__(
        self,
        shape: tuple[int, ...] = (),
        epsilon: float = 1.0e-8,
        device: Union[torch.device, str] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize empty running statistics.

        Input: Statistic shape, numerical epsilon, device, and floating dtype.
        Output: An empty tracker with zero count and zero mean/variance.
        Mathematical meaning: Initializes ``n=0``, ``mu=0``, and ``M2=0`` for
            the online population-moment estimator.
        """
        if any(d <= 0 for d in shape):
            raise ValueError("shape dimensions must be positive")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if not dtype.is_floating_point:
            raise TypeError("normalization dtype must be floating point")
        self.shape = shape
        self.epsilon = epsilon
        self.device = torch.device(device)
        self.dtype = dtype
        self.count = 0.0
        self.mean = torch.zeros(shape, device=self.device, dtype=dtype)
        self._m2 = torch.zeros(shape, device=self.device, dtype=dtype)

    @property
    def variance(self) -> Tensor:
        """Return the current population variance estimate.

        Input: This running-statistics object.
        Output: Non-negative tensor with the configured statistic shape.
        Mathematical meaning: Returns ``M2 / n`` for ``n>0`` and zero before
            any samples have been observed.
        """
        if self.count == 0.0:
            return torch.zeros_like(self.mean)
        return (self._m2 / self.count).clamp_min(0.0)

    @property
    def standard_deviation(self) -> Tensor:
        """Return numerically safe running standard deviation.

        Input: This running-statistics object.
        Output: ``sqrt(variance + epsilon)`` with the statistic shape.
        Mathematical meaning: Supplies the denominator in standardized
            novelty/reward values while preventing division by zero.
        """
        return torch.sqrt(self.variance + self.epsilon)

    def update(self, values: Tensor) -> None:
        """Update moments from a batch without retaining its computation graph.

        Input: Tensor shaped ``[batch, *shape]``; scalar statistics accept
            ``[batch]`` or ``[batch, 1]``.
        Output: No value; count, mean, and second central moment are updated.
        Mathematical meaning: Applies the parallel population-variance merge
            equations to combine existing statistics with batch statistics.
        """
        values = values.detach().to(device=self.device, dtype=self.dtype)
        if self.shape == ():
            if values.ndim == 0:
                values = values.reshape(1)
            elif values.ndim != 1:
                values = values.reshape(-1)
        elif values.ndim < 1 or tuple(values.shape[1:]) != self.shape:
            raise ValueError(
                f"values must have shape [batch, {self.shape}], got {tuple(values.shape)}"
            )
        if values.shape[0] == 0:
            return

        batch_count = float(values.shape[0])
        batch_mean = values.mean(dim=0)
        batch_m2 = ((values - batch_mean) ** 2).sum(dim=0)
        if self.count == 0.0:
            self.count = batch_count
            self.mean.copy_(batch_mean)
            self._m2.copy_(batch_m2)
            return

        total_count = self.count + batch_count
        mean_delta = batch_mean - self.mean
        self._m2.add_(batch_m2 + mean_delta.square() * self.count * batch_count / total_count)
        self.mean.add_(mean_delta * (batch_count / total_count))
        self.count = total_count

    def state(self) -> NormalizationState:
        """Return detached copies suitable for checkpoint serialization.

        Input: This running-statistics object.
        Output: ``NormalizationState`` containing count, mean, and variance.
        Mathematical meaning: Captures the sufficient statistics defining the
            current online estimator without exposing mutable internal tensors.
        """
        return NormalizationState(
            count=self.count,
            mean=self.mean.detach().clone(),
            variance=self.variance.detach().clone(),
        )

    def load_state(self, state: NormalizationState) -> None:
        """Restore statistics from a previously saved normalization state.

        Input: A state with matching shape and non-negative finite count.
        Output: No value; replaces the current running statistics.
        Mathematical meaning: Restores the same estimated distribution used to
            normalize a previous experiment or checkpoint.
        """
        if state.count < 0.0 or not torch.isfinite(state.mean).all():
            raise ValueError("normalization state contains invalid count or mean")
        if state.mean.shape != self.mean.shape or state.variance.shape != self.mean.shape:
            raise ValueError("normalization state shape does not match tracker")
        if not torch.isfinite(state.variance).all() or (state.variance < 0.0).any():
            raise ValueError("normalization variance must be finite and non-negative")
        self.count = float(state.count)
        self.mean.copy_(state.mean.to(device=self.device, dtype=self.dtype))
        self._m2.copy_(state.variance.to(device=self.device, dtype=self.dtype) * self.count)


class RunningNormalizer:
    """Normalize a scalar or element-wise stream using running moments.

    Args:
        shape: Shape of one value excluding batch dimension.
        epsilon: Standard-deviation stability constant.
        clip_range: Optional symmetric output clipping bound.
        device: Device for running statistics.
    """

    def __init__(
        self,
        shape: tuple[int, ...] = (),
        epsilon: float = 1.0e-8,
        clip_range: Optional[float] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        """Initialize a running normalizer.

        Input: Value shape, epsilon, optional clipping bound, and device.
        Output: A normalizer containing an empty ``RunningMeanVariance``.
        Mathematical meaning: Defines the online affine transformation used to
            convert non-stationary scores to approximately standardized values.
        """
        if clip_range is not None and clip_range <= 0.0:
            raise ValueError("clip_range must be positive when provided")
        self.statistics = RunningMeanVariance(shape=shape, epsilon=epsilon, device=device)
        self.clip_range = clip_range

    def update(self, values: Tensor) -> None:
        """Incorporate a batch of raw values into running statistics.

        Input: Tensor shaped ``[batch, *shape]``.
        Output: No value; updates statistics using detached values.
        Mathematical meaning: Expands the empirical distribution used by the
            normalization transform without allowing statistics to affect
            gradients of the novelty model.
        """
        self.statistics.update(values)

    def normalize(
        self,
        values: Tensor,
        update: bool = False,
        mean_offset: Optional[Union[Tensor, float]] = None,
    ) -> Tensor:
        """Normalize values with an optional additive mean offset.

        Input: Raw values, optional statistics-update flag, and optional
            ``mean_offset`` such as the running extrinsic reward mean.
        Output: Values transformed by
            ``(values-running_mean+mean_offset)/running_std`` and optionally
            clipped to ``[-clip_range, clip_range]``.
        Mathematical meaning: With ``mean_offset=0`` this is standardization.
            With ``mean_offset=mu(r^e)`` it implements Eq. (5):
            ``r^i=(B-mu(B)+mu(r^e))/sigma(B)``.
        """
        if update:
            self.update(values)
        mean = self.statistics.mean.to(device=values.device, dtype=values.dtype)
        standard_deviation = self.statistics.standard_deviation.to(
            device=values.device, dtype=values.dtype
        )
        if mean_offset is None:
            offset = torch.zeros_like(mean)
        else:
            offset = torch.as_tensor(mean_offset, device=values.device, dtype=values.dtype)
        normalized = (values - mean + offset) / standard_deviation
        if self.clip_range is not None:
            normalized = normalized.clamp(-self.clip_range, self.clip_range)
        return normalized

    def __call__(self, values: Tensor, update: bool = False) -> Tensor:
        """Provide callable syntax for ``normalize``.

        Input: Raw values and optional update flag.
        Output: Normalized, optionally clipped values.
        Mathematical meaning: Applies the running standardization transform.
        """
        return self.normalize(values, update=update)
