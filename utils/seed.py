"""Reproducibility utilities for Adventurer experiments.

Reinforcement-learning reproducibility requires more than setting one global
seed. This module coordinates Python, NumPy, PyTorch, CUDA, worker, and
Gymnasium-compatible environment seeds. Deterministic execution can reduce
throughput, so it is controlled explicitly by ``SeedConfig``.
"""

from __future__ import annotations

import os
import random
from typing import Any, Union

import numpy as np
import torch

from config import SeedConfig


def seed_everything(config: SeedConfig) -> None:
    """Seed all process-level random sources and configure backend behavior.

    Input: A validated ``SeedConfig`` containing the base seed and deterministic
        backend settings.
    Output: No value; global random generators and PyTorch backend flags are
        configured in place.
    Mathematical meaning: Makes stochastic policy sampling, environment
        initialization, parameter initialization, minibatch shuffling, and
        optimization noise reproducible for a fixed software/hardware stack.
    """
    os.environ["PYTHONHASHSEED"] = str(config.seed)
    # Required by CUDA/cuBLAS when deterministic PyTorch algorithms are
    # enabled. Set before the first CUDA operation in the process.
    if config.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.deterministic = config.deterministic
    torch.backends.cudnn.benchmark = config.benchmark
    try:
        torch.use_deterministic_algorithms(config.deterministic)
    except RuntimeError as error:
        if config.deterministic:
            raise RuntimeError(
                "the requested deterministic PyTorch algorithms are unavailable"
            ) from error


def create_torch_generator(
    seed: int,
    device: Union[torch.device, str] = "cpu",
) -> torch.Generator:
    """Create an independently seeded PyTorch random-number generator.

    Input: Non-negative integer seed and a CPU/CUDA device specification.
    Output: A seeded ``torch.Generator`` associated with the requested device.
    Mathematical meaning: Provides an isolated random stream for operations
        such as rollout minibatch permutations without changing global policy
        sampling or model-initialization streams.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")
    target_device = torch.device(device)
    generator = torch.Generator(device=target_device)
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """Seed a data-loader worker from PyTorch's worker-specific initial seed.

    Input: Integer worker identifier supplied by a PyTorch data loader. The
        worker's PyTorch initial seed is read from the active worker context.
    Output: No value; Python and NumPy worker-local RNGs are seeded.
    Mathematical meaning: Ensures stochastic data transformations use distinct,
        deterministic streams instead of duplicating samples across workers.
    """
    if worker_id < 0:
        raise ValueError("worker_id must be non-negative")
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def seed_gym_environment(
    environment: Any,
    seed: int,
    seed_action_space: bool = True,
) -> None:
    """Seed a Gymnasium-compatible environment and optionally its action space.

    Input: Environment object, non-negative seed, and action-space seeding flag.
        The environment may expose either modern ``reset(seed=...)`` or a
        legacy ``seed(...)`` method.
    Output: No value; available environment RNGs are seeded in place.
    Mathematical meaning: Fixes the random transition/initial-state process so
        identical agent randomness is evaluated against the same MDP sample
        path, subject to environment implementation determinism.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")
    reset = getattr(environment, "reset", None)
    if reset is None:
        raise TypeError("environment must provide reset or seed")
    try:
        reset(seed=seed)
    except TypeError:
        legacy_seed = getattr(environment, "seed", None)
        if legacy_seed is None:
            raise TypeError("environment does not support seeded reset")
        legacy_seed(seed)

    if seed_action_space:
        action_space = getattr(environment, "action_space", None)
        seed_method = getattr(action_space, "seed", None)
        if seed_method is not None:
            seed_method(seed)


def derive_seed(base_seed: int, stream_id: int) -> int:
    """Derive a stable non-negative child seed for an experiment stream.

    Input: Non-negative base seed and non-negative stream identifier, such as
        environment index, evaluation stream, or BiGAN replay stream.
    Output: A deterministic integer seed in the valid NumPy/PyTorch range.
    Mathematical meaning: Separates stochastic processes while preserving a
        reproducible one-to-one mapping from experiment seed and stream.
    """
    if base_seed < 0 or stream_id < 0:
        raise ValueError("base_seed and stream_id must be non-negative")
    # Constants from SplitMix-style integer mixing provide distinct streams
    # without relying on Python's process-randomized hash implementation.
    mixed = (base_seed + 0x9E3779B97F4A7C15 * (stream_id + 1)) & ((1 << 63) - 1)
    mixed ^= mixed >> 30
    mixed = (mixed * 0xBF58476D1CE4E5B9) & ((1 << 63) - 1)
    mixed ^= mixed >> 27
    return mixed & ((1 << 32) - 1)
