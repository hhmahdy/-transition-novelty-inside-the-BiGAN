"""Execution and reproducibility metadata for experiment runs.

Two complementary files describe one run:

* ``config.json`` -- written by :class:`utils.logger.ExperimentLogger`, it is
  the complete **experimental configuration** (every frozen dataclass field).
* ``run_info.json`` -- written by this module, it is the **execution /
  reproducibility metadata**: git commit and dirty flag, timestamp, library
  versions, platform, resolved device, and the key settings of the run.

Neither file replaces the other. ``run_info.json`` makes a single seed
directory self-contained: it records the software state that produced the
numbers in ``metrics.jsonl`` without duplicating the whole configuration.

Every lookup in this module is best effort. A missing ``git`` binary, a
non-repository working directory, or an uninstalled optional package yields
``None`` instead of an exception, so metadata collection can never abort a
training run.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from config import ExperimentConfig

#: Repository root, i.e. the parent directory of the ``utils`` package.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def resolve_device_label(device_policy: str) -> str:
    """Resolve a device policy to the device label recorded for the run.

    Input: One of ``"auto"``, ``"cpu"``, or ``"cuda"``.
    Output: The device string that a trainer using this policy would select.
    Mathematical meaning: None; this records the execution substrate of the
        stochastic optimization process.
    """
    import torch

    if device_policy == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device_policy)


def git_information(repository_root: Union[str, Path] = REPOSITORY_ROOT) -> Dict[str, Any]:
    """Return the git commit, dirty flag, and branch of the working tree.

    Input: Repository root to inspect (read-only git commands).
    Output: Mapping with ``git_commit``, ``git_dirty``, and ``git_branch``;
        values are ``None`` when git information is unavailable.
    Mathematical meaning: None; this identifies the exact source code that
        produced the recorded measurements.
    """
    root = Path(repository_root)

    def _run(arguments: list[str]) -> Optional[str]:
        """Run one read-only git command and return its stripped stdout.

        Input: Git argument list.
        Output: Standard output text, or ``None`` when git is unavailable or
            the command fails (for example outside a repository).
        Mathematical meaning: None.
        """
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    commit = _run(["rev-parse", "HEAD"])
    status = _run(["status", "--porcelain"])
    branch = _run(["rev-parse", "--abbrev-ref", "HEAD"])
    return {
        "git_commit": commit,
        # ``git status --porcelain`` prints one line per modified/untracked
        # path, so an empty output means the working tree matches the commit.
        "git_dirty": None if status is None else bool(status),
        "git_branch": branch,
    }


def library_versions() -> Dict[str, Any]:
    """Return interpreter, library, and platform versions for the run.

    Input: The current process environment.
    Output: Mapping with ``python_version``, ``torch_version``,
        ``gymnasium_version``, ``ale_version``, ``numpy_version``,
        ``platform``, and ``machine``.
    Mathematical meaning: None; these identify the software stack that must be
        matched to reproduce the recorded results.
    """
    def _version(module_name: str, attribute: str = "__version__") -> Optional[str]:
        """Import a module and return one version attribute if available.

        Input: Module name and version attribute name.
        Output: Version string, or ``None`` when the module or attribute is
            missing.
        Mathematical meaning: None.
        """
        try:
            module = __import__(module_name)
        except ImportError:
            return None
        return str(getattr(module, attribute, None))

    return {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "torch_version": _version("torch"),
        "gymnasium_version": _version("gymnasium"),
        "ale_version": _version("ale_py"),
        "numpy_version": _version("numpy"),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def key_settings(
    config: ExperimentConfig,
    method: Optional[str],
    environment_id: Optional[str],
    device_label: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract the run settings a thesis appendix needs in one place.

    Input: Validated experiment configuration, the Master's method label
        (``"state"``, ``"transition"``, or ``None`` for legacy CLI usage), the
        environment identifier, and the resolved device label.
    Output: Flat mapping of environment, PPO, BiGAN, and novelty settings.
    Mathematical meaning: Records the parameter point of the optimization:
        discount factors, GAE lambdas, learning rates, clipping bounds, rollout
        geometry, and which intrinsic-reward signal was active.
    """
    ppo = config.ppo
    bigan = config.bigan
    novelty = config.novelty
    return {
        "method": method,
        "environment_id": environment_id,
        "device": device_label,
        "seed": config.seed.seed,
        "deterministic": config.seed.deterministic,
        "num_parallel_envs": config.environment.num_parallel_envs,
        "observation_shape": list(config.environment.observation_shape),
        "action_dim": config.environment.action_dim,
        "discrete_actions": config.environment.discrete_actions,
        "total_environment_steps": config.training.total_environment_steps,
        "rollout_steps": ppo.rollout_steps,
        "minibatch_size": ppo.minibatch_size,
        "update_epochs": ppo.update_epochs,
        "samples_per_update": config.environment.num_parallel_envs * ppo.rollout_steps,
        "total_updates": config.training.total_environment_steps
        // (config.environment.num_parallel_envs * ppo.rollout_steps),
        "gamma": ppo.gamma,
        "intrinsic_gamma": ppo.intrinsic_gamma,
        "gae_lambda": ppo.gae_lambda,
        "intrinsic_gae_lambda": ppo.intrinsic_gae_lambda,
        "learning_rate": ppo.learning_rate,
        "entropy_coefficient": ppo.entropy_coefficient,
        "clip_epsilon": ppo.clip_epsilon,
        "value_loss_coefficient": ppo.value_loss_coefficient,
        "intrinsic_advantage_coefficient": ppo.intrinsic_advantage_coefficient,
        "max_grad_norm": ppo.max_grad_norm,
        "lr_decay": ppo.lr_decay,
        "bigan_latent_dim": bigan.latent_dim,
        "bigan_learning_rate": bigan.learning_rate,
        "bigan_batch_size": bigan.batch_size,
        "novelty_type": novelty.novelty_type,
        "novelty_alpha": novelty.alpha,
        "transition_variant": novelty.transition_variant,
        "transition_hidden_dim": novelty.transition_hidden_dim,
        "transition_learning_rate": novelty.transition_learning_rate,
        "transition_batch_size": novelty.transition_batch_size,
        "transition_update_epochs": novelty.transition_update_epochs,
        "transition_max_grad_norm": novelty.transition_max_grad_norm,
        "normalization_epsilon": novelty.normalization_epsilon,
        "clip_intrinsic_reward": novelty.clip_intrinsic_reward,
        "use_intrinsic_reward": config.training.use_intrinsic_reward,
        "metric_eme_enabled": config.metric_eme.enabled,
        "resettable": config.training.resettable,
    }


def write_run_info(
    output_directory: Union[str, Path],
    config: ExperimentConfig,
    method: Optional[str] = None,
    environment_id: Optional[str] = None,
    device_label: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write ``run_info.json`` next to the run's ``config.json``.

    Input: Run output directory, validated configuration, Master's method
        label, environment identifier, resolved device label, and optional
        additional JSON-safe entries.
    Output: Path of the written ``run_info.json``.
    Mathematical meaning: None; this persists the execution metadata that makes
        the recorded stochastic measurements reproducible and auditable.
    """
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    if device_label is None:
        device_label = resolve_device_label(config.training.device.value)
    payload: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "environment_id": environment_id,
        "seed": config.seed.seed,
        "device": device_label,
        "num_parallel_envs": config.environment.num_parallel_envs,
        "rollout_steps": config.ppo.rollout_steps,
        "total_environment_steps": config.training.total_environment_steps,
        "output_directory": str(directory),
    }
    payload.update(git_information())
    payload.update(library_versions())
    payload["key_settings"] = key_settings(config, method, environment_id, device_label)
    if extra:
        payload.update(extra)
    path = directory / "run_info.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, default=str)
        file.write("\n")
    return path
