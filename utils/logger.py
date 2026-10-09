"""Structured experiment logging with JSONL, text, and TensorBoard output.

All scalar training diagnostics are persisted in machine-readable JSONL and
TensorBoard event files. Nested PPO/BiGAN metric dataclasses are flattened into
stable slash-separated TensorBoard tags.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import torch

from config import ExperimentConfig


class ExperimentLogger:
    """Write configuration and timestamped metrics to disk and TensorBoard."""

    def __init__(
        self,
        output_directory: Union[str, Path],
        experiment_config: Optional[ExperimentConfig] = None,
        flush_every: int = 1,
        tensorboard_directory: Optional[Union[str, Path]] = None,
        enable_tensorboard: Optional[bool] = None,
    ) -> None:
        """Create output streams and an optional TensorBoard writer.

        Input: Output directory, optional experiment configuration, flush
            interval, optional TensorBoard directory, and enable flag.
        Output: Initialized logger writing JSONL, text, and optionally
            TensorBoard event records.
        Mathematical meaning: Persists the objective and diagnostic trajectory
            of one stochastic training run.
        """
        if flush_every <= 0:
            raise ValueError("flush_every must be positive")
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self._records_written = 0
        self._closed = False
        self._metrics_file = (self.output_directory / "metrics.jsonl").open("a", encoding="utf-8")
        self._text_file = (self.output_directory / "run.log").open("a", encoding="utf-8")

        config_enable_tensorboard = experiment_config.training.enable_tensorboard if experiment_config else True
        self.enable_tensorboard = config_enable_tensorboard if enable_tensorboard is None else enable_tensorboard
        if tensorboard_directory is None and experiment_config is not None:
            tensorboard_directory = experiment_config.training.tensorboard_directory
        if tensorboard_directory is None:
            tensorboard_directory = self.output_directory / "tensorboard"
        self.tensorboard_directory = Path(tensorboard_directory)
        self._writer = self._create_summary_writer() if self.enable_tensorboard else None
        if experiment_config is not None:
            self.write_config(experiment_config)

    def _create_summary_writer(self) -> Any:
        """Create a TensorBoard SummaryWriter with an actionable dependency error.

        Input: Logger TensorBoard directory.
        Output: Initialized ``SummaryWriter``.
        Mathematical meaning: Creates the event stream for scalar objective
            observations indexed by environment timestep.
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise ImportError(
                "TensorBoard logging requires: python -m pip install tensorboard"
            ) from error
        self.tensorboard_directory.mkdir(parents=True, exist_ok=True)
        return SummaryWriter(log_dir=str(self.tensorboard_directory))

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Convert tensors, dataclasses, mappings, and sequences to JSON values.

        Input: Arbitrary metric or metadata value.
        Output: JSON-serializable value.
        Mathematical meaning: Records numerical diagnostics without changing
            their scalar values.
        """
        if is_dataclass(value):
            return ExperimentLogger._json_safe(asdict(value))
        if isinstance(value, torch.Tensor):
            detached = value.detach().cpu()
            return detached.item() if detached.numel() == 1 else detached.tolist()
        if isinstance(value, Mapping):
            return {str(key): ExperimentLogger._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [ExperimentLogger._json_safe(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if hasattr(value, "item"):
            return value.item()
        return str(value)

    @staticmethod
    def _flatten_scalars(value: Any, prefix: str = "") -> Dict[str, float]:
        """Flatten nested metrics into TensorBoard scalar tags.

        Input: Nested mappings, dataclasses, tensors, or scalar values and an
            optional tag prefix.
        Output: Mapping from slash-separated tags to finite numeric scalars.
        Mathematical meaning: Converts structured objective diagnostics into
            scalar empirical observations for TensorBoard.
        """
        if is_dataclass(value):
            value = asdict(value)
        if isinstance(value, Mapping):
            result: Dict[str, float] = {}
            for key, item in value.items():
                child_prefix = f"{prefix}/{key}" if prefix else str(key)
                result.update(ExperimentLogger._flatten_scalars(item, child_prefix))
            return result
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return {}
            value = value.detach().cpu().item()
        if isinstance(value, bool):
            return {prefix: float(value)} if prefix else {}
        if isinstance(value, (int, float)):
            return {prefix: float(value)} if prefix else {}
        return {}

    def write_config(self, experiment_config: ExperimentConfig) -> Path:
        """Persist the complete experiment configuration as JSON.

        Input: Validated experiment configuration.
        Output: Path to ``config.json``.
        Mathematical meaning: Records the exact parameter point of the run.
        """
        if self._closed:
            raise RuntimeError("logger is closed")
        path = self.output_directory / "config.json"
        with path.open("w", encoding="utf-8") as file:
            json.dump(experiment_config.to_dict(), file, indent=2, sort_keys=True)
            file.write("\n")
        return path

    def log(self, step: int, metrics: Mapping[str, Any]) -> None:
        """Append metrics to JSONL, text, and TensorBoard outputs.

        Input: Non-negative environment/update step and nested metric mapping.
        Output: No value; all scalar leaves are written to TensorBoard.
        Mathematical meaning: Records training objectives at a specific point
            in the optimization trajectory.
        """
        if self._closed:
            raise RuntimeError("logger is closed")
        if step < 0:
            raise ValueError("step must be non-negative")
        timestamp = datetime.now(timezone.utc).isoformat()
        record = {"timestamp": timestamp, "step": step, "metrics": self._json_safe(metrics)}
        self._metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
        metric_text = " ".join(f"{name}={value}" for name, value in record["metrics"].items())
        self._text_file.write(f"{timestamp} step={step} {metric_text}\n")
        if self._writer is not None:
            for tag, value in self._flatten_scalars(metrics).items():
                self._writer.add_scalar(tag, value, global_step=step)
        self._records_written += 1
        if self._records_written % self.flush_every == 0:
            self.flush()

    def log_scalar(self, tag: str, value: Union[int, float, Tensor], step: int) -> None:
        """Write one explicit scalar to TensorBoard.

        Input: Non-empty tag, scalar value, and non-negative step.
        Output: No value; scalar is written to the configured event stream.
        Mathematical meaning: Records one named training statistic.
        """
        if not tag:
            raise ValueError("tag must not be empty")
        if step < 0:
            raise ValueError("step must be non-negative")
        if self._writer is not None:
            scalar = value.detach().cpu().item() if isinstance(value, Tensor) else float(value)
            self._writer.add_scalar(tag, scalar, global_step=step)

    def log_message(self, message: str) -> None:
        """Append a timestamped operational message to the text log.

        Input: Non-empty message.
        Output: No value; message is persisted in ``run.log``.
        Mathematical meaning: Records a non-objective event such as checkpoint
            creation or environment initialization.
        """
        if self._closed:
            raise RuntimeError("logger is closed")
        if not message:
            raise ValueError("message must not be empty")
        timestamp = datetime.now(timezone.utc).isoformat()
        self._text_file.write(f"{timestamp} {message}\n")
        self.flush()

    def flush(self) -> None:
        """Flush JSONL, text, and TensorBoard event streams.

        Input: This logger.
        Output: No value; buffered records are flushed.
        Mathematical meaning: Makes diagnostics durable during training.
        """
        if self._closed:
            return
        self._metrics_file.flush()
        self._text_file.flush()
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        """Flush and close all logger resources exactly once.

        Input: This logger.
        Output: No value; future writes are rejected.
        Mathematical meaning: Finalizes one experiment's persistent record.
        """
        if self._closed:
            return
        self.flush()
        if self._writer is not None:
            self._writer.close()
        self._metrics_file.close()
        self._text_file.close()
        self._closed = True

    def __enter__(self) -> "ExperimentLogger":
        """Enter a logger context.

        Input: This logger.
        Output: The same open logger.
        Mathematical meaning: Begins a bounded logging lifetime.
        """
        if self._closed:
            raise RuntimeError("logger is closed")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Close resources when leaving a logger context.

        Input: Standard context-manager exception information.
        Output: No value; logger is closed.
        Mathematical meaning: Finalizes logs even when training raises.
        """
        self.close()
