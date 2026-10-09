"""Evaluation and publication-quality comparison plotting utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def load_episode_scores(metrics_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Load episode timesteps and game scores from JSONL metrics.

    Input: Path to ``metrics.jsonl`` produced by ``ExperimentLogger``.
    Output: ``(timesteps, scores)`` arrays for completed episodes.
    Mathematical meaning: Recovers empirical episodic returns indexed by
        environment interaction count.
    """
    timesteps: List[float] = []
    scores: List[float] = []
    with Path(metrics_path).open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            metrics = record.get("metrics", {})
            score = metrics.get("episode/game_score")
            if score is not None:
                timesteps.append(float(record["step"]))
                scores.append(float(score))
    return np.asarray(timesteps), np.asarray(scores)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Compute a causal rolling mean over score values.

    Input: One-dimensional values and positive window.
    Output: Same-length rolling-mean array.
    Mathematical meaning: Estimates the smoothed empirical training metric.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    result = np.empty_like(values, dtype=np.float64)
    for index in range(values.size):
        result[index] = values[max(0, index + 1 - window) : index + 1].mean()
    return result


def load_training_metric(
    metrics_path: Union[str, Path],
    metric_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a scalar metric logged at training-update timesteps.

    Input: JSONL metrics path and nested metric name such as ``reward/total``.
    Output: Timestep and metric-value arrays.
    Mathematical meaning: Recovers the empirical update-level statistic used
        for training-dynamics plots.
    """
    timesteps: List[float] = []
    values: List[float] = []
    with Path(metrics_path).open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            value = record.get("metrics", {}).get(metric_name)
            if value is not None:
                timesteps.append(float(record["step"]))
                values.append(float(value))
    return np.asarray(timesteps), np.asarray(values)


def save_multi_game_score_comparison(
    runs: Dict[str, Union[str, Path]],
    output_directory: Union[str, Path],
    smoothing_window: int = 10,
    dpi: int = 300,
    metric_name: str = "episode/game_score",
    x_label: str = "Timesteps",
    y_label: str = "Game Score",
    file_stem: str = "game_score_comparison",
) -> Tuple[Path, Path]:
    """Save a multi-variant metric CSV and PNG.

    Input: Mapping from curve labels to run directories or metrics JSONL paths,
        output directory, smoothing window, DPI, metric name, axis labels, and
        output filename stem.
    Output: ``(csv_path, png_path)`` for all supplied variants.
    Mathematical meaning: Compares empirical learning curves under identical
        timestep axes while preserving raw and smoothed metric values.
    """
    if not runs:
        raise ValueError("runs must not be empty")
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    loaded: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for label, path in runs.items():
        candidate = Path(path)
        metrics_path = candidate / "metrics.jsonl" if candidate.is_dir() else candidate
        if metric_name == "episode/game_score":
            timesteps, scores = load_episode_scores(metrics_path)
        else:
            timesteps, scores = load_training_metric(metrics_path, metric_name)
        if scores.size == 0:
            raise ValueError(f"run {label!r} contains no {metric_name!r} metrics")
        loaded[label] = (timesteps, scores, _rolling_mean(scores, smoothing_window))

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / f"{file_stem}.csv"
    labels = list(loaded.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        header: List[str] = []
        for label in labels:
            safe_label = label.lower().replace(" ", "_").replace("+", "plus")
            header.extend([f"{safe_label}_timesteps", f"{safe_label}_score", f"{safe_label}_score_mean"])
        writer.writerow(header)
        rows = max(values[0].size for values in loaded.values())
        for index in range(rows):
            row: List[Union[float, str]] = []
            for label in labels:
                timesteps, scores, means = loaded[label]
                row.extend(
                    [
                        timesteps[index] if index < timesteps.size else "",
                        scores[index] if index < scores.size else "",
                        means[index] if index < means.size else "",
                    ]
                )
            writer.writerow(row)

    figure, axis = plt.subplots(figsize=(8.0, 5.0), dpi=dpi)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    for index, (label, (timesteps, _, means)) in enumerate(loaded.items()):
        axis.plot(
            timesteps,
            means,
            linewidth=2.2,
            color=colors[index % len(colors)],
            label=label,
        )
    axis.set_xlabel(x_label, fontsize=12)
    axis.set_ylabel(y_label, fontsize=12)
    axis.set_title(f"{y_label} vs. Training Steps", fontsize=14, fontweight="bold")
    axis.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.85)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, loc="best")
    figure.tight_layout()
    png_path = output / f"{file_stem}.png"
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return csv_path, png_path


def save_game_score_comparison(
    baseline_metrics: Union[str, Path],
    transition_metrics: Union[str, Path],
    output_directory: Union[str, Path],
    smoothing_window: int = 10,
    dpi: int = 300,
) -> Tuple[Path, Path]:
    """Save the backward-compatible two-run baseline/transition comparison.

    Input: Baseline and transition metrics paths, output directory, smoothing
        window, and DPI.
    Output: Comparison CSV and PNG paths.
    Mathematical meaning: Provides the original two-curve comparison API.
    """
    return save_multi_game_score_comparison(
        {
            "Adventurer": baseline_metrics,
            "Adventurer + Transition Novelty": transition_metrics,
        },
        output_directory,
        smoothing_window,
        dpi,
    )
