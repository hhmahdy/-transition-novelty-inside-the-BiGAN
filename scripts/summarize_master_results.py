#!/usr/bin/env python3
"""Aggregate Master's thesis results into CSV tables.

The Master's experiment compares exactly two intrinsic-reward signals:

* ``state``      -- Adventurer BiGAN state novelty ``B(s)`` (baseline)
* ``transition`` -- BiGAN action-conditioned transition novelty
                    ``N_T = ||f(E(s_t),a_t) - E(s_{t+1})||_2`` (proposed)

Expected layout (produced by ``scripts/run_master_experiments.sh``)::

    results/master/
    ├── state/seed_0/       config.json  run_info.json  metrics.jsonl  run.log
    ├── state/seed_1/       ...
    ├── transition/seed_0/  ...
    └── ...

This script reads the ``metrics.jsonl`` of every seed directory and writes

* ``<results-dir>/master_summary.csv``        one row per (method, seed)
* ``<results-dir>/master_summary_stats.csv``  mean/std/median per method
* ``<results-dir>/<method>/seed_<k>/metrics.csv``  update-level metric table

Metric definitions (deliberately simple and explicit):

``final_score``
    Mean true extrinsic game score over the last ``--final-window`` completed
    episodes (the last episode alone when fewer were completed). Empty when no
    episode finished.
``total_extrinsic_return``
    Cumulative extrinsic reward over all logged transitions, recovered from the
    per-update mean as ``sum_updates mean(r^e) * rollout_steps * num_envs``.
``first_reward_step``
    Environment step of the first positive extrinsic signal, i.e. the earlier of
    the first completed episode with ``game_score > 0`` and the first update
    whose mean ``reward/extrinsic > 0``. ``-1`` when no reward was ever obtained
    -- a legitimate and informative outcome on a sparse-reward game.
``episode_count``
    Number of completed episodes recorded during training.

No statistical significance is claimed: with three seeds the summary reports
mean, standard deviation, and median only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

# Reuse the project's existing JSONL readers instead of duplicating parsing.
from evaluation.comparison import load_episode_scores, load_training_metric  # noqa: E402

SUMMARY_COLUMNS = [
    "method",
    "seed",
    "final_score",
    "total_extrinsic_return",
    "first_reward_step",
    "episode_count",
]

#: Update-level metrics exported to each seed's ``metrics.csv``.
UPDATE_METRICS = [
    "update",
    "reward/extrinsic",
    "reward/intrinsic",
    "reward/total",
    "novelty/pixel",
    "novelty/feature",
    "novelty/normalized",
    "ppo/policy_loss",
    "ppo/value_loss",
    "ppo/entropy",
    "ppo/approximate_kl",
    "ppo/clip_fraction",
    "ppo/learning_rate",
    "bigan/discriminator_loss",
    "bigan/encoder_generator_loss",
    "bigan/reconstruction_loss",
    "transition/forward_model_mse",
    "transition/mean_prediction_error",
    "transition/forward_prediction_loss",
    "transition/feature_matching_loss",
]


@dataclass(frozen=True)
class RunSummary:
    """Aggregated results of one (method, seed) run."""

    method: str
    seed: int
    run_directory: Path
    final_score: Optional[float]
    total_extrinsic_return: float
    first_reward_step: int
    episode_count: int
    total_environment_steps: int
    mean_episode_length: Optional[float]
    max_score: Optional[float]
    mean_intrinsic_reward: Optional[float]
    environment_id: Optional[str]
    git_commit: Optional[str]
    notes: List[str] = field(default_factory=list)


def _flatten(metrics: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten one nested JSONL metric mapping into slash-separated keys.

    Input: Metric mapping as written by :class:`utils.logger.ExperimentLogger`
        and an optional key prefix.
    Output: Flat mapping such as ``{"transition/forward_model_mse": 0.01}``.
    Mathematical meaning: None; this only reshapes the recorded diagnostics so
        they can be written to a spreadsheet-friendly CSV.
    """
    flat: Dict[str, Any] = {}
    for key, value in metrics.items():
        tag = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten(value, tag))
        else:
            flat[tag] = value
    return flat


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a metrics JSONL file into a list of records.

    Input: Path to ``metrics.jsonl``.
    Output: List of ``{"step": int, "metrics": {...}}`` records with nested
        metrics flattened.
    Mathematical meaning: Recovers the recorded empirical training trajectory.
    """
    records: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records.append(
                {"step": int(record.get("step", 0)), "metrics": _flatten(record.get("metrics", {}))}
            )
    return records


def _read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON file when present.

    Input: Path to a JSON document.
    Output: Parsed mapping, or an empty mapping when the file does not exist.
    Mathematical meaning: None.
    """
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def find_run_directories(results_directory: Path, methods: Sequence[str]) -> Dict[str, List[Tuple[int, Path]]]:
    """Locate every seed directory of every requested method.

    Input: Results root directory and method names.
    Output: Mapping ``method -> [(seed, run_directory)]`` sorted by seed.
    Mathematical meaning: None; this discovers the experiment layout.
    """
    discovered: Dict[str, List[Tuple[int, Path]]] = {}
    for method in methods:
        method_directory = results_directory / method
        runs: List[Tuple[int, Path]] = []
        if method_directory.is_dir():
            for candidate in sorted(method_directory.glob("seed_*")):
                if not candidate.is_dir():
                    continue
                suffix = candidate.name.split("seed_", 1)[1]
                try:
                    seed = int(suffix)
                except ValueError:
                    continue
                if (candidate / "metrics.jsonl").is_file():
                    runs.append((seed, candidate))
        discovered[method] = runs
    return discovered


def write_seed_metrics_csv(run_directory: Path) -> Optional[Path]:
    """Write the update-level ``metrics.csv`` of one run.

    Input: Run directory containing ``metrics.jsonl``.
    Output: Path of the written CSV, or ``None`` when no update record exists.
    Mathematical meaning: Exports the recorded per-update statistics
        (rewards, novelty terms, PPO losses, forward-model diagnostics).
    """
    records = _read_jsonl(run_directory / "metrics.jsonl")
    update_records = [record for record in records if "update" in record["metrics"]]
    if not update_records:
        return None
    columns = ["step"] + UPDATE_METRICS
    path = run_directory / "metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for record in update_records:
            metrics = record["metrics"]
            writer.writerow(
                [record["step"]] + [metrics.get(column, "") for column in UPDATE_METRICS]
            )
    return path


def summarize_run(method: str, seed: int, run_directory: Path, final_window: int) -> RunSummary:
    """Aggregate one run's logs into the Master's summary statistics.

    Input: Method label, seed, run directory, and final-score window size.
    Output: Populated :class:`RunSummary`.
    Mathematical meaning: Computes the empirical episodic return statistics and
        the cumulative extrinsic reward of one reproducible sample path.
    """
    notes: List[str] = []
    metrics_path = run_directory / "metrics.jsonl"
    episode_steps, episode_scores = load_episode_scores(metrics_path)
    episode_count = int(episode_scores.size)
    if episode_count:
        window = episode_scores[-final_window:] if final_window > 0 else episode_scores
        final_score: Optional[float] = float(window.mean())
        max_score: Optional[float] = float(episode_scores.max())
    else:
        final_score = None
        max_score = None
        notes.append("no completed episode recorded")

    _, episode_lengths = load_training_metric(metrics_path, "episode/length")
    mean_episode_length = float(episode_lengths.mean()) if episode_lengths.size else None

    update_steps, extrinsic_means = load_training_metric(metrics_path, "reward/extrinsic")
    _, intrinsic_means = load_training_metric(metrics_path, "reward/intrinsic")

    configuration = _read_json(run_directory / "config.json")
    rollout_steps = int(configuration.get("ppo", {}).get("rollout_steps", 0))
    num_envs = int(configuration.get("environment", {}).get("num_parallel_envs", 0))
    total_environment_steps = int(configuration.get("training", {}).get("total_environment_steps", 0))
    samples_per_update = rollout_steps * num_envs
    if samples_per_update > 0 and extrinsic_means.size:
        total_extrinsic_return = float(extrinsic_means.sum() * samples_per_update)
    else:
        total_extrinsic_return = 0.0
        notes.append("rollout geometry unavailable; total_extrinsic_return set to 0")

    first_reward_step = -1
    positive_episodes = episode_steps[episode_scores > 0.0]
    positive_updates = update_steps[extrinsic_means > 0.0]
    candidates: List[float] = []
    if positive_episodes.size:
        candidates.append(float(positive_episodes[0]))
    if positive_updates.size:
        candidates.append(float(positive_updates[0]))
    if candidates:
        first_reward_step = int(min(candidates))

    run_info = _read_json(run_directory / "run_info.json")
    environment_id = run_info.get("environment_id")
    git_commit = run_info.get("git_commit")
    if not run_info:
        notes.append("run_info.json missing")
    return RunSummary(
        method=method,
        seed=seed,
        run_directory=run_directory,
        final_score=final_score,
        total_extrinsic_return=total_extrinsic_return,
        first_reward_step=first_reward_step,
        episode_count=episode_count,
        total_environment_steps=total_environment_steps,
        mean_episode_length=mean_episode_length,
        max_score=max_score,
        mean_intrinsic_reward=float(intrinsic_means.mean()) if intrinsic_means.size else None,
        environment_id=environment_id,
        git_commit=git_commit,
        notes=notes,
    )


def write_summary_csv(summaries: Sequence[RunSummary], path: Path) -> Path:
    """Write ``master_summary.csv`` with one row per (method, seed).

    Input: Aggregated run summaries and destination path.
    Output: The written CSV path.
    Mathematical meaning: Records the per-seed measurements the thesis reports
        individually, without averaging away seed variability.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(SUMMARY_COLUMNS)
        for summary in summaries:
            writer.writerow(
                [
                    summary.method,
                    summary.seed,
                    "" if summary.final_score is None else f"{summary.final_score:.6f}",
                    f"{summary.total_extrinsic_return:.6f}",
                    summary.first_reward_step,
                    summary.episode_count,
                ]
            )
    return path


def write_stats_csv(summaries: Sequence[RunSummary], path: Path) -> Path:
    """Write per-method mean/std/median statistics.

    Input: Aggregated run summaries and destination path.
    Output: The written CSV path.
    Mathematical meaning: Summarizes the seed-to-seed variability of the final
        score, cumulative extrinsic return, and first-reward step. With the
        small seed counts of a Master's thesis these are descriptive statistics
        only; no significance test is performed or implied.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    methods: List[str] = []
    for summary in summaries:
        if summary.method not in methods:
            methods.append(summary.method)
    columns = ["method", "seeds", "metric", "mean", "std", "median", "min", "max", "values"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for method in methods:
            subset = [summary for summary in summaries if summary.method == method]
            seeds = " ".join(str(summary.seed) for summary in subset)
            for metric_name, values in (
                ("final_score", [s.final_score for s in subset]),
                ("total_extrinsic_return", [s.total_extrinsic_return for s in subset]),
                ("first_reward_step", [float(s.first_reward_step) for s in subset]),
                ("episode_count", [float(s.episode_count) for s in subset]),
            ):
                available = np.asarray(
                    [value for value in values if value is not None and math.isfinite(value)],
                    dtype=np.float64,
                )
                if available.size == 0:
                    writer.writerow([method, seeds, metric_name, "", "", "", "", "", "none"])
                    continue
                standard_deviation = float(available.std(ddof=1)) if available.size > 1 else 0.0
                writer.writerow(
                    [
                        method,
                        seeds,
                        metric_name,
                        f"{float(available.mean()):.6f}",
                        f"{standard_deviation:.6f}",
                        f"{float(np.median(available)):.6f}",
                        f"{float(available.min()):.6f}",
                        f"{float(available.max()):.6f}",
                        " ".join(f"{value:.6g}" for value in available),
                    ]
                )
    return path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse aggregation options.

    Input: Optional argument list (defaults to the process arguments).
    Output: Parsed namespace.
    Mathematical meaning: None; selects which runs to aggregate.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=Path("results/master"))
    parser.add_argument("--methods", nargs="+", default=["state", "transition"])
    parser.add_argument("--final-window", type=int, default=10)
    parser.add_argument("--summary-name", default="master_summary.csv")
    parser.add_argument("--stats-name", default="master_summary_stats.csv")
    parser.add_argument("--no-seed-csv", action="store_true", help="skip per-seed metrics.csv")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Aggregate all discovered runs and report them.

    Input: Optional argument list.
    Output: Process exit status (``1`` when no run was found).
    Mathematical meaning: Converts recorded training logs into the per-seed and
        per-method tables reported in the thesis.
    """
    args = parse_args(argv)
    results_directory: Path = args.results_dir
    if not results_directory.is_dir():
        print(f"results directory not found: {results_directory}", file=sys.stderr)
        return 1
    discovered = find_run_directories(results_directory, args.methods)
    summaries: List[RunSummary] = []
    for method, runs in discovered.items():
        for seed, run_directory in runs:
            if not args.no_seed_csv:
                write_seed_metrics_csv(run_directory)
            summaries.append(summarize_run(method, seed, run_directory, args.final_window))
    if not summaries:
        print(f"no runs found under {results_directory} for methods {args.methods}", file=sys.stderr)
        return 1
    summary_path = write_summary_csv(summaries, results_directory / args.summary_name)
    stats_path = write_stats_csv(summaries, results_directory / args.stats_name)

    print(f"Runs aggregated: {len(summaries)}")
    print(f"Summary CSV:     {summary_path}")
    print(f"Statistics CSV:  {stats_path}")
    print()
    header = f"| {'Method':<11} | {'Seed':>4} | {'Final Score':>12} | {'Total Extrinsic':>16} | {'First Reward':>13} | {'Episodes':>8} |"
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")
    for summary in summaries:
        final = "n/a" if summary.final_score is None else f"{summary.final_score:.2f}"
        first = "never" if summary.first_reward_step < 0 else f"{summary.first_reward_step}"
        print(
            f"| {summary.method:<11} | {summary.seed:>4} | {final:>12} | "
            f"{summary.total_extrinsic_return:>16.2f} | {first:>13} | {summary.episode_count:>8} |"
        )
    print()
    for method in args.methods:
        subset = [summary for summary in summaries if summary.method == method]
        if not subset:
            continue
        scores = [s.final_score for s in subset if s.final_score is not None]
        if scores:
            array = np.asarray(scores, dtype=np.float64)
            spread = float(array.std(ddof=1)) if array.size > 1 else 0.0
            print(
                f"{method}: final_score mean={array.mean():.2f} std={spread:.2f} "
                f"median={float(np.median(array)):.2f} over {array.size} seed(s)"
            )
        else:
            print(f"{method}: no completed episodes to score")
    for summary in summaries:
        if summary.notes:
            print(f"note [{summary.method}/seed_{summary.seed}]: {'; '.join(summary.notes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
