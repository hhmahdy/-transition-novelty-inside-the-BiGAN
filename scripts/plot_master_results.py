#!/usr/bin/env python3
"""Generate the five Master's thesis figures from recorded run logs.

Reads ``results/master/<method>/seed_<k>/metrics.jsonl`` and writes, into
``results/master/figures/``:

1. ``plot1_game_score_vs_steps.png``        true extrinsic game score vs.
   environment steps, every run overlaid (colour = method, one thin line per
   seed).
2. ``plot2_game_score_mean_std.png``        binned mean +/- standard deviation
   of the game score across seeds, per method.
3. ``plot3_game_score_individual_seeds.png`` individual seed learning curves,
   one subplot per method.
4. ``plot4_first_reward.png``               time of the first positive extrinsic
   reward, per method and seed.
5. ``plot5_intrinsic_reward.png``           intrinsic reward over training,
   binned mean +/- standard deviation across seeds, per method.

Only these five figures are produced; no additional diagnostics are generated.
The parsing helpers are reused from :mod:`evaluation.comparison` so the figures
always describe exactly what the trainer logged.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.comparison import (  # noqa: E402
    _rolling_mean,
    load_episode_scores,
    load_training_metric,
)

METHOD_COLORS = {"state": "#08519c", "transition": "#d62728"}
FALLBACK_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
METHOD_LABELS = {
    "state": "Baseline: BiGAN state novelty $B(s)$",
    "transition": "Proposed: BiGAN transition novelty $N_T$",
}


def method_color(method: str, index: int) -> str:
    """Return the colour assigned to one method.

    Input: Method label and its position in the method list.
    Output: Hexadecimal colour string.
    Mathematical meaning: None; a plotting convention that keeps the two
        compared conditions visually distinct across all five figures.
    """
    return METHOD_COLORS.get(method, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])


def method_label(method: str) -> str:
    """Return the legend label of one method.

    Input: Method label as used in the directory layout.
    Output: Human-readable label naming the intrinsic-reward signal.
    Mathematical meaning: None.
    """
    return METHOD_LABELS.get(method, method)


def collect_runs(results_directory: Path, methods: Sequence[str]) -> Dict[str, List[Tuple[int, Path]]]:
    """Discover seed directories that contain metrics.

    Input: Results root and method names.
    Output: Mapping ``method -> [(seed, run_directory)]``.
    Mathematical meaning: None.
    """
    discovered: Dict[str, List[Tuple[int, Path]]] = {}
    for method in methods:
        runs: List[Tuple[int, Path]] = []
        method_directory = results_directory / method
        if method_directory.is_dir():
            for candidate in sorted(method_directory.glob("seed_*")):
                if not (candidate / "metrics.jsonl").is_file():
                    continue
                try:
                    seed = int(candidate.name.split("seed_", 1)[1])
                except (IndexError, ValueError):
                    continue
                runs.append((seed, candidate))
        discovered[method] = runs
    return discovered


def binned_curve(
    steps: np.ndarray,
    values: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """Average one run's values inside fixed step bins.

    Input: Timestep array, value array, and monotonic bin edges.
    Output: Array of length ``len(edges) - 1`` holding the mean value per bin,
        ``NaN`` where the run recorded nothing.
    Mathematical meaning: Estimates the conditional mean of the recorded
        statistic over each training interval, which makes seeds with different
        event counts comparable on a common grid.
    """
    result = np.full(edges.size - 1, np.nan, dtype=np.float64)
    if steps.size == 0:
        return result
    indices = np.clip(np.digitize(steps, edges) - 1, 0, edges.size - 2)
    for index in range(edges.size - 1):
        selected = values[indices == index]
        if selected.size:
            result[index] = float(selected.mean())
    return result


def across_seed_statistics(
    curves: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine per-seed binned curves into mean, standard deviation, and count.

    Input: Sequence of equal-length binned curves, possibly containing ``NaN``.
    Output: ``(mean, std, count)`` arrays; ``NaN`` where no seed has data.
    Mathematical meaning: Sample mean and sample standard deviation across the
        reproducible seed runs of one method. With the small seed counts of a
        Master's thesis these are descriptive summaries, not significance tests.
    """
    if not curves:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty
    stacked = np.vstack([np.asarray(curve, dtype=np.float64) for curve in curves])
    # Bins in which a run recorded nothing are NaN; nanmean/nanstd warn about
    # those empty slices, which is expected here rather than an error.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(stacked, axis=0)
        count = np.sum(~np.isnan(stacked), axis=0)
        std = np.where(count > 1, np.nanstd(stacked, axis=0, ddof=1), 0.0)
    mean = np.where(count == 0, np.nan, mean)
    std = np.where(count == 0, np.nan, std)
    return mean, std, count.astype(np.float64)


def bin_centers(edges: np.ndarray) -> np.ndarray:
    """Return the centre of every bin.

    Input: Monotonic bin edges.
    Output: Array of length ``len(edges) - 1``.
    Mathematical meaning: None.
    """
    return 0.5 * (edges[:-1] + edges[1:])


def _style_axis(axis: plt.Axes, x_label: str, y_label: str, title: str) -> None:
    """Apply the shared thesis figure style to one axis.

    Input: Axis object and label/title strings.
    Output: No value; the axis is styled in place.
    Mathematical meaning: None.
    """
    axis.set_xlabel(x_label, fontsize=11)
    axis.set_ylabel(y_label, fontsize=11)
    axis.set_title(title, fontsize=13, fontweight="bold")
    axis.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.85)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_all_runs(
    runs: Dict[str, List[Tuple[int, Path]]],
    output_directory: Path,
    smoothing_window: int,
    dpi: int,
) -> Optional[Path]:
    """Plot 1: extrinsic game score vs. environment steps for every run.

    Input: Discovered runs, output directory, episode smoothing window, and DPI.
    Output: PNG path, or ``None`` when no episode was recorded.
    Mathematical meaning: Displays the primary metric -- the true extrinsic game
        score -- as a function of environment interaction count.
    """
    figure, axis = plt.subplots(figsize=(8.0, 5.0), dpi=dpi)
    plotted = False
    for index, (method, seeds) in enumerate(runs.items()):
        color = method_color(method, index)
        for position, (seed, directory) in enumerate(seeds):
            steps, scores = load_episode_scores(directory / "metrics.jsonl")
            if scores.size == 0:
                continue
            plotted = True
            label = method_label(method) if position == 0 else None
            axis.plot(
                steps,
                _rolling_mean(scores, smoothing_window),
                color=color,
                linewidth=1.6,
                alpha=0.85,
                label=label,
            )
    if not plotted:
        plt.close(figure)
        return None
    _style_axis(axis, "Environment Steps", "Extrinsic Game Score", "Game Score vs. Environment Steps")
    axis.legend(frameon=False, loc="best")
    figure.tight_layout()
    path = output_directory / "plot1_game_score_vs_steps.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_mean_std(
    runs: Dict[str, List[Tuple[int, Path]]],
    output_directory: Path,
    bins: int,
    dpi: int,
) -> Optional[Path]:
    """Plot 2: mean +/- standard deviation of the game score across seeds.

    Input: Discovered runs, output directory, number of step bins, and DPI.
    Output: PNG path, or ``None`` when no episode was recorded.
    Mathematical meaning: Summarizes the seed distribution of the episodic
        return over training progress.
    """
    per_method: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    maximum_step = 0.0
    for method, seeds in runs.items():
        curves: List[Tuple[np.ndarray, np.ndarray]] = []
        for _, directory in seeds:
            steps, scores = load_episode_scores(directory / "metrics.jsonl")
            if scores.size == 0:
                continue
            curves.append((steps, scores))
            maximum_step = max(maximum_step, float(steps.max()))
        per_method[method] = curves
    if maximum_step <= 0.0:
        return None
    edges = np.linspace(0.0, maximum_step, bins + 1)
    centers = bin_centers(edges)
    figure, axis = plt.subplots(figsize=(8.0, 5.0), dpi=dpi)
    for index, (method, pairs) in enumerate(per_method.items()):
        color = method_color(method, index)
        mean, std, _ = across_seed_statistics(
            [binned_curve(steps, scores, edges) for steps, scores in pairs]
        )
        if mean.size == 0 or np.all(np.isnan(mean)):
            continue
        # Very short runs populate only a few bins; without markers those
        # isolated points would be invisible between the NaN gaps.
        sparse = int(np.count_nonzero(~np.isnan(mean))) < 30
        axis.plot(
            centers,
            mean,
            color=color,
            linewidth=2.2,
            marker="o" if sparse else None,
            markersize=4.0 if sparse else 0.0,
            label=method_label(method),
        )
        axis.fill_between(
            centers,
            mean - std,
            mean + std,
            color=color,
            alpha=0.20,
            linewidth=0.0,
            label=f"{method}: $\\pm$ std over {len(pairs)} seed(s)",
        )
    _style_axis(axis, "Environment Steps", "Extrinsic Game Score", "Game Score: Mean $\\pm$ Std Across Seeds")
    axis.legend(frameon=False, loc="best", fontsize=9)
    figure.tight_layout()
    path = output_directory / "plot2_game_score_mean_std.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_individual_seeds(
    runs: Dict[str, List[Tuple[int, Path]]],
    output_directory: Path,
    smoothing_window: int,
    dpi: int,
) -> Optional[Path]:
    """Plot 3: individual seed learning curves, one subplot per method.

    Input: Discovered runs, output directory, smoothing window, and DPI.
    Output: PNG path, or ``None`` when no episode was recorded.
    Mathematical meaning: Reports every reproducible sample path separately, as
        required for honest small-seed reporting.
    """
    methods = [method for method, seeds in runs.items() if seeds]
    if not methods:
        return None
    figure, axes = plt.subplots(1, len(methods), figsize=(6.0 * len(methods), 4.5), dpi=dpi, squeeze=False)
    plotted = False
    for index, method in enumerate(methods):
        axis = axes[0][index]
        for seed, directory in runs[method]:
            steps, scores = load_episode_scores(directory / "metrics.jsonl")
            if scores.size == 0:
                continue
            plotted = True
            axis.plot(
                steps,
                _rolling_mean(scores, smoothing_window),
                linewidth=1.4,
                alpha=0.9,
                label=f"seed {seed}",
            )
        _style_axis(axis, "Environment Steps", "Extrinsic Game Score", method_label(method))
        axis.legend(frameon=False, loc="best", fontsize=9)
    if not plotted:
        plt.close(figure)
        return None
    figure.tight_layout()
    path = output_directory / "plot3_game_score_individual_seeds.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def first_reward_step(directory: Path) -> int:
    """Return the environment step of the first positive extrinsic reward.

    Input: Run directory containing ``metrics.jsonl``.
    Output: Step of the first positive episode score or of the first update with
        a positive mean extrinsic reward, whichever is earlier; ``-1`` when no
        reward was ever observed.
    Mathematical meaning: The discovery time of the sparse extrinsic reward,
        the standard exploration metric for hard-exploration Atari games.
    """
    episode_steps, episode_scores = load_episode_scores(directory / "metrics.jsonl")
    candidates: List[float] = []
    positive_episodes = episode_steps[episode_scores > 0.0]
    if positive_episodes.size:
        candidates.append(float(positive_episodes[0]))
    update_steps, extrinsic_means = load_training_metric(directory / "metrics.jsonl", "reward/extrinsic")
    positive_updates = update_steps[extrinsic_means > 0.0]
    if positive_updates.size:
        candidates.append(float(positive_updates[0]))
    return int(min(candidates)) if candidates else -1


def plot_first_reward(
    runs: Dict[str, List[Tuple[int, Path]]],
    output_directory: Path,
    dpi: int,
) -> Optional[Path]:
    """Plot 4: distribution/time of the first positive reward per method.

    Input: Discovered runs, output directory, and DPI.
    Output: PNG path, or ``None`` when no run exists.
    Mathematical meaning: Compares the discovery-time random variable across the
        two intrinsic-reward signals. Runs that never scored are marked
        explicitly rather than dropped, because "never discovered" is a valid
        experimental outcome on a sparse-reward game.
    """
    entries: List[Tuple[str, int, int]] = []
    for method, seeds in runs.items():
        for seed, directory in seeds:
            entries.append((method, seed, first_reward_step(directory)))
    if not entries:
        return None
    figure, axis = plt.subplots(figsize=(8.0, 5.0), dpi=dpi)
    tick_positions: List[float] = []
    tick_labels: List[str] = []
    method_indices = {method: index for index, method in enumerate(runs)}
    for position, (method, seed, step) in enumerate(entries):
        color = method_color(method, method_indices[method])
        tick_positions.append(float(position))
        tick_labels.append(f"{method}\nseed {seed}")
        if step < 0:
            axis.plot([float(position)], [0.0], marker="x", color=color, markersize=9)
            axis.annotate(
                "never",
                xy=(float(position), 0.0),
                xytext=(float(position), 0.05),
                ha="center",
                va="bottom",
                color=color,
                fontsize=9,
            )
        else:
            axis.plot([float(position)], [float(step)], marker="o", color=color, markersize=9)
    for method, seeds in runs.items():
        known = [
            float(step)
            for seed, directory in seeds
            for step in (first_reward_step(directory),)
            if step >= 0
        ]
        if not known or not seeds:
            continue
        first_position = float(next(index for index, entry in enumerate(entries) if entry[0] == method))
        color = method_color(method, method_indices[method])
        axis.hlines(
            float(np.mean(known)),
            first_position,
            first_position + len(seeds) - 1.0,
            color=color,
            linestyle="--",
            linewidth=1.4,
            label=f"{method_label(method)}: mean {float(np.mean(known)):.0f} ({len(known)}/{len(seeds)} seeds)",
        )
    known_values = [step for _, _, step in entries if step >= 0]
    if known_values:
        axis.set_ylim(0.0, float(max(known_values)) * 1.15)
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(tick_labels, fontsize=9)
    _style_axis(axis, "Run", "Environment Step of First Positive Reward", "First Positive Extrinsic Reward")
    axis.legend(frameon=False, loc="best", fontsize=9)
    figure.tight_layout()
    path = output_directory / "plot4_first_reward.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_intrinsic_reward(
    runs: Dict[str, List[Tuple[int, Path]]],
    output_directory: Path,
    bins: int,
    dpi: int,
) -> Optional[Path]:
    """Plot 5: intrinsic reward over training, mean +/- std across seeds.

    Input: Discovered runs, output directory, number of step bins, and DPI.
    Output: PNG path, or ``None`` when no intrinsic reward was logged.
    Mathematical meaning: Compares the magnitude and temporal profile of the two
        intrinsic signals ``r^int`` after the shared Equation (5) normalization.
    """
    maximum_step = 0.0
    per_method: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    for method, seeds in runs.items():
        curves: List[Tuple[np.ndarray, np.ndarray]] = []
        for _, directory in seeds:
            steps, values = load_training_metric(directory / "metrics.jsonl", "reward/intrinsic")
            if values.size == 0:
                continue
            curves.append((steps, values))
            maximum_step = max(maximum_step, float(steps.max()))
        per_method[method] = curves
    if maximum_step <= 0.0:
        return None
    edges = np.linspace(0.0, maximum_step, bins + 1)
    centers = bin_centers(edges)
    figure, axis = plt.subplots(figsize=(8.0, 5.0), dpi=dpi)
    for index, (method, pairs) in enumerate(per_method.items()):
        color = method_color(method, index)
        for steps, values in pairs:
            axis.plot(centers, binned_curve(steps, values, edges), color=color, linewidth=0.7, alpha=0.25)
        curves = [binned_curve(steps, values, edges) for steps, values in pairs]
        mean, std, _ = across_seed_statistics(curves)
        if mean.size == 0 or np.all(np.isnan(mean)):
            continue
        sparse = int(np.count_nonzero(~np.isnan(mean))) < 30
        axis.plot(
            centers,
            mean,
            color=color,
            linewidth=2.2,
            marker="o" if sparse else None,
            markersize=4.0 if sparse else 0.0,
            label=method_label(method),
        )
        axis.fill_between(centers, mean - std, mean + std, color=color, alpha=0.20, linewidth=0.0)
    _style_axis(axis, "Environment Steps", "Intrinsic Reward $r^{int}$", "Intrinsic Reward Over Training")
    axis.legend(frameon=False, loc="best")
    figure.tight_layout()
    path = output_directory / "plot5_intrinsic_reward.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse plotting options.

    Input: Optional argument list.
    Output: Parsed namespace.
    Mathematical meaning: None.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=Path("results/master"))
    parser.add_argument("--output-dir", type=Path, default=None, help="default: <results-dir>/figures")
    parser.add_argument("--methods", nargs="+", default=["state", "transition"])
    parser.add_argument("--bins", type=int, default=25)
    parser.add_argument("--smoothing-window", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Generate all five thesis figures.

    Input: Optional argument list.
    Output: Process exit status (``1`` when no run was found).
    Mathematical meaning: Renders the empirical comparison of the two
        intrinsic-reward signals.
    """
    args = parse_args(argv)
    results_directory: Path = args.results_dir
    output_directory: Path = args.output_dir or (results_directory / "figures")
    if not results_directory.is_dir():
        print(f"results directory not found: {results_directory}", file=sys.stderr)
        return 1
    runs = collect_runs(results_directory, args.methods)
    if not any(runs.values()):
        print(f"no runs found under {results_directory}", file=sys.stderr)
        return 1
    output_directory.mkdir(parents=True, exist_ok=True)
    generated = [
        plot_all_runs(runs, output_directory, args.smoothing_window, args.dpi),
        plot_mean_std(runs, output_directory, args.bins, args.dpi),
        plot_individual_seeds(runs, output_directory, args.smoothing_window, args.dpi),
        plot_first_reward(runs, output_directory, args.dpi),
        plot_intrinsic_reward(runs, output_directory, args.bins, args.dpi),
    ]
    for path in generated:
        if path is None:
            print("skipped one figure: the required metric was never logged")
        else:
            print(f"figure: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
