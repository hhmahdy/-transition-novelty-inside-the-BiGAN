"""Training visualization utilities for Adventurer experiments.

This module records episode/game scores and periodically writes a publication-
style ``Game Score vs. Timesteps`` PNG. It uses Matplotlib's non-interactive
Agg backend so plotting works on headless research clusters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ScoreRecord:
    """One completed episode score and its global environment timestep."""

    timestep: int
    score: float
    episode_length: int


class ScoreHistory:
    """Store episode scores and calculate rolling statistics."""

    def __init__(self, smoothing_window: int = 10) -> None:
        """Initialize an empty score history.

        Input: Positive rolling-average window size.
        Output: Empty score history.
        Mathematical meaning: Defines the empirical sequence of game returns
            used for the score-versus-timestep curve.
        """
        if smoothing_window <= 0:
            raise ValueError("smoothing_window must be positive")
        self.smoothing_window = smoothing_window
        self.records: List[ScoreRecord] = []

    def append(self, timestep: int, score: float, episode_length: int) -> None:
        """Append one completed episode record.

        Input: Non-negative global timestep, game score, and positive episode
            length.
        Output: No value; record is appended in collection order.
        Mathematical meaning: Adds one sample from the episode-return process
            for plotting against training progress.
        """
        if timestep < 0:
            raise ValueError("timestep must be non-negative")
        if episode_length <= 0:
            raise ValueError("episode_length must be positive")
        self.records.append(ScoreRecord(timestep, float(score), episode_length))

    def arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return timesteps, raw scores, and rolling-average scores.

        Input: Current score history.
        Output: Three NumPy arrays with equal length.
        Mathematical meaning: Provides the empirical game-score curve and its
            causal rolling estimate over the most recent window.
        """
        if not self.records:
            empty = np.asarray([], dtype=np.float64)
            return empty, empty, empty
        timesteps = np.asarray([record.timestep for record in self.records], dtype=np.float64)
        scores = np.asarray([record.score for record in self.records], dtype=np.float64)
        rolling = np.empty_like(scores)
        for index in range(scores.size):
            start = max(0, index + 1 - self.smoothing_window)
            rolling[index] = scores[start : index + 1].mean()
        return timesteps, scores, rolling

    def latest(self) -> Optional[ScoreRecord]:
        """Return the most recent score record, if one exists.

        Input: Current score history.
        Output: Latest ``ScoreRecord`` or ``None`` when no episode is complete.
        Mathematical meaning: Provides the latest observed game-return sample.
        """
        return self.records[-1] if self.records else None


class TrainingPlotter:
    """Save configurable Game Score versus Timesteps figures."""

    def __init__(
        self,
        output_directory: Union[str, Path],
        smoothing_window: int = 10,
        interval_updates: int = 1,
        figure_width: float = 8.0,
        figure_height: float = 5.0,
        dpi: int = 150,
    ) -> None:
        """Initialize score history and plot output configuration.

        Input: Output directory, smoothing window, save interval, figure size,
            and DPI.
        Output: Configured plotter with no saved records.
        Mathematical meaning: Defines the visualization estimator and sampling
            frequency for the game-return learning curve.
        """
        if interval_updates <= 0:
            raise ValueError("interval_updates must be positive")
        if figure_width <= 0.0 or figure_height <= 0.0 or dpi <= 0:
            raise ValueError("figure dimensions and dpi must be positive")
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.history = ScoreHistory(smoothing_window)
        self.interval_updates = interval_updates
        self.figure_width = figure_width
        self.figure_height = figure_height
        self.dpi = dpi

    def record_episode(self, timestep: int, score: float, episode_length: int) -> None:
        """Record one completed episode for later visualization.

        Input: Global environment timestep, game score, and episode length.
        Output: No value; updates the score history.
        Mathematical meaning: Adds one observation of the episodic return
            random variable at a defined training progress point.
        """
        self.history.append(timestep, score, episode_length)

    def plot_game_score(self, update: Optional[int] = None) -> Optional[Path]:
        """Render and save the Game Score versus Timesteps figure.

        Input: Optional PPO update index used in the output filename.
        Output: PNG path, or ``None`` if no episode scores exist.
        Mathematical meaning: Visualizes raw game returns and their causal
            rolling estimate as functions of environment interaction count.
        """
        timesteps, scores, rolling = self.history.arrays()
        if scores.size == 0:
            return None
        figure, axis = plt.subplots(figsize=(self.figure_width, self.figure_height), dpi=self.dpi)
        figure.patch.set_facecolor("white")
        axis.set_facecolor("white")
        axis.plot(
            timesteps,
            scores,
            color="#9ecae1",
            linewidth=1.0,
            alpha=0.65,
            label="Game Score",
        )
        axis.plot(
            timesteps,
            rolling,
            color="#08519c",
            linewidth=2.2,
            label=f"{self.history.smoothing_window}-Episode Mean",
        )
        axis.set_xlabel("Timesteps", fontsize=11)
        axis.set_ylabel("Game Score", fontsize=11)
        axis.set_title("Game Score vs. Timesteps", fontsize=13, fontweight="bold")
        axis.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(frameon=False, loc="best")
        figure.tight_layout()
        suffix = f"_{update:06d}" if update is not None else ""
        path = self.output_directory / f"game_score_vs_timesteps{suffix}.png"
        figure.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(figure)
        return path

    def maybe_plot(self, update: int, force: bool = False) -> Optional[Path]:
        """Save a plot when the configured update interval is reached.

        Input: Non-negative update index and optional force flag.
        Output: Saved PNG path when plotting occurs, otherwise ``None``.
        Mathematical meaning: Samples the evolving return curve at controlled
            intervals without changing the training algorithm.
        """
        if update < 0:
            raise ValueError("update must be non-negative")
        if force or update % self.interval_updates == 0:
            return self.plot_game_score(update)
        return None

    def close(self) -> Optional[Path]:
        """Save the final score figure.

        Input: Current plotter state.
        Output: Final PNG path, or ``None`` if no episode was completed.
        Mathematical meaning: Produces the terminal visualization of learning
            progress after the final optimization update.
        """
        return self.plot_game_score(None)
