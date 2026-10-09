r"""Episodic visit-count habituation over the BiGAN latent space.

A raw consecutive-state distance ``d_t = ||E(s_t) - E(s_{t+1})||_p`` never
habituates: walking back and forth in a fully explored room keeps producing a
large distance, and the largest frame change of all is death/respawn. RIDE and
NovelD fix this by discounting the metric with the *episodic* visit count of
the reached state,

.. math:: b_t = d_t \, / \, \sqrt{N_{ep}(s_{t+1})},

where :math:`N_{ep}` counts visits within the current episode only. The first
visit of a state pays the full distance; every revisit within the same episode
decays with :math:`1/\sqrt{N}`. Counts are stored per environment slot in a
flat table keyed by a discretized latent code, and a slot's table is cleared
whenever that environment terminates.

The discretization deliberately trades precision for robustness: latent codes
(rescaled to ``[-1, 1]``) are rounded onto a grid with ``resolution`` steps per
coordinate, so near-identical states share a bucket. This is the standard
bucketing used by episodic-count exploration methods.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
from torch import Tensor


class EpisodicLatentCountScaling:
    """Maintain per-slot episodic counts of discretized latent codes.

    Args:
        num_envs: Number of parallel environment slots; row ``i`` of any batch
            belongs to slot ``i``.
        resolution: Grid steps per latent coordinate used to bucket codes.
            Codes are clamped to ``[-1, 1]`` and multiplied by this value, so
            the key space per coordinate is ``2*resolution + 1`` buckets.
        device: Ignored beyond tensor placement of outputs; counters live on
            the CPU as plain Python dictionaries.

    Notes:
        Callers must pass *normalized* latent codes (unit sphere) or codes that
        naturally live in ``[-1, 1]``; raw unnormalized codes would collapse
        onto a handful of extreme buckets.
    """

    def __init__(self, num_envs: int, resolution: int = 32) -> None:
        """Create one empty count table per environment slot.

        Input: Positive parallel-environment count and grid resolution.
        Output: Initialized counter.
        Mathematical meaning: Defines the episodic visit maps
            ``N_ep^{(i)}: keys -> counts`` for every slot ``i``.
        """
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        self.num_envs = int(num_envs)
        self.resolution = int(resolution)
        self.tables: List[Dict[bytes, int]] = [dict() for _ in range(self.num_envs)]

    def reset(self, done: Optional[Tensor]) -> None:
        """Clear the count tables of every environment that finished an episode.

        Input: Boolean done mask ``[num_envs]`` (``None`` is a no-op).
        Output: No value; the corresponding tables become empty.
        Mathematical meaning: Restarts ``N_ep`` for the new episodes, because
            episodic novelty is defined relative to the current trajectory.
        """
        if done is None:
            return
        done_cpu = done.detach().cpu().reshape(-1).bool()
        for index in torch.nonzero(done_cpu, as_tuple=False).flatten().tolist():
            self.tables[index].clear()

    @torch.no_grad()
    def scale(self, codes: Tensor) -> Tensor:
        """Register one state batch and return the ``1/sqrt(N_ep)`` factors.

        Input: Latent codes of the reached states ``s_{t+1}``, shaped
            ``[num_envs, latent_dim]``; row ``i`` belongs to slot ``i``. Codes
            are clamped to ``[-1, 1]`` before bucketing.
        Output: Tensor ``[num_envs]`` of habituation factors.
        Mathematical meaning: Increments ``N_ep(s_{t+1})`` per slot and returns
            ``1/sqrt(N_ep)``; the very first visit yields exactly one.
        """
        if codes.shape[0] != self.num_envs:
            raise ValueError("codes must provide one row per environment slot")
        keys = (codes.clamp(-1.0, 1.0) * self.resolution).round().to(torch.int8)
        keys_cpu = keys.cpu().numpy()
        factors = torch.empty(self.num_envs, dtype=torch.float32)
        for index in range(self.num_envs):
            key = keys_cpu[index].tobytes()
            count = self.tables[index].get(key, 0) + 1
            self.tables[index][key] = count
            factors[index] = 1.0 / math.sqrt(count)
        return factors.to(codes.device)

    def average_table_size(self) -> float:
        """Return the mean number of occupied buckets per environment slot.

        Input: This counter.
        Output: Non-negative float diagnostic.
        Mathematical meaning: Estimates the per-episode distinct-state count
            seen by the discretization; useful to detect a degenerate grid
            (sizes near one) or an over-fine one (very large tables).
        """
        if self.num_envs == 0:
            return 0.0
        return sum(len(table) for table in self.tables) / self.num_envs
