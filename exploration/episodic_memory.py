"""Top-K episodic memory for Adventurer's resettable premise.

Algorithm 2 maintains one episodic memory per epoch containing the K visited
states with the highest novelty score. This module stores both observations and
optional environment snapshots so a compatible environment can restore the
actual simulator state at the beginning of a later episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import torch
from torch import Tensor


@dataclass(frozen=True)
class EpisodicMemoryEntry:
    """A novel state candidate retained by episodic memory."""

    observation: Tensor
    novelty_score: float
    environment_snapshot: Any = None


class EpisodicMemory:
    """Maintain current and previous epoch top-K novel states."""

    def __init__(self, capacity: int) -> None:
        """Initialize empty previous/current memories.

        Input: Positive top-K memory capacity.
        Output: Empty episodic-memory manager.
        Mathematical meaning: Defines the finite set size K in Algorithm 2.
        """
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.previous: List[EpisodicMemoryEntry] = []
        self.current: List[EpisodicMemoryEntry] = []
        self.epoch = 0

    def begin_epoch(self) -> None:
        """Move current memory to previous and start an empty current memory.

        Input: Current memory manager at an epoch boundary.
        Output: No value; ``previous=M_(l-1)`` and ``current=M_l`` are set.
        Mathematical meaning: Implements the two-memory optimization described
            in Appendix A so episode starts sample only the last epoch's states.
        """
        self.previous = list(self.current)
        self.current = []
        self.epoch += 1

    def add(
        self,
        observation: Tensor,
        novelty_score: float,
        environment_snapshot: Any = None,
    ) -> bool:
        """Insert a state if it belongs to the current top-K novelty set.

        Input: Observation, scalar novelty score B(s), and optional simulator
            snapshot capable of later restoration.
        Output: ``True`` if inserted/replaced, otherwise ``False``.
        Mathematical meaning: Updates ``M_l`` when a visited state has novelty
            higher than the current K-th highest stored state.
        """
        if observation.ndim < 1:
            raise ValueError("observation must have at least one dimension")
        if not torch.isfinite(torch.as_tensor(novelty_score)).item():
            raise ValueError("novelty_score must be finite")
        entry = EpisodicMemoryEntry(
            observation=observation.detach().cpu().clone(),
            novelty_score=float(novelty_score),
            environment_snapshot=environment_snapshot,
        )
        if len(self.current) < self.capacity:
            self.current.append(entry)
            self.current.sort(key=lambda item: item.novelty_score, reverse=True)
            return True
        minimum_index = min(
            range(len(self.current)),
            key=lambda index: self.current[index].novelty_score,
        )
        if entry.novelty_score <= self.current[minimum_index].novelty_score:
            return False
        self.current[minimum_index] = entry
        self.current.sort(key=lambda item: item.novelty_score, reverse=True)
        return True

    def sample_previous(
        self,
        generator: Optional[torch.Generator] = None,
    ) -> Optional[EpisodicMemoryEntry]:
        """Sample one previous-epoch novel state uniformly.

        Input: Optional seeded PyTorch generator.
        Output: One previous-memory entry, or ``None`` when no prior memory
            exists, as in epoch zero.
        Mathematical meaning: Implements Algorithm 2's sampling of an initial
            state from ``M_(l-1)`` under the resettable premise.
        """
        if not self.previous:
            return None
        index = torch.randint(
            low=0,
            high=len(self.previous),
            size=(1,),
            generator=generator,
        ).item()
        return self.previous[index]

    def __len__(self) -> int:
        """Return the number of current-epoch memory entries.

        Input: This memory manager.
        Output: Current memory size in ``[0,K]``.
        Mathematical meaning: Gives the support size of ``M_l``.
        """
        return len(self.current)

    def state_dict(self) -> dict[str, Any]:
        """Return a checkpoint representation of both episodic memories.

        Input: This memory manager.
        Output: Dictionary containing capacity, epoch, previous entries, and
            current entries.
        Mathematical meaning: Preserves the resettable exploration frontier for
            exact checkpoint/resume behavior.
        """
        return {
            "capacity": self.capacity,
            "epoch": self.epoch,
            "previous": list(self.previous),
            "current": list(self.current),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore compatible episodic-memory state.

        Input: State produced by ``state_dict`` with matching capacity.
        Output: No value; previous/current memory and epoch are restored.
        Mathematical meaning: Resumes Algorithm 2 from the same novel-state
            frontier.
        """
        if int(state.get("capacity", -1)) != self.capacity:
            raise ValueError("episodic-memory capacity does not match")
        previous = state.get("previous")
        current = state.get("current")
        if not isinstance(previous, list) or not isinstance(current, list):
            raise TypeError("invalid episodic-memory state")
        if len(previous) > self.capacity or len(current) > self.capacity:
            raise ValueError("episodic-memory state exceeds capacity")
        self.previous = list(previous)
        self.current = list(current)
        self.epoch = int(state.get("epoch", 0))
