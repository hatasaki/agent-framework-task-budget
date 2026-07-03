"""Framework-agnostic task budget state machine.

This module intentionally has **no dependency on Microsoft Agent Framework**
(or any agent library). It models a token budget for a whole task plus an advisory
countdown, using only the standard library, so it is immune to upstream framework
changes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TaskBudget:
    """A task-wide token budget with an advisory countdown.

    Models a ``total`` / ``remaining`` token budget. Responsibilities are limited
    to: consuming tokens, rendering the advisory countdown text shown to the model,
    and reporting exhaustion. It never imports or references any agent framework, so
    framework upgrades cannot break it.

    Args:
        total: Total token budget for the entire task.
        remaining: Remaining tokens. Defaults to ``total`` when omitted.
        min_total: Lower bound enforced on ``total`` (default floor is 20,000).
            Set to ``0`` to disable the check.
    """

    total: int
    remaining: int | None = None
    min_total: int = 20_000  # Sanity floor to reject accidentally-tiny budgets; set 0 to disable.

    def __post_init__(self) -> None:
        if self.min_total and self.total < self.min_total:
            raise ValueError(
                f"task budget 'total' must be >= {self.min_total} (got {self.total})"
            )
        if self.remaining is None:
            self.remaining = self.total

    def consume(self, tokens: int) -> None:
        """Subtract the tokens spent by one model call from the remaining budget."""
        if tokens and tokens > 0:
            self.remaining = max(0, (self.remaining or 0) - int(tokens))

    @property
    def exhausted(self) -> bool:
        """True once the remaining budget reaches zero."""
        return (self.remaining or 0) <= 0

    @property
    def fraction_left(self) -> float:
        """Remaining budget as a fraction of total, in ``[0.0, 1.0]``."""
        return 0.0 if not self.total else (self.remaining or 0) / self.total

    def render_status(self) -> str:
        """Render the advisory countdown the model sees before each model call.

        Implemented as a client-side system message injected before each call.
        """
        return (
            "## Task budget (advisory)\n"
            f"You have about {self.remaining:,} of {self.total:,} tokens left for this "
            "entire task — counting your thinking, tool calls, tool results and final "
            "output. Pace yourself, do the most important work first, and wrap up "
            "gracefully before the budget runs out. This is guidance, not a hard limit."
        )

    def snapshot(self) -> dict[str, int]:
        """Serializable view of the budget for persistence across compaction/sessions."""
        return {"total": self.total, "remaining": int(self.remaining or 0)}

    @classmethod
    def restore(cls, data: dict[str, int], *, min_total: int = 20_000) -> "TaskBudget":
        """Rebuild a budget from a :meth:`snapshot` (e.g. after compaction)."""
        return cls(total=data["total"], remaining=data.get("remaining"), min_total=min_total)
