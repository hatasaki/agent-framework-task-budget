"""agent_framework_task_budget — a minimal task-budget extension for Microsoft Agent Framework.

Layering (ports & adapters):

* :mod:`agent_framework_task_budget.core` — framework-agnostic budget state (no MAF imports).
* :mod:`agent_framework_task_budget.maf_adapter` — the only module that imports ``agent_framework``.
* :mod:`agent_framework_task_budget.integration` — minimal-change helpers (decorator / one-liners).

The framework-agnostic :class:`~agent_framework_task_budget.core.TaskBudget` is always importable.
The MAF-coupled symbols are exported only when ``agent_framework`` is installed,
so the core can be used and unit-tested without the framework present.
"""

from __future__ import annotations

from .core import TaskBudget

__all__ = ["TaskBudget"]

try:  # MAF-coupled layer; optional at import time.
    from .integration import (
        budget_invocations_host,
        budget_responses_host,
        enable_task_budget,
        extract_budget_enforce,
        extract_budget_tokens,
    )
    from .maf_adapter import (
        BUDGET_ENFORCE_KEY,
        BUDGET_REMAINING_KEY,
        BUDGET_TOKENS_KEY,
        TaskBudgetChatMiddleware,
        TaskBudgetEnforcementMiddleware,
        bind_budget,
        bind_budget_over,
    )
except ImportError:  # pragma: no cover - agent_framework not installed
    pass
else:
    __all__ += [
        "TaskBudgetChatMiddleware",
        "TaskBudgetEnforcementMiddleware",
        "enable_task_budget",
        "extract_budget_tokens",
        "extract_budget_enforce",
        "budget_responses_host",
        "budget_invocations_host",
        "bind_budget",
        "bind_budget_over",
        "BUDGET_TOKENS_KEY",
        "BUDGET_REMAINING_KEY",
        "BUDGET_ENFORCE_KEY",
    ]
