"""Minimal-change integration helpers.

These are thin ergonomic wrappers that let existing Microsoft Agent Framework code
adopt a task budget by **importing this library and adding a single line**.

The model is :func:`enable_task_budget`: wire the agent once on the server/owner
side so it carries the budget-reading middleware (an advisory countdown plus an
optional enforcement backstop). A remote caller then chooses the budget per request
by adding a plain field to the call it already makes — e.g. an OpenAI-SDK client
sends ``metadata={"agent_framework_task_budget_tokens": "80000"}`` — and **imports nothing from
this library**.

For a MAF agent deployed as a **hosted agent** (a container behind a remote API,
e.g. on Azure AI Foundry), the loop — and therefore this library's middleware —
runs server-side. :func:`budget_responses_host` / :func:`budget_invocations_host`
wrap the hosting server so it lifts the budget out of the incoming request and
binds it across the server-side run; :func:`extract_budget_tokens` /
:func:`extract_budget_enforce` are the underlying server-side helpers.

They are deliberately framework-class-agnostic: agents are treated as duck-typed
objects with a ``middleware`` attribute and an awaitable ``run`` method, so they
keep working even though the concrete agent class name has changed across versions
(e.g. ``ChatAgent`` → ``Agent``). All MAF coupling stays in
:mod:`agent_framework_task_budget.maf_adapter`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

from .maf_adapter import (
    BUDGET_ENFORCE_KEY,
    BUDGET_TOKENS_KEY,
    TaskBudgetChatMiddleware,
    TaskBudgetEnforcementMiddleware,
    bind_budget,
    bind_budget_over,
)

AgentT = TypeVar("AgentT")


def _merge_middleware(existing: Any, extra: Sequence[Any]) -> list[Any]:
    # The ``middleware`` argument may be a single object, a list, or None.
    if existing is None:
        return list(extra)
    if isinstance(existing, list):
        return [*existing, *extra]
    return [existing, *extra]


def _set_agent_middleware(agent: Any, merged: list[Any]) -> None:
    # ``agent.middleware`` is the public registration list. Some restricted agent
    # instances forbid normal attribute assignment; fall back to ``__setattr__``.
    try:
        agent.middleware = merged
    except (AttributeError, ValueError, TypeError):
        object.__setattr__(agent, "middleware", merged)


def enable_task_budget(agent: AgentT) -> AgentT:
    """Wire an agent (once) to honour a **per-request** budget from the caller.

    Call this once on the agent/server side. It attaches the budget-reading
    middleware — an advisory remaining-token countdown injected before every model
    call, plus an enforcement backstop that stays inert unless a run opts in — so
    that a budget bound for a run (by :func:`budget_responses_host` /
    :func:`budget_invocations_host`, which lift it from the incoming request) is
    actually read and applied. A remote caller sets the budget per request without
    importing anything from this library — e.g. an OpenAI-SDK client sends::

        client.responses.create(
            model="my-agent",
            input="Investigate X...",
            metadata={"agent_framework_task_budget_tokens": "80000"},
        )

    A fresh budget is created per run and isolated across concurrent runs (via a
    ``ContextVar``).

    **Enforcement is chosen by the caller, per request** (default advisory-only).
    By default the budget is purely advisory — a remaining-token countdown that lets
    the model pace itself. A caller that wants a hard backstop adds
    ``"agent_framework_task_budget_enforce": "true"`` to the same request; once that run's budget is
    spent, further tool calls are skipped and the model is steered to wrap up with
    what it already gathered (a graceful partial) instead of erroring out::

        metadata={"agent_framework_task_budget_tokens": "80000", "agent_framework_task_budget_enforce": "true"}

    Returns the same agent so the call can be chained. The request fields are
    ``"agent_framework_task_budget_tokens"`` and ``"agent_framework_task_budget_enforce"``.
    """
    extra: list[Any] = [
        TaskBudgetChatMiddleware(),
        # Always attached, but inert unless this run's caller opts into enforcement
        # via the ``agent_framework_task_budget_enforce`` request field (it carries no fixed budget,
        # so it defers to the per-run flag). Default runs stay purely advisory.
        TaskBudgetEnforcementMiddleware(),
    ]
    _set_agent_middleware(agent, _merge_middleware(getattr(agent, "middleware", None), extra))
    return agent


# --------------------------------------------------------------------------- #
# Server-side hosting (e.g. Foundry hosted agents)
# --------------------------------------------------------------------------- #
# When a MAF agent is deployed as a *hosted agent* (a container behind a remote
# API), the loop — and therefore this library's middleware — runs server-side.
# A remote caller cannot attach middleware; it can only put a value in the
# request. ``budget_responses_host`` / ``budget_invocations_host`` wrap the hosting
# server so it lifts the budget out of the incoming request and binds it across the
# server-side run; ``extract_budget_tokens`` / ``extract_budget_enforce`` are the
# underlying helpers. The caller imports nothing — it adds a plain field such as
# ``metadata={"agent_framework_task_budget_tokens": "80000"}`` to the call it already makes.

#: Request-body field names a remote caller may use to carry the budget. These
#: are plain JSON keys — the caller imports nothing from this library.
BUDGET_REQUEST_KEYS: tuple[str, ...] = (BUDGET_TOKENS_KEY, "agent_framework_task_budget")
#: Sub-objects of the request body that are also searched for the budget keys,
#: so OpenAI-SDK callers may nest the value under ``metadata`` or ``extra_body``.
BUDGET_REQUEST_CONTAINERS: tuple[str, ...] = ("metadata", "extra_body")


def _coerce_pos_int(value: Any) -> int | None:
    # Accept a positive int, or a digit string (OpenAI ``metadata`` values are
    # transmitted as strings), and reject bool (an int subclass).
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            number = int(stripped)
            return number if number > 0 else None
    return None


def _first_pos_int(mapping: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    for key in keys:
        number = _coerce_pos_int(mapping.get(key))
        if number is not None:
            return number
    return None


_ENFORCE_TRUTHY = frozenset({"true", "1", "yes", "on"})


def _coerce_flag(value: Any) -> bool:
    # Accept a real bool or the truthy strings a JSON body / metadata field may
    # carry ("true"/"1"/"yes"/"on", case-insensitive); everything else is False.
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _ENFORCE_TRUTHY
    return False


def extract_budget_tokens(
    payload: Any,
    *,
    keys: Sequence[str] = BUDGET_REQUEST_KEYS,
    containers: Sequence[str] = BUDGET_REQUEST_CONTAINERS,
) -> int | None:
    """Pull a positive token budget out of an incoming request body.

    Designed for the **server side** of a remotely-called agent (for example a
    Foundry *hosted agent* invocations handler): the remote caller passes the
    budget as a plain field in the JSON body — ``{"message": "...",
    "agent_framework_task_budget_tokens": 80000}`` — and imports nothing from this library.

    The value is looked up first at the top level under each name in ``keys``,
    then inside each sub-object named in ``containers`` (so callers may nest it
    under ``metadata`` or ``extra_body``). Digit strings are accepted because
    OpenAI ``metadata`` values are transmitted as strings. Returns ``None`` when
    no usable budget is present.
    """
    if not isinstance(payload, Mapping):
        return None
    found = _first_pos_int(payload, keys)
    if found is not None:
        return found
    for container in containers:
        sub = payload.get(container)
        if isinstance(sub, Mapping):
            found = _first_pos_int(sub, keys)
            if found is not None:
                return found
    return None


def extract_budget_enforce(
    payload: Any,
    *,
    containers: Sequence[str] = BUDGET_REQUEST_CONTAINERS,
) -> bool:
    """Pull the optional enforcement flag out of an incoming request body.

    The **server-side** companion to :func:`extract_budget_tokens` for the
    ``agent_framework_task_budget_enforce`` flag: a remote caller adds one plain field
    (``{"agent_framework_task_budget_enforce": "true"}``), optionally nested under ``metadata`` or
    ``extra_body``, and imports nothing from this library. The flag is looked up
    first at the top level, then inside each ``containers`` sub-object; a real
    ``bool`` and the truthy strings (``"true"``/``"1"``/``"yes"``/``"on"``) are
    accepted. A missing or unrecognized value returns ``False`` so enforcement is
    strictly opt-in.
    """
    if not isinstance(payload, Mapping):
        return False
    if _coerce_flag(payload.get(BUDGET_ENFORCE_KEY)):
        return True
    for container in containers:
        sub = payload.get(container)
        if isinstance(sub, Mapping) and _coerce_flag(sub.get(BUDGET_ENFORCE_KEY)):
            return True
    return False


def _responses_request_metadata(request: Any) -> dict[str, Any]:
    # OpenAI Responses ``metadata`` is a ``{str: str}`` map on the request model.
    # Read it defensively so a shape change degrades to "no budget" rather than
    # crashing the host.
    meta = getattr(request, "metadata", None)
    if isinstance(meta, Mapping):
        return dict(meta)
    return {}


#: Middleware classes whose presence means an agent is already budget-enabled.
_BUDGET_MIDDLEWARE_TYPES = (
    TaskBudgetChatMiddleware,
    TaskBudgetEnforcementMiddleware,
)


def _ensure_budget_enabled(agent: Any) -> None:
    """Guarantee the agent carries the budget-reading middleware, wiring it if not.

    The host factories only **bind** the per-request budget (via
    :func:`bind_budget_over` / :func:`bind_budget`); the middleware that actually
    **reads** it — the advisory countdown and the enforcement backstop — is
    attached by :func:`enable_task_budget`. Hosting an agent that was never wired
    would bind the budget to a value nothing consumes, so the budget silently has
    no effect (exactly the "``agent_framework_task_budget_tokens`` changes nothing" symptom).

    To make the one-liner ``budget_responses_host(agent)`` correct on its own,
    this wires the agent when it is not already budget-enabled. It is idempotent:
    an agent that already carries the budget middleware (e.g. because the caller
    ran :func:`enable_task_budget` explicitly) is left untouched, so there is no
    duplicate countdown. The host's own ``default_total`` remains the server-side
    fallback applied when a request omits the budget.
    """
    existing = getattr(agent, "middleware", None)
    if existing is None:
        current: list[Any] = []
    elif isinstance(existing, list):
        current = existing
    else:
        current = [existing]
    if any(isinstance(m, _BUDGET_MIDDLEWARE_TYPES) for m in current):
        return  # already budget-enabled; do not double-wire
    enable_task_budget(agent)


def budget_responses_host(agent: Any, *, default_total: int | None = None, **kwargs: Any) -> Any:
    """Build a Foundry **Responses** host that honours a per-request task budget.

    The OpenAI-compatible Responses host normally **drops** the request's
    ``metadata``/``extra_body`` before running the agent, so an OpenAI-SDK client
    that sends ``metadata={"agent_framework_task_budget_tokens": "80000"}`` would otherwise have
    its budget ignored. This factory returns a thin ``ResponsesHostServer``
    subclass that lifts the budget out of ``request.metadata`` and binds it across
    the run via :func:`bind_budget_over`, so a client can drive the agent with the
    plain OpenAI SDK and still govern the server-side loop::

        # client (imports only the OpenAI SDK):
        client.responses.create(
            model="my-agent",
            input="Investigate the outage",
            metadata={"agent_framework_task_budget_tokens": "80000"},
        )

    Wire the agent once with :func:`enable_task_budget` before hosting it; pass any
    server-side fallback via ``default_total`` here. Requires the
    ``agent-framework-foundry-hosting`` package; it is imported lazily so the rest
    of this library has no hard dependency on it.

    Note: this overrides the host's ``_handle_response`` and reads
    ``request.metadata``. The stock host drops ``request.metadata`` when building
    the run options, so the budget must be lifted here. It is the one spot coupled
    to Responses-host internals; pin the hosting version or re-check on upgrade.
    """
    _ensure_budget_enabled(agent)  # make the budget actually take effect if unwired
    from agent_framework_foundry_hosting import ResponsesHostServer  # lazy, optional dep

    class _BudgetResponsesHostServer(ResponsesHostServer):  # type: ignore[misc, valid-type]
        async def _handle_response(self, request: Any, context: Any, cancellation_signal: Any) -> Any:
            events = await super()._handle_response(request, context, cancellation_signal)
            metadata = _responses_request_metadata(request)
            total = extract_budget_tokens(metadata)
            if total is None:
                total = default_total
            enforce = extract_budget_enforce(metadata)
            return bind_budget_over(events, total, enforce=enforce)

    return _BudgetResponsesHostServer(agent, **kwargs)


def budget_invocations_host(agent: Any, *, default_total: int | None = None, **kwargs: Any) -> Any:
    """Build a Foundry **Invocations** host that honours a per-request task budget.

    The Invocations host expects a JSON body with a ``"message"`` field and runs the
    agent itself. Its stock ``_handle_invoke`` reads only ``"message"``/``"stream"``
    and **drops** any extra field, so a caller's ``agent_framework_task_budget_tokens`` in the body
    would otherwise be ignored. This factory returns a thin ``InvocationsHostServer``
    subclass that lifts the budget out of the request body and binds it across the
    run, so a remote caller can govern the server-side loop by adding one plain
    field to the JSON it already sends::

        # client (imports nothing from this library):
        {"message": "Investigate the outage", "agent_framework_task_budget_tokens": 80000}

    Wire the agent once with :func:`enable_task_budget` before hosting it; pass any
    server-side fallback via ``default_total`` here. Requires the
    ``agent-framework-foundry-hosting`` package; it is imported lazily so the rest
    of this library has no hard dependency on it.

    Both run paths are covered: the non-streaming ``await agent.run(...)`` is wrapped
    with :func:`bind_budget`, while the streaming ``StreamingResponse`` body iterator
    (which runs *after* the handler returns) is wrapped with :func:`bind_budget_over`
    so the budget stays bound while the response streams.

    Note: this overrides the host's ``_handle_invoke`` and reads the JSON body.
    The stock handler reads only ``message``/``stream`` and drops any extra field,
    so the budget must be lifted here. It is the one spot coupled to
    Invocations-host internals; pin the hosting version or re-check on upgrade.
    """
    _ensure_budget_enabled(agent)  # make the budget actually take effect if unwired
    from agent_framework_foundry_hosting import InvocationsHostServer  # lazy, optional dep

    class _BudgetInvocationsHostServer(InvocationsHostServer):  # type: ignore[misc, valid-type]
        async def _handle_invoke(self, request: Any) -> Any:
            data = await request.json()  # Starlette caches the body, so super() can re-read it
            total = extract_budget_tokens(data)
            if total is None:
                total = default_total
            if total is None:
                return await super()._handle_invoke(request)  # no budget → stay out of the way
            enforce = extract_budget_enforce(data)
            if data.get("stream", False):
                # The streaming body iterator runs AFTER this handler returns, so the
                # budget must be bound across the iterator, not just this call.
                response = await super()._handle_invoke(request)
                inner = getattr(response, "body_iterator", None)
                if inner is not None:
                    response.body_iterator = bind_budget_over(inner, total, enforce=enforce)
                return response
            with bind_budget(total, enforce=enforce):  # non-streaming: the whole run happens here
                return await super()._handle_invoke(request)

    return _BudgetInvocationsHostServer(agent, **kwargs)
