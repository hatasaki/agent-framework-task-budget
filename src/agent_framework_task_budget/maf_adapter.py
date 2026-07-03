"""Microsoft Agent Framework adapter — the **only** module that imports ``agent_framework``.

All coupling to MAF lives here, behind a thin boundary, so that:

* :mod:`agent_framework_task_budget.core` stays framework-agnostic, and
* a MAF upgrade can only ever require changes to this one file.

The real-API touch-points that this adapter deliberately isolates and guards against:

* The chat message class is :class:`agent_framework.Message` (constructed with
  ``contents=[...]``); the streaming flag on :class:`ChatContext` is ``stream``.
* ``call_next`` takes **no** arguments — call it as ``await call_next()``.
* Token usage hangs off ``ChatResponse.usage_details`` and ``UsageDetails`` is a
  ``Mapping`` (``dict`` subclass), so counts are read by key, not attribute.
* A tool result is overridden by *setting* ``FunctionInvocationContext.result`` and
  skipping ``call_next`` — the framework then feeds that value back to the model as
  the tool's result instead of executing the tool (used by the enforcement backstop
  to make the model wrap up gracefully rather than hard-terminating the run).

Each of these touch-points is accessed defensively so that renames or shape changes
in future versions degrade gracefully instead of crashing.
"""

from __future__ import annotations

import contextvars
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from agent_framework import (  # ← the sole MAF dependency surface
    ChatContext,
    ChatMiddleware,
    FunctionInvocationContext,
    FunctionMiddleware,
    Message,
)

from .core import TaskBudget

#: Library logger. Emits a single INFO line when a per-request task budget is applied
#: to a run (see :func:`bind_budget`), so operators can confirm the server detected
#: the budget in the request — no extra middleware needed. No handler is attached
#: here: the host application decides whether/where ``agent_framework_task_budget`` logs go (standard
#: library convention; silent unless this logger is enabled at INFO).
logger = logging.getLogger(__name__)

#: Request field a remote caller sets to choose a run's token budget (e.g. in the
#: Responses ``metadata`` an OpenAI-SDK client sends). The server lifts it from the
#: request; the caller imports nothing from this library — a plain value is enough.
BUDGET_TOKENS_KEY = "task_budget_tokens"
#: Optional request field to seed ``remaining`` separately from ``total`` (e.g. to
#: resume a partially-spent budget). Defaults to ``total`` when omitted.
BUDGET_REMAINING_KEY = "task_budget_remaining"
#: Optional request field a caller sets to arm the enforcement backstop for a run
#: (e.g. ``metadata={"task_budget_enforce": "true"}``). Accepts the string
#: ``"true"``/``"false"`` (case-insensitive) or a real ``bool``; **defaults to off**
#: when omitted, so enforcement is strictly opt-in per request.
BUDGET_ENFORCE_KEY = "task_budget_enforce"

#: The budget in force for the current agent run, seeded per run by
#: :func:`bind_budget` and read by the chat/enforcement middleware. A ``ContextVar``
#: keeps concurrent runs isolated without threading state through every call site.
_current_budget: contextvars.ContextVar["TaskBudget | None"] = contextvars.ContextVar(
    "task_budget_current", default=None
)
#: Whether the enforcement backstop is armed for the current run, seeded per run by
#: :func:`bind_budget` from the caller's ``task_budget_enforce`` request field
#: (default ``False``) and read by :class:`TaskBudgetEnforcementMiddleware`, so each
#: concurrent run decides enforcement independently of the others.
_current_enforce: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "task_budget_enforce_current", default=False
)

_TOTAL_KEYS = ("total_token_count", "total_tokens")
_INPUT_KEYS = ("input_token_count", "input_tokens", "prompt_tokens")
_OUTPUT_KEYS = ("output_token_count", "output_tokens", "completion_tokens")


def _read_count(usage: Any, names: tuple[str, ...]) -> int | None:
    """Read an integer count from a usage object by trying several key/attr names.

    Supports both ``Mapping``-style (``UsageDetails`` is a ``dict`` subclass)
    and attribute-style usage objects, so it survives shape changes across versions.
    """
    for name in names:
        if isinstance(usage, Mapping):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        if isinstance(value, bool):  # guard: bool is an int subclass
            continue
        if isinstance(value, int):
            return value
    return None


def _extract_total_tokens(result: Any) -> int:
    """Best-effort total token count for one model call.

    Looks at ``result.usage_details`` then ``result.usage`` (defensive),
    preferring an explicit total and falling back to ``input + output``.
    Returns ``0`` when usage is unavailable (e.g. providers that omit it).
    """
    usage = getattr(result, "usage_details", None)
    if usage is None:
        usage = getattr(result, "usage", None)
    if usage is None:
        return 0
    total = _read_count(usage, _TOTAL_KEYS)
    if total is not None:
        return total
    in_tok = _read_count(usage, _INPUT_KEYS) or 0
    out_tok = _read_count(usage, _OUTPUT_KEYS) or 0
    return int(in_tok) + int(out_tok)


def _is_streaming(context: Any) -> bool:
    flag = getattr(context, "is_streaming", None)
    if flag is None:
        flag = getattr(context, "stream", False)
    return bool(flag)


@contextmanager
def bind_budget(
    total: int | None, *, min_total: int = 0, enforce: bool = False
) -> Iterator[TaskBudget | None]:
    """Bind a fresh per-run :class:`TaskBudget` for the duration of the block.

    The **synchronous** sibling of :func:`bind_budget_over`, for transports whose
    per-request entry point is a single ``await agent.run(...)`` rather than a
    lazily-iterated event stream — most notably a Foundry **Invocations**
    handler's non-streaming path, whose stock handler reads only ``message`` and
    drops any extra request field. The budget is published on the same
    ``ContextVar`` the chat/enforcement middleware read, and reset on exit.
    ``ContextVar`` values propagate into coroutines awaited inside the same task,
    so wrapping ``with bind_budget(...): await agent.run(...)`` makes the budget
    visible for the whole run. (For a streaming generator, use
    :func:`bind_budget_over`, which keeps the budget bound *while it is iterated*.)

    When ``total`` is ``None`` nothing is bound, so a request that carries no
    budget stays a clean no-op. ``min_total=0`` takes the chosen total at face
    value, since it is an explicit per-request decision. ``enforce=True`` arms the
    enforcement backstop for the block (the transport equivalent of the caller's
    ``task_budget_enforce`` metadata), defaulting to advisory-only.
    """
    if total is None:
        yield None
        return
    budget = TaskBudget(total=total, remaining=None, min_total=min_total)
    # One INFO line per budgeted request, so operators can confirm the server picked
    # up the task budget from the request and is applying it (no middleware needed).
    # Silent unless the host app enables the ``agent_framework_task_budget`` logger at INFO.
    logger.info(
        "task budget applied for this run: total=%d tokens, enforce=%s",
        budget.total,
        enforce,
    )
    token = _current_budget.set(budget)
    enforce_token = _current_enforce.set(enforce)
    try:
        yield budget
    finally:
        _current_enforce.reset(enforce_token)
        _current_budget.reset(token)


async def bind_budget_over(
    events: AsyncIterator[Any],
    total: int | None,
    *,
    min_total: int = 0,
    enforce: bool = False,
) -> AsyncIterator[Any]:
    """Iterate ``events`` with a fresh per-run :class:`TaskBudget` bound.

    This is the escape hatch for transports that **cannot carry run metadata** to
    ``agent.run`` — most notably a Foundry **Responses** host, whose request
    ``metadata`` is dropped before the agent runs (and the streaming path of a
    Foundry **Invocations** handler, whose body iterator runs after the handler
    returns). Wrap the host's per-request event stream with this so the agent's
    chat/enforcement middleware sees the budget for the whole run (the budget is
    published on the same ``ContextVar`` the chat/enforcement middleware read).
    ``agent.run`` happens *while* the stream is iterated, so
    binding here — rather than before the (lazy) stream is created — is what makes
    the budget visible to the run.

    When ``total`` is ``None`` the events pass through untouched, so a request
    that carries no budget stays a clean no-op. ``min_total=0`` takes the chosen
    total at face value, since it is an explicit per-request decision. ``enforce``
    arms the backstop for the run (see :func:`bind_budget`), defaulting to off.
    """
    with bind_budget(total, min_total=min_total, enforce=enforce):
        async for event in events:
            yield event


class TaskBudgetChatMiddleware(ChatMiddleware):
    """Inject the advisory countdown before each model call and account usage after.

    This is the primary mechanism of the task budget: chat middleware runs **per
    model call** (including tool-result calls), which is the natural granularity at
    which to update the countdown.

    The budget may be **fixed** (passed here at construction, for an agent that owns
    one budget) or **per-request** (left as ``None`` here and bound each run by
    :func:`bind_budget` / :func:`bind_budget_over` on the hosting path). When neither
    is present the middleware is a no-op, so it is always safe to leave attached.
    """

    def __init__(self, budget: TaskBudget | None = None) -> None:
        self._budget = budget

    @property
    def budget(self) -> TaskBudget | None:
        return self._budget

    def _active_budget(self) -> TaskBudget | None:
        # A fixed budget wins; otherwise use the current run's budget (if any).
        return self._budget if self._budget is not None else _current_budget.get()

    async def process(
        self,
        context: ChatContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        budget = self._active_budget()
        if budget is None:
            await call_next()  # no budget in force for this run: no-op
            return

        # --- before: inject the remaining-budget countdown as an ephemeral system
        # message. ``context.messages`` is this request's list; the insert affects
        # only this call and is not persisted to the durable conversation history.
        self._inject_status(context, budget)

        # Register streaming accounting up-front. For a streaming call the token
        # usage is only known once the response is *finalized*, so charge it from a
        # stream-result hook the framework runs at finalization — which the
        # function-calling loop does before it processes that call's tool requests,
        # so the enforcement backstop sees an up-to-date budget on the next tool
        # boundary. Non-streaming calls are charged directly after ``call_next``
        # below. A framework without ``stream_result_hooks`` simply falls back to
        # non-streaming accounting (the streamed run stays advisory-only).
        streaming = _is_streaming(context)
        hooks = getattr(context, "stream_result_hooks", None)
        if streaming and isinstance(hooks, list):
            hooks.append(self._make_consume_hook(budget))

        await call_next()  # the actual model call

        # --- after: account the tokens spent (thinking + tools + results + output).
        # Streaming was already hooked above; here we charge the non-streaming result.
        if not streaming:
            budget.consume(_extract_total_tokens(getattr(context, "result", None)))

    def _make_consume_hook(self, budget: TaskBudget) -> Callable[[Any], Any]:
        """Return a stream-result hook that charges the finalized response's usage.

        The framework applies :attr:`ChatContext.stream_result_hooks` to the
        aggregated :class:`~agent_framework.ChatResponse` once the streamed model
        call is finalized — which the function-calling loop does *before* it
        processes that call's tool requests — so the budget is current when the
        enforcement backstop checks it on the next tool boundary. The hook returns
        the response unchanged so it stays a transparent observer in the hook chain.
        """

        def _consume(response: Any) -> Any:
            budget.consume(_extract_total_tokens(response))
            return response

        return _consume

    def _inject_status(self, context: ChatContext, budget: TaskBudget) -> None:
        message = self._status_message(budget)
        messages = getattr(context, "messages", None)
        try:
            messages.insert(0, message)  # ChatContext.messages is a list
        except AttributeError:
            # Defensive fallback if a version exposes an immutable sequence.
            context.messages = [message, *(messages or [])]

    def _status_message(self, budget: TaskBudget) -> Message:
        # The exact Message constructor is confined to this one method on purpose,
        # so a constructor change only touches here.
        return Message(role="system", contents=[budget.render_status()])


#: Default instruction handed to the model *in place of* a tool result once the budget
#: is spent. It steers the model to finish with what it has already gathered (a
#: graceful partial) instead of the empty answer a mid-tool termination would leave.
_BUDGET_EXHAUSTED_WRAPUP = (
    "Task budget exhausted. Do not call this or any other tool again. "
    "Using only the information you have already gathered, write your best final "
    "answer now, and briefly note anything you could not complete."
)


class TaskBudgetEnforcementMiddleware(FunctionMiddleware):
    """Optional backstop: once the budget is spent, stop *new* tool work and steer the
    model to finish with what it already has — rather than hard-killing the run.

    The token budget is purely **advisory** by default — the model is made aware of
    its remaining token budget so it can pace tool use and wrap up before the limit.
    This backstop follows the same principle: it stops the loop **between tool turns**,
    keeping the text the model has already produced; it never truncates a run to an
    empty answer mid-tool.

    So when the budget is exhausted this middleware does **not** execute the requested
    tool. Instead it overrides the call's result — by setting
    :attr:`FunctionInvocationContext.result` and skipping ``call_next`` — with a short
    "budget spent, wrap up now" instruction. The framework feeds that back to the model
    as the tool result, and the model writes a final answer from the data it already
    gathered (a graceful partial), instead of the empty response a mid-tool
    :class:`agent_framework.MiddlewareTermination` would leave behind.

    The advisory countdown (:class:`TaskBudgetChatMiddleware`) still runs alongside, so
    the model is nudged to finish *before* it ever reaches this backstop. If a model
    ignores the wrap-up instruction and keeps calling tools, each call is short-circuited
    the same way and the framework's tool-iteration cap bounds the loop.

    Whether the backstop actually bites is decided **per run**. A middleware given a
    *fixed* budget always enforces once that budget is spent — its owner opted into
    enforcement explicitly. A *metadata-driven* middleware (no fixed budget, the
    ``enable_task_budget`` model) stays advisory-only **unless this run's caller set**
    ``task_budget_enforce`` truthy in the request; the default is off, so it is always
    safe to leave attached.
    """

    def __init__(
        self,
        budget: TaskBudget | None = None,
        *,
        message: str = _BUDGET_EXHAUSTED_WRAPUP,
    ) -> None:
        self._budget = budget
        self._message = message

    @property
    def budget(self) -> TaskBudget | None:
        return self._budget

    def _active_budget(self) -> TaskBudget | None:
        return self._budget if self._budget is not None else _current_budget.get()

    def _enforce_active(self) -> bool:
        # A fixed-budget middleware is only ever attached when the agent owner has
        # explicitly opted into enforcement, so it always bites once spent. A
        # metadata-driven middleware (no fixed budget, e.g. wired via
        # ``enable_task_budget``) bites only when *this run's* caller set the
        # ``task_budget_enforce`` metadata truthy — advisory-only otherwise.
        if self._budget is not None:
            return True
        return _current_enforce.get()

    async def process(
        self,
        context: FunctionInvocationContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        budget = self._active_budget()
        if budget is not None and budget.exhausted and self._enforce_active():
            # Skip the tool and hand the model a wrap-up instruction in its place: set
            # the result and return *without* call_next, so the framework never invokes
            # the tool but still returns a result to the model, which then finishes
            # with what it has (a graceful partial) instead of an empty answer.
            context.result = self._message
            return
        await call_next()
