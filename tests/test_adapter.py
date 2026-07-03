"""Tests for the MAF adapter (token extraction + middleware behaviour).

These exercise the real ``agent_framework`` classes where it is cheap and
informative (``Message``, ``ChatContext``, ``ChatResponse``, ``UsageDetails``),
and use small duck-typed fakes where constructing a real object would add noise.
The adapter itself is duck-typed by design, so both styles are valid.
"""

from types import SimpleNamespace

import pytest
from agent_framework import (
    ChatContext,
    ChatResponse,
    Message,
    UsageDetails,
)

from agent_framework_task_budget import (
    TaskBudget,
    TaskBudgetChatMiddleware,
    TaskBudgetEnforcementMiddleware,
    bind_budget,
    bind_budget_over,
)
from agent_framework_task_budget.maf_adapter import (
    _current_budget,
    _current_enforce,
    _extract_total_tokens,
    _is_streaming,
)


# --------------------------------------------------------------------------- #
# Token extraction
# --------------------------------------------------------------------------- #
def test_extract_prefers_explicit_total():
    usage = UsageDetails(input_token_count=100, output_token_count=40, total_token_count=140)
    result = ChatResponse(messages=[Message("assistant", ["ok"])], usage_details=usage)
    assert _extract_total_tokens(result) == 140


def test_extract_falls_back_to_input_plus_output():
    usage = UsageDetails(input_token_count=100, output_token_count=40)
    result = ChatResponse(messages=[Message("assistant", ["ok"])], usage_details=usage)
    assert _extract_total_tokens(result) == 140


def test_extract_returns_zero_when_no_result():
    assert _extract_total_tokens(None) == 0


def test_extract_returns_zero_when_usage_missing():
    result = ChatResponse(messages=[Message("assistant", ["ok"])])
    assert _extract_total_tokens(result) == 0


def test_extract_supports_attribute_style_usage():
    class AttrUsage:
        total_token_count = 77

    class AttrResult:
        usage_details = AttrUsage()

    assert _extract_total_tokens(AttrResult()) == 77


# --------------------------------------------------------------------------- #
# Streaming detection
# --------------------------------------------------------------------------- #
def test_is_streaming_reads_stream_flag():
    class Ctx:
        stream = True

    assert _is_streaming(Ctx()) is True


def test_is_streaming_defaults_false():
    assert _is_streaming(object()) is False


# --------------------------------------------------------------------------- #
# Chat middleware
# --------------------------------------------------------------------------- #
class FakeChatContext:
    """Minimal duck-typed stand-in for ChatContext."""

    def __init__(self, messages=None, stream=False, with_stream_hooks=False):
        self.messages = list(messages or [])
        self.stream = stream
        self.result = None
        if with_stream_hooks:
            self.stream_result_hooks: list = []


async def test_chat_middleware_injects_countdown_and_accounts():
    budget = TaskBudget(total=50_000)
    mw = TaskBudgetChatMiddleware(budget)
    ctx = FakeChatContext(messages=[Message("user", ["hi"])])

    async def call_next():
        # countdown must be injected *before* the model call
        assert ctx.messages[0].role == "system"
        assert "Task budget" in ctx.messages[0].text
        usage = UsageDetails(input_token_count=100, output_token_count=40, total_token_count=140)
        ctx.result = ChatResponse(messages=[Message("assistant", ["ok"])], usage_details=usage)

    await mw.process(ctx, call_next)

    assert budget.remaining == 50_000 - 140
    assert ctx.messages[0].role == "system"  # ephemeral system message present


async def test_chat_middleware_streaming_registers_hook_and_charges_on_finalize():
    # With stream_result_hooks available, streaming defers accounting to a hook the
    # framework runs on the finalized response — so the budget is still charged.
    budget = TaskBudget(total=50_000)
    mw = TaskBudgetChatMiddleware(budget)
    ctx = FakeChatContext(stream=True, with_stream_hooks=True)

    async def call_next():
        ctx.result = None  # streaming: the real result would be a ResponseStream

    await mw.process(ctx, call_next)

    assert budget.remaining == 50_000  # not charged yet — deferred to the hook
    assert len(ctx.stream_result_hooks) == 1

    # Simulate the framework finalizing the streamed response:
    resp = ChatResponse(
        messages=[Message("assistant", ["ok"])],
        usage_details=UsageDetails(total_token_count=140),
    )
    returned = ctx.stream_result_hooks[0](resp)
    assert returned is resp  # transparent observer in the hook chain
    assert budget.remaining == 50_000 - 140  # charged at finalization


async def test_chat_middleware_streaming_without_hooks_degrades_to_advisory_only():
    # An older framework without stream_result_hooks: streaming stays advisory-only
    # (no accounting) — the safe fallback, never a crash.
    budget = TaskBudget(total=50_000)
    mw = TaskBudgetChatMiddleware(budget)
    ctx = FakeChatContext(stream=True)  # no stream_result_hooks attribute

    async def call_next():
        usage = UsageDetails(total_token_count=999)
        ctx.result = ChatResponse(messages=[Message("assistant", ["ok"])], usage_details=usage)

    await mw.process(ctx, call_next)

    assert budget.remaining == 50_000  # unchanged: no hook available to defer to


async def test_chat_middleware_with_real_chat_context():
    budget = TaskBudget(total=50_000)
    mw = TaskBudgetChatMiddleware(budget)
    ctx = ChatContext(client=None, messages=[Message("user", ["hi"])], options=None)

    async def call_next():
        usage = UsageDetails(total_token_count=200)
        ctx.result = ChatResponse(messages=[Message("assistant", ["ok"])], usage_details=usage)

    await mw.process(ctx, call_next)

    assert budget.remaining == 50_000 - 200
    assert ctx.messages[0].role == "system"


async def test_chat_middleware_streaming_hook_on_real_chat_context():
    # End-to-end on the *real* ChatContext: streaming injects the countdown and
    # registers a stream-result hook that charges the finalized response's usage.
    budget = TaskBudget(total=50_000)
    mw = TaskBudgetChatMiddleware(budget)
    ctx = ChatContext(client=None, messages=[Message("user", ["hi"])], options=None, stream=True)

    async def call_next():
        ctx.result = None  # streaming path; framework would set a ResponseStream

    await mw.process(ctx, call_next)

    assert ctx.messages[0].role == "system"  # countdown injected even in streaming
    assert budget.remaining == 50_000  # accounting deferred to the hook
    assert len(ctx.stream_result_hooks) == 1

    resp = ChatResponse(
        messages=[Message("assistant", ["ok"])],
        usage_details=UsageDetails(total_token_count=321),
    )
    assert ctx.stream_result_hooks[0](resp) is resp
    assert budget.remaining == 50_000 - 321  # charged when the stream finalizes


# --------------------------------------------------------------------------- #
# Enforcement middleware
# --------------------------------------------------------------------------- #
async def test_enforcement_wraps_up_when_exhausted():
    budget = TaskBudget(total=20_000)
    budget.consume(20_000)
    assert budget.exhausted
    mw = TaskBudgetEnforcementMiddleware(budget, message="WRAP UP NOW")

    called = False

    async def call_next():
        nonlocal called
        called = True

    ctx = SimpleNamespace(result=None)
    await mw.process(ctx, call_next)

    # The tool is skipped and the model is handed a wrap-up instruction in its
    # place, so it finishes with what it has instead of terminating empty.
    assert called is False
    assert ctx.result == "WRAP UP NOW"


async def test_enforcement_default_message_steers_model_to_finish():
    budget = TaskBudget(total=20_000)
    budget.consume(20_000)
    mw = TaskBudgetEnforcementMiddleware(budget)

    ctx = SimpleNamespace(result=None)
    await mw.process(ctx, _noop)

    assert isinstance(ctx.result, str)
    low = ctx.result.lower()
    assert "budget" in low and "final answer" in low


async def test_enforcement_passes_when_budget_remains():
    budget = TaskBudget(total=20_000)
    mw = TaskBudgetEnforcementMiddleware(budget)

    called = False

    async def call_next():
        nonlocal called
        called = True

    ctx = SimpleNamespace(result=None)
    await mw.process(ctx, call_next)
    assert called is True
    assert ctx.result is None  # not overridden while budget remains


# --------------------------------------------------------------------------- #
# Per-run ContextVar budget — bound by the host shims (bind_budget / bind_budget_over)
# and read by the chat/enforcement middleware.
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _reset_current_budget():
    """Guarantee the per-run ContextVars never leak between tests."""
    budget_token = _current_budget.set(None)
    enforce_token = _current_enforce.set(False)
    try:
        yield
    finally:
        _current_enforce.reset(enforce_token)
        _current_budget.reset(budget_token)


async def _noop():
    return None


async def test_chat_middleware_uses_contextvar_budget_when_unbound():
    budget = TaskBudget(total=50_000)
    _current_budget.set(budget)
    mw = TaskBudgetChatMiddleware()  # no fixed budget => read the current run's
    ctx = FakeChatContext(messages=[Message("user", ["hi"])])

    async def call_next():
        assert ctx.messages[0].role == "system"  # countdown injected
        ctx.result = ChatResponse(
            messages=[Message("assistant", ["ok"])],
            usage_details=UsageDetails(total_token_count=300),
        )

    await mw.process(ctx, call_next)

    assert budget.remaining == 50_000 - 300


async def test_chat_middleware_is_noop_without_any_budget():
    mw = TaskBudgetChatMiddleware()  # no fixed budget, no ContextVar budget
    ctx = FakeChatContext(messages=[Message("user", ["hi"])])
    ran = False

    async def call_next():
        nonlocal ran
        ran = True

    await mw.process(ctx, call_next)

    assert ran is True
    assert all(getattr(m, "role", None) != "system" for m in ctx.messages)


async def test_enforcement_uses_contextvar_budget_when_unbound():
    budget = TaskBudget(total=20_000)
    budget.consume(20_000)
    _current_budget.set(budget)
    _current_enforce.set(True)  # caller opted in via task_budget_enforce metadata
    mw = TaskBudgetEnforcementMiddleware()  # no fixed budget

    ctx = SimpleNamespace(result=None)
    await mw.process(ctx, _noop)
    assert ctx.result is not None  # wrap-up injected from the contextvar budget


async def test_enforcement_metadata_driven_is_advisory_when_flag_off():
    # No fixed budget + the per-run enforce flag left False (the default): even an
    # exhausted budget must NOT short-circuit the tool. This is the enable_task_budget
    # default — advisory-only unless the caller sets task_budget_enforce truthy.
    budget = TaskBudget(total=20_000)
    budget.consume(20_000)
    _current_budget.set(budget)
    _current_enforce.set(False)
    mw = TaskBudgetEnforcementMiddleware()  # metadata-driven

    called = False

    async def call_next():
        nonlocal called
        called = True

    ctx = SimpleNamespace(result=None)
    await mw.process(ctx, call_next)
    assert called is True  # tool ran: no enforcement without the opt-in flag
    assert ctx.result is None


async def test_enforcement_fixed_budget_always_enforces_regardless_of_flag():
    # A fixed budget means the owner opted into enforcement explicitly, so it bites
    # once spent even though the per-run flag is off.
    budget = TaskBudget(total=20_000)
    budget.consume(20_000)
    _current_enforce.set(False)
    mw = TaskBudgetEnforcementMiddleware(budget)  # fixed budget

    ctx = SimpleNamespace(result=None)
    await mw.process(ctx, _noop)
    assert ctx.result is not None  # enforced despite the flag being off


async def test_enforcement_is_noop_without_any_budget():
    mw = TaskBudgetEnforcementMiddleware()
    called = False

    async def call_next():
        nonlocal called
        called = True

    await mw.process(object(), call_next)

    assert called is True


# --------------------------------------------------------------------------- #
# bind_budget_over — bind a per-run budget while a host event stream is iterated
# (escape hatch for transports that drop run metadata, e.g. a Responses host)
# --------------------------------------------------------------------------- #
async def _events_reading_budget(seen, items=("a", "b")):
    """Fake host event stream that records the bound budget as it is iterated."""
    for item in items:
        seen.append(_current_budget.get())
        yield item


async def test_bind_budget_over_binds_during_iteration():
    seen = []
    out = [ev async for ev in bind_budget_over(_events_reading_budget(seen), 80_000)]

    assert out == ["a", "b"]
    assert [b.total for b in seen] == [80_000, 80_000]  # budget visible mid-stream
    assert _current_budget.get() is None  # cleared after the stream is exhausted


async def test_bind_budget_over_is_noop_when_total_none():
    seen = []
    out = [ev async for ev in bind_budget_over(_events_reading_budget(seen), None)]

    assert out == ["a", "b"]
    assert seen == [None, None]  # nothing bound
    assert _current_budget.get() is None


async def test_bind_budget_over_resets_on_error():
    async def boom():
        yield "a"
        raise RuntimeError("stream failed")

    with pytest.raises(RuntimeError, match="stream failed"):
        async for _ in bind_budget_over(boom(), 80_000):
            pass

    assert _current_budget.get() is None  # cleared even when the stream errors


async def test_bind_budget_over_takes_small_total_at_face_value():
    seen = []
    async for _ in bind_budget_over(_events_reading_budget(seen, items=("x",)), 500):
        pass

    assert seen[0].total == 500  # min_total=0 => no floor, no raise


# --------------------------------------------------------------------------- #
# bind_budget — synchronous sibling that binds across a single ``await agent.run``
# (escape hatch for an Invocations handler's non-streaming path)
# --------------------------------------------------------------------------- #
async def test_bind_budget_binds_during_awaited_run():
    seen = {}

    async def fake_run():
        seen["total"] = _current_budget.get().total

    with bind_budget(80_000) as budget:
        assert budget.total == 80_000
        assert _current_budget.get() is budget  # visible synchronously…
        await fake_run()  # …and across the awaited run in the same task

    assert seen["total"] == 80_000
    assert _current_budget.get() is None  # cleared after the block


def test_bind_budget_is_noop_when_total_none():
    with bind_budget(None) as budget:
        assert budget is None  # nothing bound
        assert _current_budget.get() is None
    assert _current_budget.get() is None


def test_bind_budget_resets_on_error():
    with pytest.raises(RuntimeError, match="boom"):
        with bind_budget(80_000):
            assert _current_budget.get().total == 80_000
            raise RuntimeError("boom")

    assert _current_budget.get() is None  # cleared even when the block errors


def test_bind_budget_takes_small_total_at_face_value():
    with bind_budget(500) as budget:
        assert budget.total == 500  # min_total=0 => no floor, no raise


def test_bind_budget_logs_info_when_applied(caplog):
    # A budgeted run emits ONE INFO line so operators can confirm the server picked
    # up the task budget from the request without adding any middleware.
    import logging

    with caplog.at_level(logging.INFO, logger="agent_framework_task_budget"):
        with bind_budget(1234, enforce=True):
            pass

    applied = [r.getMessage() for r in caplog.records if "task budget applied" in r.getMessage()]
    assert len(applied) == 1
    assert "1234" in applied[0]
    assert "enforce=True" in applied[0]


def test_bind_budget_does_not_log_when_total_none(caplog):
    # A request without a budget is a clean no-op — and stays quiet in the log.
    import logging

    with caplog.at_level(logging.INFO, logger="agent_framework_task_budget"):
        with bind_budget(None):
            pass

    assert not [r for r in caplog.records if "task budget applied" in r.getMessage()]
