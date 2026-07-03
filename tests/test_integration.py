"""Tests for the duck-typed integration helpers.

The integration layer never imports a concrete agent class; it treats agents as
objects with a ``middleware`` attribute and an awaitable ``run`` method. These
tests use a small ``FakeAgent`` to prove the wiring without any real model calls.
"""

import pytest

from agent_framework_task_budget import (
    TaskBudgetChatMiddleware,
    TaskBudgetEnforcementMiddleware,
    enable_task_budget,
    extract_budget_enforce,
    extract_budget_tokens,
)
from agent_framework_task_budget.integration import _merge_middleware


class FakeAgent:
    """Duck-typed agent: a ``middleware`` attribute and an async ``run``."""

    def __init__(self, middleware=None):
        self.middleware = middleware
        self.run_calls = []

    async def run(self, *args, **kwargs):
        self.run_calls.append((args, kwargs))
        return "RESPONSE"


def _chat_mw(middleware):
    return [m for m in middleware if isinstance(m, TaskBudgetChatMiddleware)]


# --------------------------------------------------------------------------- #
# _merge_middleware
# --------------------------------------------------------------------------- #
def test_merge_middleware_from_none():
    assert _merge_middleware(None, ["a"]) == ["a"]


def test_merge_middleware_from_list():
    assert _merge_middleware(["x"], ["a"]) == ["x", "a"]


def test_merge_middleware_from_single():
    sentinel = object()
    assert _merge_middleware(sentinel, ["a"]) == [sentinel, "a"]


# --------------------------------------------------------------------------- #
# enable_task_budget — per-request model (caller imports nothing from this lib)
# --------------------------------------------------------------------------- #
def test_enable_wires_per_request_chat_and_enforcement_middleware():
    agent = FakeAgent()
    returned = enable_task_budget(agent)
    assert returned is agent  # chainable
    assert len(_chat_mw(agent.middleware)) == 1
    # per-request: the chat middleware carries no fixed budget
    assert _chat_mw(agent.middleware)[0].budget is None
    # enforcement middleware is always attached, but metadata-driven: it carries no
    # fixed budget, so it stays advisory-only unless a run opts in via the request.
    enf = [m for m in agent.middleware if isinstance(m, TaskBudgetEnforcementMiddleware)]
    assert len(enf) == 1
    assert enf[0].budget is None


def test_enable_no_longer_accepts_server_side_enforce():
    # Enforcement is the caller's per-run choice, so the old server-side switch is
    # gone: passing it must be a TypeError, not a silent no-op.
    agent = FakeAgent()
    with pytest.raises(TypeError):
        enable_task_budget(agent, enforce=True)


def test_enable_preserves_existing_middleware():
    sentinel = object()
    agent = FakeAgent(middleware=[sentinel])
    enable_task_budget(agent)
    assert agent.middleware[0] is sentinel
    assert len(_chat_mw(agent.middleware)) == 1


def test_enable_uses_setattr_fallback_for_restricted_agent():
    class RestrictedAgent:
        def __init__(self):
            object.__setattr__(self, "middleware", None)

        def __setattr__(self, name, value):
            raise ValueError("attribute assignment is blocked")

    agent = RestrictedAgent()
    enable_task_budget(agent)
    assert len(_chat_mw(agent.middleware)) == 1


# --------------------------------------------------------------------------- #
# Server-side hosting helpers — remote caller passes a plain JSON field
# --------------------------------------------------------------------------- #
def test_extract_budget_top_level_int():
    assert extract_budget_tokens({"message": "x", "agent_framework_task_budget_tokens": 80_000}) == 80_000


def test_extract_budget_alias_key():
    assert extract_budget_tokens({"agent_framework_task_budget": 50_000}) == 50_000


def test_extract_budget_from_metadata_string():
    # OpenAI `metadata` values are transmitted as strings.
    assert extract_budget_tokens({"metadata": {"agent_framework_task_budget_tokens": "80000"}}) == 80_000


def test_extract_budget_from_extra_body():
    assert extract_budget_tokens({"extra_body": {"agent_framework_task_budget": 12_345}}) == 12_345


def test_extract_budget_missing_returns_none():
    assert extract_budget_tokens({"message": "x"}) is None


def test_extract_budget_rejects_bool_and_nonpositive():
    assert extract_budget_tokens({"agent_framework_task_budget_tokens": True}) is None
    assert extract_budget_tokens({"agent_framework_task_budget_tokens": 0}) is None
    assert extract_budget_tokens({"agent_framework_task_budget_tokens": -5}) is None


def test_extract_budget_rejects_non_mapping():
    assert extract_budget_tokens(None) is None
    assert extract_budget_tokens("80000") is None


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"agent_framework_task_budget_enforce": "true"}, True),
        ({"agent_framework_task_budget_enforce": True}, True),
        ({"metadata": {"agent_framework_task_budget_enforce": "1"}}, True),
        ({"extra_body": {"agent_framework_task_budget_enforce": "on"}}, True),
        ({"agent_framework_task_budget_enforce": "false"}, False),
        ({"agent_framework_task_budget_enforce": "nope"}, False),
        ({"message": "x"}, False),
        (None, False),
    ],
)
def test_extract_budget_enforce(payload, expected):
    assert extract_budget_enforce(payload) is expected


# --------------------------------------------------------------------------- #
# _ensure_budget_enabled — the auto-wiring the host factories apply so that
# hosting an un-wired agent still makes the per-request budget take effect
# (guards against the silent "agent_framework_task_budget_tokens changes nothing" footgun).
# --------------------------------------------------------------------------- #
def _budget_mws(middleware):
    from agent_framework_task_budget.integration import _BUDGET_MIDDLEWARE_TYPES

    if middleware is None:
        seq = []
    elif isinstance(middleware, list):
        seq = middleware
    else:
        seq = [middleware]
    return [m for m in seq if isinstance(m, _BUDGET_MIDDLEWARE_TYPES)]


def test_ensure_budget_enabled_wires_unwired_agent():
    from agent_framework_task_budget.integration import _ensure_budget_enabled

    agent = FakeAgent()  # middleware is None — the user's un-wired server
    _ensure_budget_enabled(agent)
    # The reader middleware are now present, so a bound budget is read.
    assert len(_chat_mw(agent.middleware)) == 1
    assert any(isinstance(m, TaskBudgetEnforcementMiddleware) for m in agent.middleware)


def test_ensure_budget_enabled_is_idempotent():
    from agent_framework_task_budget.integration import _ensure_budget_enabled

    agent = FakeAgent()
    enable_task_budget(agent)  # caller wired it explicitly
    before = list(agent.middleware)
    _ensure_budget_enabled(agent)  # must NOT add a second set
    assert agent.middleware == before
    assert len(_chat_mw(agent.middleware)) == 1  # no duplicate countdown


def test_ensure_budget_enabled_preserves_existing_middleware():
    from agent_framework_task_budget.integration import _ensure_budget_enabled

    sentinel = object()  # an unrelated, non-budget middleware
    agent = FakeAgent(middleware=[sentinel])
    _ensure_budget_enabled(agent)
    assert agent.middleware[0] is sentinel
    assert len(_budget_mws(agent.middleware)) == 2


def test_ensure_budget_enabled_handles_single_middleware_object():
    from agent_framework_task_budget.integration import _ensure_budget_enabled

    sentinel = object()
    agent = FakeAgent(middleware=sentinel)  # a bare object, not a list
    _ensure_budget_enabled(agent)
    assert sentinel in agent.middleware
    assert len(_budget_mws(agent.middleware)) == 2


def test_ensure_budget_enabled_detects_partial_wiring_via_single_object():
    from agent_framework_task_budget.integration import _ensure_budget_enabled

    agent = FakeAgent(middleware=TaskBudgetChatMiddleware())  # already budget-ish
    _ensure_budget_enabled(agent)  # should treat as enabled → no-op
    assert len(_budget_mws(agent.middleware)) == 1


# --------------------------------------------------------------------------- #
# The host factories themselves apply the fallback: calling budget_responses_host
# / budget_invocations_host on an agent that was NOT enable_task_budget'd still
# wires the reader middleware, so a per-request budget still takes effect. This is
# the exact scenario a user hits when their server only wraps with the host factory.
# (Needs the optional agent-framework-foundry-hosting package.)
# --------------------------------------------------------------------------- #
def test_budget_responses_host_autowires_unwired_agent():
    pytest.importorskip("agent_framework_foundry_hosting")
    from agent_framework_task_budget import budget_responses_host

    agent = FakeAgent()  # the user's server: hosted WITHOUT enable_task_budget
    assert _budget_mws(agent.middleware) == []
    budget_responses_host(agent)  # hosting it wires the readers as a fallback
    assert len(_budget_mws(agent.middleware)) == 2


def test_budget_responses_host_does_not_double_wire_when_already_enabled():
    pytest.importorskip("agent_framework_foundry_hosting")
    from agent_framework_task_budget import budget_responses_host

    agent = FakeAgent()
    enable_task_budget(agent)  # explicitly wired first
    before = list(agent.middleware)
    budget_responses_host(agent)  # idempotent: must not add a second set of readers
    assert agent.middleware == before
    assert len(_chat_mw(agent.middleware)) == 1


def test_budget_invocations_host_autowires_before_agent_type_check():
    pytest.importorskip("agent_framework_foundry_hosting")
    from agent_framework_task_budget import budget_invocations_host

    agent = FakeAgent()
    # The invocations host validates the agent type at construction, which a
    # duck-typed FakeAgent does not satisfy; but the budget fallback runs first,
    # so the agent is wired regardless of whether construction then succeeds.
    try:
        budget_invocations_host(agent)
    except TypeError:
        pass
    assert len(_budget_mws(agent.middleware)) == 2
