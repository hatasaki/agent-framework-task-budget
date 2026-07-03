"""Tests for the framework-agnostic budget core."""

import pytest

from agent_framework_task_budget import TaskBudget


def test_defaults_remaining_to_total():
    b = TaskBudget(total=20_000)
    assert b.remaining == 20_000


def test_min_total_validation_rejects_small_budget():
    with pytest.raises(ValueError):
        TaskBudget(total=100)


def test_min_total_can_be_disabled():
    b = TaskBudget(total=100, min_total=0)
    assert b.total == 100
    assert b.remaining == 100


def test_consume_subtracts():
    b = TaskBudget(total=20_000)
    b.consume(5_000)
    assert b.remaining == 15_000


def test_consume_clamps_at_zero():
    b = TaskBudget(total=20_000)
    b.consume(25_000)
    assert b.remaining == 0
    assert b.exhausted


def test_consume_ignores_nonpositive():
    b = TaskBudget(total=20_000)
    b.consume(0)
    b.consume(-5)
    assert b.remaining == 20_000


def test_exhausted_is_false_when_budget_remains():
    b = TaskBudget(total=20_000)
    assert not b.exhausted


def test_fraction_left():
    b = TaskBudget(total=20_000)
    b.consume(5_000)
    assert b.fraction_left == pytest.approx(0.75)


def test_render_status_contains_advisory_and_numbers():
    b = TaskBudget(total=20_000)
    b.consume(5_000)
    text = b.render_status()
    assert "advisory" in text.lower()
    assert "20,000" in text
    assert "15,000" in text


def test_snapshot_restore_roundtrip():
    b = TaskBudget(total=30_000)
    b.consume(10_000)
    restored = TaskBudget.restore(b.snapshot())
    assert restored.total == 30_000
    assert restored.remaining == 20_000
