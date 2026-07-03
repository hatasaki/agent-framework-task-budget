"""Live REMOTE-CLIENT verification of the task budget over a real HTTP round-trip.

This script exercises the *actual hosted-agent path the user runs*:

  1. build a MAF ``Agent`` with a stateful tool that forces a genuine tool loop,
  2. wire it with ``enable_task_budget(agent)`` (the step the user's server omitted),
  3. host it with ``budget_responses_host(agent)`` on a real uvicorn server bound to
     127.0.0.1, and
  4. drive it from a **separate OpenAI SDK client over TCP**, passing the budget in
     the Responses ``metadata`` — exactly like the user's request body.

Patterns:
  A  no budget                         -> baseline: the whole loop runs.
  B  metadata task_budget_tokens only  -> advisory: model paces itself, fewer steps.
  C  metadata tokens + enforce="true"  -> backstop: a tool call is short-circuited
                                          and the model wraps up (graceful partial).
  USER-CODE  the user's exact server (``budget_responses_host(agent)`` with NO
         explicit ``enable_task_budget``) + the same enforce request. The host
         factory now auto-wires the reader middleware, so the budget takes effect
         (this used to be a silent no-op).

Requires az login and FOUNDRY_PROJECT_ENDPOINT. Run:
    .venv/Scripts/python.exe -W ignore scratch/verify_remote_patterns.py
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time

# Quiet the hosting stack's INFO logs and OpenTelemetry console exporters so the
# pattern results are readable (set before importing the hosting packages).
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
for _name in ("azure.ai.agentserver", "azure", "agent_framework", "uvicorn.access", "uvicorn.error", "httpx"):
    logging.getLogger(_name).setLevel(logging.WARNING)

import uvicorn
from azure.identity import AzureCliCredential

from agent_framework import ChatMiddleware, FunctionMiddleware
from agent_framework.foundry import FoundryChatClient
from openai import OpenAI

from agent_framework_task_budget import budget_responses_host, enable_task_budget
from agent_framework_task_budget.maf_adapter import _current_budget, _extract_total_tokens

ENDPOINT = os.environ.get("FOUNDRY_PROJECT_ENDPOINT")
MODEL = os.environ.get("FOUNDRY_MODEL", "gpt-5.4-mini")
if not ENDPOINT:
    raise SystemExit(
        "Set FOUNDRY_PROJECT_ENDPOINT, e.g. "
        "https://<resource>.services.ai.azure.com/api/projects/<project>"
    )

# ------------------------------------------------------------------ #
# A stateful tool that forces a genuine, un-batchable sequential loop.
# ------------------------------------------------------------------ #
_WEATHER = {
    "Tokyo": "sunny, 28C",
    "Paris": "cloudy, 19C",
    "New York": "rainy, 22C",
    "London": "overcast, 15C",
    "Sydney": "windy, 24C",
    "Cairo": "hot and dry, 34C",
    "Rio de Janeiro": "humid, 30C",
    "Berlin": "drizzly, 12C",
    "Toronto": "cold, 7C",
    "Mumbai": "muggy, 31C",
    "Oslo": "frosty, 2C",
    "Lima": "grey, 18C",
}
ITINERARY = list(_WEATHER)
_cursor = {"i": 0}

#: Cross-thread observability (server runs in the uvicorn thread; the client and the
#: reader run in the main thread — reset before a request, read after it returns, so
#: there is never concurrent access).
OBS: dict[str, object] = {}


def _reset_obs() -> None:
    OBS.update(
        attempts=[],          # every tool the model TRIED to call (before enforcement)
        executed=[],          # tool bodies that actually ran (short-circuits excluded)
        model_calls=0,        # number of model calls in the run
        countdown_calls=0,    # model calls that saw the advisory countdown
        tokens=0,             # total accounted tokens
        remaining=[],         # budget.remaining after each model call
    )
    _cursor["i"] = 0


LOOP_INSTR = (
    "You are a travel weather assistant. You gather weather ONE city at a time by "
    "calling the next_city_weather tool (it takes NO arguments and automatically "
    "advances to the next city). Call it, read the result, then call it AGAIN for the "
    "next city -- never guess weather from memory, never stop early on your own. "
    "Continue until the tool says 'ITINERARY COMPLETE' or you are told to stop, then "
    "write a one-line weather recap for every city you collected."
)
LOOP_TASK = (
    "Plan the weather for my whole multi-city trip. Use next_city_weather (no "
    "arguments) repeatedly, one call at a time, walking through the entire itinerary "
    "until it reports 'ITINERARY COMPLETE'. When you stop, give a one-line recap per "
    "city you collected."
)


async def next_city_weather() -> str:
    """Return the weather for the NEXT city on the itinerary (takes no arguments).

    Stateful: each call advances a server-side cursor, so the model MUST call it
    repeatedly (a genuine sequential loop) and cannot fire the calls in parallel.
    """
    i = _cursor["i"]
    if i >= len(ITINERARY):
        return "ITINERARY COMPLETE -- no more cities. Write your final recap now."
    city = ITINERARY[i]
    _cursor["i"] = i + 1
    OBS["executed"].append(city)  # type: ignore[union-attr]
    return (
        f"City {i + 1} of {len(ITINERARY)}: {city} -- {_WEATHER[city]}. "
        "Call next_city_weather again for the next city."
    )


def _has_countdown(context) -> bool:
    for m in getattr(context, "messages", None) or []:
        text = getattr(m, "text", None) or str(getattr(m, "contents", ""))
        if "Task budget" in text:
            return True
    return False


class ObsChat(ChatMiddleware):
    """Record, per model call: whether the countdown was injected + usage + remaining."""

    async def process(self, context, call_next):
        injected = _has_countdown(context)
        budget = _current_budget.get()
        await call_next()
        injected = injected or _has_countdown(context)
        OBS["model_calls"] += 1  # type: ignore[operator]
        if injected:
            OBS["countdown_calls"] += 1  # type: ignore[operator]
        OBS["tokens"] += _extract_total_tokens(getattr(context, "result", None))  # type: ignore[operator]
        OBS["remaining"].append(None if budget is None else budget.remaining)  # type: ignore[union-attr]


class ObsTool(FunctionMiddleware):
    """Record every tool the model TRIES to call, before enforcement can block it."""

    async def process(self, context, call_next):
        fn = getattr(context, "function", None)
        OBS["attempts"].append(getattr(fn, "name", None) or "<tool>")  # type: ignore[union-attr]
        await call_next()


def build_host(*, wire: bool):
    """Build the Responses host. ``wire=True`` calls ``enable_task_budget`` explicitly;
    ``wire=False`` is the user's exact server (``budget_responses_host(agent)`` only) --
    which now still works because the host factory auto-wires the reader middleware
    when the agent was not already budget-enabled."""
    client = FoundryChatClient(
        project_endpoint=ENDPOINT,
        model=MODEL,
        credential=AzureCliCredential(),
        allow_preview=True,
    )
    agent = client.as_agent(
        name="readme-verify",
        instructions=LOOP_INSTR,
        tools=[next_city_weather],
        middleware=[ObsChat(), ObsTool()],
        default_options={"store": False},
    )
    if wire:
        enable_task_budget(agent)  # explicit wiring (belt-and-suspenders; still fine)
    return budget_responses_host(agent)  # auto-wires the readers if wire=False


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Serving:
    """Run a host on uvicorn in a background thread; stop it on exit."""

    def __init__(self, app) -> None:
        self.port = _free_port()
        self._server = uvicorn.Server(
            uvicorn.Config(app=app, host="127.0.0.1", port=self.port, log_level="warning", lifespan="auto")
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def __enter__(self) -> "_Serving":
        self._thread.start()
        while not self._server.started:
            time.sleep(0.05)
        return self

    def __exit__(self, *exc) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)


def client_call(port: int, *, budget: int | None = None, enforce: bool = False) -> str:
    """Drive the hosted agent with the OpenAI SDK Responses API over TCP."""
    # The host registers POST /responses (prefix defaults to ""); the OpenAI SDK
    # appends "/responses" to base_url, so base_url must NOT include "/v1".
    client = OpenAI(base_url=f"http://127.0.0.1:{port}", api_key="local", timeout=600)
    metadata: dict[str, str] = {}
    if budget is not None:
        metadata["task_budget_tokens"] = str(budget)  # Responses metadata values are strings
    if enforce:
        metadata["task_budget_enforce"] = "true"
    kwargs: dict[str, object] = {"model": "readme-verify", "input": LOOP_TASK, "store": False}
    if metadata:
        kwargs["metadata"] = metadata
    resp = client.responses.create(**kwargs)  # type: ignore[arg-type]
    return resp.output_text or ""


def snapshot(label: str, answer: str) -> dict:
    remaining = OBS["remaining"]  # type: ignore[assignment]
    return {
        "label": label,
        "iterations": len(OBS["executed"]),  # type: ignore[arg-type]
        "attempts": len(OBS["attempts"]),  # type: ignore[arg-type]
        "model_calls": OBS["model_calls"],
        "countdown_calls": OBS["countdown_calls"],
        "tokens": OBS["tokens"],
        "final_remaining": remaining[-1] if remaining else None,  # type: ignore[index]
        "answer_len": len(answer),
        "answer": answer,
    }


def _print(s: dict) -> None:
    print(
        f"  [{s['label']}] iterations={s['iterations']} attempts={s['attempts']} "
        f"model_calls={s['model_calls']} countdown_calls={s['countdown_calls']} "
        f"tokens={s['tokens']} final_remaining={s['final_remaining']} answer_len={s['answer_len']}"
    )
    print(f"        answer: {s['answer'][:220]}{'...' if len(s['answer']) > 220 else ''}")


def main() -> None:
    print("=" * 78)
    print("REMOTE-CLIENT verification — OpenAI SDK -> uvicorn -> budget_responses_host")
    print(f"model={MODEL}  itinerary={len(ITINERARY)} cities (stateful loop tool)")
    print("=" * 78)

    # ---- correctly-wired host: patterns A / B / C ----
    with _Serving(build_host(wire=True)) as srv:
        print(f"\n[wired host] serving on http://127.0.0.1:{srv.port}  (enable_task_budget applied)")

        print("\n-- Pattern A: NO budget (baseline) --")
        _reset_obs()
        a = snapshot("A", client_call(srv.port))
        _print(a)

        scarce = max(1, int(a["tokens"] * 0.4))
        print(f"\n-- Pattern B: metadata task_budget_tokens={scarce} (advisory only) --")
        _reset_obs()
        b = snapshot("B", client_call(srv.port, budget=scarce))
        _print(b)

        print(f"\n-- Pattern C: metadata task_budget_tokens={scarce} + task_budget_enforce=true --")
        _reset_obs()
        c = snapshot("C", client_call(srv.port, budget=scarce, enforce=True))
        _print(c)

    # ---- the user's EXACT code (budget_responses_host only) now works via auto-wire ----
    with _Serving(build_host(wire=False)) as srv2:
        print(f"\n[USER-CODE host] serving on http://127.0.0.1:{srv2.port}  (budget_responses_host ONLY, no explicit enable_task_budget)")
        print(f"\n-- USER-CODE: same client budget={scarce} + enforce (the user's request shape) --")
        _reset_obs()
        usercode = snapshot("USER", client_call(srv2.port, budget=scarce, enforce=True))
        _print(usercode)

    # ---- verdicts ----
    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    b_metadata_effect = b["countdown_calls"] > 0  # task_budget_tokens honored server-side (countdown injected)
    c_changed = c["iterations"] < a["iterations"]  # enforce deterministically stops the loop early
    c_enforced = c["attempts"] > c["iterations"] and c["answer_len"] > 0  # a tool call was short-circuited
    usercode_works = usercode["countdown_calls"] > 0 and usercode["iterations"] < a["iterations"]  # auto-wire fix

    checks = {
        "A ran the full loop (baseline)": a["iterations"] >= 4,
        "B: task_budget_tokens honored remotely (countdown injected every call)": b_metadata_effect,
        "C: enforce deterministically stopped the loop early (fewer iterations than A)": c_changed,
        "C: enforce short-circuited a tool call AND produced a graceful non-empty partial": c_enforced,
        "USER-CODE: budget_responses_host ALONE now honors the budget (auto-wired) = the fix": usercode_works,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    print(
        "\nNOTE: advisory-only (B, task_budget_tokens without enforce) is a HINT, not a limit -- "
        f"it may or may not reduce steps. This run B did {b['iterations']} iterations vs A's "
        f"{a['iterations']} (the model chose to finish); the countdown was still injected on every "
        "call. Enforce (C) is the deterministic control that actually stops the loop."
    )

    print("\nSIDE-BY-SIDE (iterations / tokens / countdown_calls):")
    for s in (a, b, c, usercode):
        print(
            f"  {s['label']:>5}: iterations={s['iterations']:>2}  tokens={s['tokens']:>6}  "
            f"countdown_calls={s['countdown_calls']:>2}  answer_len={s['answer_len']:>4}"
        )

    overall = all(checks.values())
    print("\nOVERALL:", "PASS" if overall else "SEE FAILURES ABOVE")


if __name__ == "__main__":
    main()
