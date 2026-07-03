"""End-to-end LIVE verification of the streaming fix over the *full* hosted HTTP path.

Drives ``budget_responses_host`` through a real network round-trip with the OpenAI
SDK Responses API in **streaming** mode (``stream=True`` → SSE), confirming that the
per-request budget is charged and the enforcement backstop short-circuits a tool call
even when the client streams — i.e. the fix works end-to-end through
``budget_responses_host._handle_response`` → ``bind_budget_over`` → SSE, not just via
the binding helper in isolation.

  OpenAI SDK (stream=True) --HTTP/SSE--> uvicorn --> budget_responses_host --> agent

Requires ``az login`` and ``FOUNDRY_PROJECT_ENDPOINT``. Run:
    .venv/Scripts/python.exe -W ignore scratch/verify_remote_streaming.py
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time

os.environ.setdefault("OTEL_SDK_DISABLED", "true")
for _name in ("azure.ai.agentserver", "azure", "agent_framework", "uvicorn.access", "uvicorn.error", "httpx"):
    logging.getLogger(_name).setLevel(logging.WARNING)

import uvicorn
from azure.identity import AzureCliCredential
from openai import OpenAI

from agent_framework import ChatMiddleware, FunctionMiddleware
from agent_framework.foundry import FoundryChatClient

from agent_framework_task_budget import budget_responses_host, enable_task_budget
from agent_framework_task_budget.maf_adapter import _current_budget

ENDPOINT = os.environ.get("FOUNDRY_PROJECT_ENDPOINT")
MODEL = os.environ.get("FOUNDRY_MODEL", "gpt-5.4-mini")
if not ENDPOINT:
    raise SystemExit("Set FOUNDRY_PROJECT_ENDPOINT, e.g. https://<res>.services.ai.azure.com/api/projects/<proj>")

_WEATHER = {
    "Tokyo": "sunny, 28C", "Paris": "cloudy, 19C", "New York": "rainy, 22C",
    "London": "overcast, 15C", "Sydney": "windy, 24C", "Cairo": "hot and dry, 34C",
    "Rio de Janeiro": "humid, 30C", "Berlin": "drizzly, 12C", "Toronto": "cold, 7C",
    "Mumbai": "muggy, 31C", "Oslo": "frosty, 2C", "Lima": "grey, 18C",
}
ITINERARY = list(_WEATHER)
_cursor = {"i": 0}

OBS: dict[str, object] = {}


def _reset_obs() -> None:
    OBS.update(attempts=[], executed=[], model_calls=0, countdown_calls=0, budget_ref=None)
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
    """Return the weather for the NEXT city (no args); advances a server-side cursor."""
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
    async def process(self, context, call_next):
        budget = _current_budget.get()
        if budget is not None and OBS.get("budget_ref") is None:
            OBS["budget_ref"] = budget
        injected = _has_countdown(context)
        await call_next()
        injected = injected or _has_countdown(context)
        OBS["model_calls"] += 1  # type: ignore[operator]
        if injected:
            OBS["countdown_calls"] += 1  # type: ignore[operator]


class ObsTool(FunctionMiddleware):
    async def process(self, context, call_next):
        fn = getattr(context, "function", None)
        OBS["attempts"].append(getattr(fn, "name", None) or "<tool>")  # type: ignore[union-attr]
        await call_next()


def build_host():
    client = FoundryChatClient(
        project_endpoint=ENDPOINT, model=MODEL, credential=AzureCliCredential(), allow_preview=True
    )
    agent = client.as_agent(
        name="readme-verify",
        instructions=LOOP_INSTR,
        tools=[next_city_weather],
        middleware=[ObsChat(), ObsTool()],
        default_options={"store": False},
    )
    enable_task_budget(agent)
    return budget_responses_host(agent)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Serving:
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


def client_call(port: int, *, budget: int, enforce: bool, stream: bool) -> str:
    client = OpenAI(base_url=f"http://127.0.0.1:{port}", api_key="local", timeout=600)
    metadata = {"agent_framework_task_budget_tokens": str(budget)}
    if enforce:
        metadata["agent_framework_task_budget_enforce"] = "true"
    kwargs = {"model": "readme-verify", "input": LOOP_TASK, "store": False, "metadata": metadata}
    if not stream:
        resp = client.responses.create(**kwargs)  # type: ignore[arg-type]
        return resp.output_text or ""
    # Streaming: consume the SSE event stream and reconstruct the final text.
    parts: list[str] = []
    final_text = ""
    for event in client.responses.create(stream=True, **kwargs):  # type: ignore[arg-type]
        etype = getattr(event, "type", "") or ""
        if etype == "response.output_text.delta":
            parts.append(getattr(event, "delta", "") or "")
        elif etype in ("response.completed", "response.incomplete"):
            resp = getattr(event, "response", None)
            if resp is not None:
                final_text = getattr(resp, "output_text", "") or ""
    return final_text or "".join(parts)


def snapshot(label: str, answer: str) -> dict:
    b = OBS.get("budget_ref")
    return {
        "label": label,
        "iterations": len(OBS["executed"]),  # type: ignore[arg-type]
        "attempts": len(OBS["attempts"]),  # type: ignore[arg-type]
        "model_calls": OBS["model_calls"],
        "countdown_calls": OBS["countdown_calls"],
        "total": None if b is None else b.total,
        "remaining": None if b is None else b.remaining,
        "blocked": len(OBS["attempts"]) > len(OBS["executed"]),  # type: ignore[arg-type]
        "answer_len": len(answer.strip()),
        "answer": answer.strip()[:260],
    }


def _print(s: dict) -> None:
    print(
        f"  [{s['label']}] iterations={s['iterations']} attempts={s['attempts']} "
        f"model_calls={s['model_calls']} countdown_calls={s['countdown_calls']} "
        f"budget={s['total']}->{s['remaining']} blocked={s['blocked']} answer_len={s['answer_len']}"
    )
    print(f"        answer: {s['answer']}")


def main() -> None:
    scarce = int(os.environ.get("SCARCE_BUDGET", "3500"))
    print("=" * 78)
    print("REMOTE STREAMING verification — OpenAI SDK stream=True -> uvicorn -> budget_responses_host")
    print(f"model={MODEL}  budget={scarce}  itinerary={len(ITINERARY)} cities")
    print("=" * 78)

    with _Serving(build_host()) as srv:
        print(f"\n[host] serving on http://127.0.0.1:{srv.port}")

        print(f"\n-- STREAMING client (stream=True) + enforce, budget={scarce} --")
        _reset_obs()
        s = snapshot("STREAM", client_call(srv.port, budget=scarce, enforce=True, stream=True))
        _print(s)

        print(f"\n-- non-stream client + enforce, budget={scarce} (parity) --")
        _reset_obs()
        n = snapshot("NOSTRM", client_call(srv.port, budget=scarce, enforce=True, stream=False))
        _print(n)

    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    checks = {
        "STREAM: budget charged over HTTP/SSE (remaining < total)":
            s["remaining"] is not None and s["total"] is not None and s["remaining"] < s["total"],
        "STREAM: budget exhausted (remaining == 0)": s["remaining"] == 0,
        "STREAM: enforcement short-circuited a tool call (attempts > executed)": s["blocked"],
        "STREAM: countdown injected on every model call": s["countdown_calls"] == s["model_calls"] and s["model_calls"] > 0,
        "STREAM: non-empty graceful partial": s["answer_len"] > 0,
        "parity: non-stream also charged + blocked":
            (n["remaining"] is not None and n["total"] is not None and n["remaining"] < n["total"]) and n["blocked"],
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("\nSIDE-BY-SIDE:")
    for r in (s, n):
        print(
            f"  {r['label']}: budget {r['total']}->{r['remaining']}  iterations={r['iterations']} "
            f"attempts={r['attempts']} blocked={r['blocked']} answer_len={r['answer_len']}"
        )
    print("\nOVERALL:", "PASS" if all(checks.values()) else "SEE FAILURES ABOVE")


if __name__ == "__main__":
    main()
