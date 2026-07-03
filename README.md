# agent-framework-task-budget

Minimal task budget extension for the
[Microsoft Agent Framework](https://learn.microsoft.com/agent-framework/) (MAF).

It adds a token **task budget** to a MAF agent:
before every model call the agent is shown an **advisory countdown** of how much of
its token budget remains, and the tokens it spends (thinking + tool calls + tool
results + output) are **accounted** against that budget. An optional backstop can
**steer the model to wrap up** with what it already gathered when the budget is
exhausted (a graceful partial), instead of erroring out.

Two parts: **server side**, wire the budget into your MAF agent and host it (§1–2);
**client side**, a remote caller sets a token budget per request with one `metadata`
field (§3), importing nothing from this library.

## 1. Install

Install straight from GitHub — **no `git clone` needed** (pip fetches and builds it):

```pwsh
pip install git+https://github.com/hatasaki/agent-framework-task-budget.git
```

This also pulls in `agent-framework` automatically. Pin a specific version by
appending `@<tag-or-commit>`. If `git` isn't available on the machine, install the
source tarball instead (no git required):

```pwsh
pip install https://github.com/hatasaki/agent-framework-task-budget/archive/refs/heads/main.tar.gz
```

Working from a local clone instead? Use `pip install -e .` (editable), or copy this
repo's [`src/agent_framework_task_budget/`](src/agent_framework_task_budget/) folder into your project (so you can
`import agent_framework_task_budget`).

## 2. Add it to your existing MAF agent

Adding a task budget to an existing MAF agent is **two added lines** — wire the
budget, then host the wrapped agent. Only the two marked lines are new; the rest is
your existing code:

```python
from agent_framework_task_budget import enable_task_budget, budget_responses_host

agent = FoundryChatClient(model="gpt-4o").as_agent(instructions="...")  # your existing agent

enable_task_budget(agent)              # ← add: advisory countdown + token accounting
app = budget_responses_host(agent)     # ← add: host it, per agent-framework-foundry-hosting
```

The stock Responses host drops request `metadata`, so `budget_responses_host` is the
piece that lifts the per-request budget out of `metadata` and binds it across the
server-side run. That is all the **server-side** wiring; the budget itself is supplied
per request by the **remote client** — see
[§3, Calling a remote hosted agent](#3-calling-a-remote-hosted-agent-with-the-openai-sdk-responses-api).

## 3. Calling a remote hosted agent with the OpenAI SDK (Responses API)

This section is the **client** side. The agent — and this library's middleware —
run server-side inside the hosted container (§2); the client just drives that hosted
agent with the **OpenAI SDK Responses API** and passes the budget **in the request's
`metadata`**. The client **imports nothing** from this library.

Set the budget per request; a request without it is a no-op. Responses `metadata`
values are **strings**:

```python
from openai import OpenAI

client = OpenAI(base_url="https://<your-hosted-agent-endpoint>", api_key="...")

client.responses.create(
    model="my-agent",
    input="Investigate the flaky CI test.",
    metadata={"agent_framework_task_budget_tokens": "80000"},      # advisory only
)
```

By default the budget is **advisory** (a countdown the model self-paces against). To
add a hard **wrap-up backstop** for a request, the client opts in on that same
`metadata` — no extra wiring on the server:

```python
client.responses.create(
    model="my-agent",
    input="Investigate the flaky CI test.",
    metadata={
        "agent_framework_task_budget_tokens": "80000",
        "agent_framework_task_budget_enforce": "true",             # omit or "false" = advisory only
    },
)
```

The same `metadata` also governs **streaming** requests: call
`client.responses.create(..., stream=True)` and the budget is still accounted and the
backstop still enforced across the streamed run — no change to the request beyond
`stream=True`.

A full deployment reference — server host, client call, and the Invocations
alternative — is in
[examples/foundry_hosted_agent.py](examples/foundry_hosted_agent.py).

## 4. What happens when you set a budget — verified patterns

The behaviour below was verified live against a Foundry-hosted `gpt-5.4-mini`,
comparing runs **with and without** a budget on the *same* agent. What you get
depends on the budget size, the shape of the task, and whether the enforcement
backstop is on.

### Pattern 1 — advisory budget on a real tool loop: the model paces itself

```python
# client request metadata — advisory only (see §3 for the full call):
metadata={"agent_framework_task_budget_tokens": "3582"}
```

The countdown is a **hint, not a limit** — the loop is never force-stopped. But on a
genuine multi-step tool loop (here: a 12-city itinerary walked **one tool call at a
time**), seeing the remaining budget before every model call makes the model **stop
early and wrap up** instead of grinding through every step. Measured on the same
task, budget set to ~40% of the run's natural cost:

| | advisory OFF | advisory ON |
|---|---|---|
| loop stopped? | no (ran to completion) | no (model *chose* to stop) |
| tool iterations | 12 | 6 |
| total tokens | 8,955 | 5,216 |
| final answer | all 12 cities | first 6 cities + wrap-up |

So on a real loop the advisory **does** cut the token bill, because the model runs
fewer iterations: seeing the remaining budget lets the model pace its tool use and
wrap up before it runs out.

> **Caveat — short tasks.** If the task is already just one or two tool calls, there
> is nothing to pace: the countdown text is extra input the model pays for, so total
> tokens can go slightly *up* while the answer gets marginally shorter. The
> self-pacing benefit shows up on genuine multi-step loops, not on trivial tasks.

### Pattern 2 — budget runs out with enforcement on: a graceful partial, not an empty answer

```python
# client request metadata — opt into the wrap-up backstop (see §3 for the full call):
metadata={"agent_framework_task_budget_tokens": "3582", "agent_framework_task_budget_enforce": "true"}
```

Adding `"agent_framework_task_budget_enforce": "true"` turns on a backstop **on top of** the advisory
(the countdown is still injected before every model call). When the budget is spent,
the next tool call is **short-circuited**: instead of running the tool, the backstop
hands the model a *"budget exhausted — stop calling tools and write your best answer
from what you already have"* result. The model then produces a **graceful partial**,
not an empty response.

Verified on the 12-city loop, budget ~40% of the natural cost (runs out after 5
cities):

- the model gathered Tokyo, Paris, New York, London and Sydney,
- the 6th tool call was short-circuited by the backstop,
- the run finished with a **337-character recap** of the five cities it had, plus a
  note that it *"couldn't complete the full itinerary because the tool budget was
  exhausted."*

The backstop stops the loop **between turns** and keeps the assistant's
already-generated text — it never truncates a reply to empty. The advisory-only run
at the same budget wraps up similarly on its own; `agent_framework_task_budget_enforce` simply
*guarantees* the tool loop stops.

### Pattern 3 — enforcement on but the task fits: it just completes

```python
# client request metadata for a task that comfortably fits (see §3 for the full call):
metadata={"agent_framework_task_budget_tokens": "200000", "agent_framework_task_budget_enforce": "true"}
```

When the work comfortably fits the budget, `agent_framework_task_budget_enforce` behaves exactly like
advisory mode with a backstop that never fires. Observed: both tool calls ran, the
countdown was injected before **every** model call, the budget went
`200,000 → 199,300`, and the run finished normally with a clean one-sentence answer.

## Compatibility

**Verified with MAF (`agent-framework`) v1.10.**

## Test

```pwsh
pip install -e ".[dev]"
pytest
```

## License

MIT
