"""Reference: enforce a *per-request* task budget in a Foundry **hosted agent**,
driven by a client using the **OpenAI SDK Responses API**.

THE SCENARIO
------------
The MAF agent is deployed as a hosted agent (a container behind a remote API,
e.g. on Azure AI Foundry). The agent *loop* — and therefore this library's
middleware — runs **server-side, inside the container**. A remote client drives
the agent with the **OpenAI SDK Responses API** and passes the budget *in the
request's ``metadata``*; the server-side agent then governs its loop against it.

The remote client imports NOTHING from ``agent_framework_task_budget``. It only adds one field
to the standard Responses call::

    client.responses.create(
        model="my-agent",
        input="Investigate the outage",
        metadata={"agent_framework_task_budget_tokens": "80000"},
    )

WHY ``budget_responses_host`` IS NEEDED
---------------------------------------
The stock OpenAI-compatible **Responses** host in
``agent-framework-foundry-hosting`` maps only
``temperature``/``top_p``/``max_output_tokens``/``parallel_tool_calls`` onto the
run and **drops** request ``metadata``/``extra_body`` before the agent runs. So a
custom budget field would never reach the agent on that path.

``budget_responses_host`` is a thin ``ResponsesHostServer`` subclass that reads
``request.metadata``, extracts ``agent_framework_task_budget_tokens``, and binds it across the
server-side run (via ``bind_budget_over``) so the existing chat middleware sees
it. It overrides one internal method (``_handle_response``); pin the hosting
version or re-check it on upgrade.

NOTE: this file is a *deployment reference*. It needs the hosting package and an
actual Foundry deployment to run; it is not a local, offline demo.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# SERVER SIDE — this code ships INSIDE the hosted-agent container.
# --------------------------------------------------------------------------- #
# `agent_framework_task_budget` is imported ONLY here, on the server/agent-setup side.
from agent_framework_task_budget import budget_responses_host, enable_task_budget


def build_agent():
    """Create the MAF agent that this container hosts.

    Swap the chat client for your real connector. The budget wiring below is
    connector-independent.
    """
    import os

    from agent_framework.foundry import FoundryChatClient  # type: ignore
    from azure.identity import DefaultAzureCredential

    client = FoundryChatClient(
        project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        model=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
        credential=DefaultAzureCredential(),
    )
    agent = client.as_agent(
        name="incident-investigator",
        instructions="Investigate incidents and report findings.",
        # ...tools=[...]
    )

    # Wire the budget ONCE: advisory countdown + token accounting on every run.
    # This attaches the middleware that reads the budget bound per request by
    # budget_responses_host. Pass any server-side fallback via
    # budget_responses_host(default_total=...) below.
    enable_task_budget(agent)
    return agent


agent = build_agent()

# Host the agent on the Responses API. budget_responses_host reads the budget the
# client put in request.metadata and binds it across the server-side run.
# Pass default_total=... for a server-side fallback when a request omits it.
app = budget_responses_host(agent)  # add hosting kwargs (credential, etc.) as needed
# app.run(...)  # serve it per agent-framework-foundry-hosting


# --------------------------------------------------------------------------- #
# CLIENT SIDE — runs anywhere, imports ONLY the OpenAI SDK.
# --------------------------------------------------------------------------- #
def call_remote_agent() -> str:  # pragma: no cover - needs a live endpoint
    """Drive the hosted agent with the OpenAI SDK Responses API.

    The budget rides along in ``metadata`` (Responses metadata values are
    strings; the server converts the digit string). To resume a partly-spent
    budget across calls, also send ``"agent_framework_task_budget_remaining": "<int>"``.
    """
    from openai import OpenAI

    client = OpenAI(base_url="https://<your-hosted-agent-endpoint>", api_key="...")

    response = client.responses.create(
        model="my-agent",
        input="Investigate the outage...",
        metadata={"agent_framework_task_budget_tokens": "80000"},  # ← the budget, in the request
    )
    return response.output_text


# --------------------------------------------------------------------------- #
# ALTERNATIVE — the Invocations protocol (plain JSON body instead of OpenAI SDK).
# --------------------------------------------------------------------------- #
# If you host on the Invocations protocol instead of Responses, the client sends a
# plain JSON body ({"message": "...", "agent_framework_task_budget_tokens": 80000}) and needs no
# OpenAI SDK. The stock InvocationsHostServer's handler reads only "message"/"stream"
# and drops extra fields, so budget_invocations_host subclasses it the same way
# budget_responses_host does — lifting the budget from the body and binding it across
# both the non-streaming and streaming run paths:
#
#   from agent_framework_task_budget import budget_invocations_host, enable_task_budget
#
#   enable_task_budget(agent)                 # attach the budget-reading middleware
#   inv_app = budget_invocations_host(agent)  # pass default_total=... here if needed
#   # inv_app.run(...) — serve it per agent-framework-foundry-hosting
