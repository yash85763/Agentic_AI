Here’s a practical, side-by-side deep-dive on **LangGraph** vs the **OpenAI SDK** for agentic AI—what they are, how you build agents with each, where they shine (and don’t), and how to combine them.

---

# TL;DR (opinionated)

* **If you want explicit, testable orchestration** (branching graphs, subgraphs, reliable persistence, time-travel, human-in-the-loop), choose **LangGraph**. It gives you a real *stateful graph runtime* with step-level checkpointing and reducers for parallel branches. ([LangChain][1])
* **If you want a lightweight, batteries-included agent loop** with built-in tracing, sessions, guardrails, handoffs, and first-class access to OpenAI-hosted tools (web/file search, code interpreter, realtime/voice), choose the **OpenAI Agents SDK** (or the vanilla OpenAI SDK + Responses/Tools). ([OpenAI GitHub][2], [GitHub][3])
* You can **mix them**: use LangGraph as the top-level orchestrator and call an OpenAI Agent from a node; or build a small Agent that *handoffs* to a LangGraph subgraph wrapped as a tool. (Pattern shown below.)

---

# Mental model & primitives

| Dimension            | LangGraph                                                                                                                                                 | OpenAI SDK (Agents SDK / Responses)                                                                                                                                                                     |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Core abstraction     | **Stateful graph** of nodes & edges operating over a shared **State** (channels).                                                                         | **Agent loop**: an Agent (instructions + tools) runs until final output; optional **handoffs** between Agents; or low-level **Responses API** with tool calling. ([LangChain][4], [GitHub][3])          |
| Control flow         | Deterministic graphs: `add_edge`, `add_conditional_edges`, subgraphs, parallelism; reducers reconcile parallel writes.                                    | Implicit loop; you model flow via tool calls, guardrails, and handoffs. Complex branching is implemented in code (conditionals) or by composing Agents. ([LangChain][5], [GitHub][3])                   |
| Memory & persistence | **Checkpointers** save step-level state to **threads** (short-term memory), plus a **Store** for long-term memory. Enables time-travel, resume, and HITL. | **Sessions** preserve conversation history; built-in tracing; durable long-running patterns via Temporal integration if needed. ([LangChain][1], [GitHub][3])                                           |
| Tools                | Any Python function/LangChain tool/MCP; you wire execution into nodes or use prebuilt ReAct agent.                                                        | First-class **Tools** (function tools, MCP, plus **hosted** tools like web/file search & code interpreter) through the SDK; or tool calling via **Responses API**. ([LangChain][6], [OpenAI GitHub][7]) |
| Observability        | LangGraph/Smith/Studio ecosystem; state snapshots & replay via checkpoints.                                                                               | Built-in tracing in the OpenAI dashboard; stream events; trace viewer. ([LangChain][1], [OpenAI GitHub][2])                                                                                             |
| Realtime & voice     | Use any stack; you orchestrate.                                                                                                                           | First-class **realtime/voice** primitives and examples. ([OpenAI Platform][8])                                                                                                                          |
| Vendor lock-in       | Model/provider-agnostic by design.                                                                                                                        | Optimized for OpenAI platform & hosted tools; supports other LLMs via LiteLLM in the Agents SDK. ([GitHub][3])                                                                                          |

---

# How agents are created

## A) LangGraph

You can either use a **prebuilt ReAct agent** or build a **custom StateGraph**. The prebuilt is great for “agent-with-tools” loops; the custom path is for complex flows.

**Prebuilt ReAct agent (fast path):**

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

def get_weather(city: str) -> str:
    return f"It's always sunny in {city}!"

checkpointer = InMemorySaver()  # enables thread memory
agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    prompt="You are a helpful assistant.",
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "user-123"}}
agent.invoke({"messages": [{"role": "user", "content": "weather in SF?"}]}, config)
```

* `checkpointer` + `thread_id` → **persistent short-term memory** at every step; enables replay/time-travel/HITL. ([LangChain][6])

**Custom StateGraph (full control):**

```python
from typing_extensions import TypedDict
from typing import Annotated
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    messages: Annotated[list, add]  # reducer accumulates across branches

def call_model(state: State): ...
def maybe_call_tool(state: State): ...

g = StateGraph(State)
g.add_node("llm", call_model)
g.add_node("tools", maybe_call_tool)
g.add_edge(START, "llm")

# Route based on model output: continue to tools or end
def route(state: State): ...
g.add_conditional_edges("llm", route, {"use_tools": "tools", "done": END})

graph = g.compile(checkpointer=InMemorySaver())
graph.invoke({"messages": [{"role": "user", "content": "…"}]},
             config={"configurable": {"thread_id": "u-1"}})
```

* **Reducers** (`Annotated[..., add]`) define how to merge concurrent updates (parallel branches). ([LangChain][4])
* **Conditional edges** give you explicit branching. ([LangChain][5])
* **Persistence/threads** provide durable execution & replay. ([LangChain][1])

## B) OpenAI SDK

You can go **low-level** (Responses API + tool calling) or use the **Agents SDK** (higher-level primitives: Agents, Handoffs, Guardrails, Sessions).

### 1) Low-level: Responses + Tools (DIY orchestration)

* Define tools (functions/MCP); the model emits tool calls; your app executes them and loops until done. ([OpenAI Platform][9])

### 2) High-level: OpenAI **Agents SDK** (recommended)

```python
from agents import Agent, Runner, function_tool, SQLiteSession

@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."

router = Agent(
    name="Triage",
    instructions="Answer or call tools. If it's weather, use the tool.",
    tools=[get_weather],
)

session = SQLiteSession("user-123")  # remembers history
result = Runner.run_sync(router, "Weather in Tokyo?")
print(result.final_output)
```

* **Agents** are LLMs with instructions, tools, and optional **handoffs** to other Agents.
* **Sessions** handle conversation memory; **tracing** is built-in; **guardrails** let you validate inputs/outputs; **handoffs** let agents delegate. ([OpenAI GitHub][2], [GitHub][3])

---

# Advantages & disadvantages (in practice)

## LangGraph

**Pros**

* **Explicit orchestration**: complex branching, subgraphs, parallel fan-out/fan-in with **reducers**—easy to test and reason about. ([LangChain][4])
* **Durable persistence**: checkpoints per step + **threads** → resume, rewind, **time-travel**, human-in-the-loop interrupts; robust for long jobs. ([LangChain][1])
* **Provider-agnostic**: swap models/tooling freely.
* **Composability**: graphs as building blocks; good for multi-agent “systems.”

**Cons**

* More plumbing (state schema, reducers, edges); steeper learning curve than an agent loop.
* You own more ops (where to host, persistence backend) unless you adopt LangGraph’s hosted platform.
* For quick, simple chat-with-tools, it’s “more than you need.”

## OpenAI SDK (Agents SDK / Responses)

**Pros**

* **Minimal primitives** (Agents, Handoffs, Guardrails, Sessions) with **built-in tracing**—fast to production. ([OpenAI GitHub][2])
* **Hosted tools** (web search, file search, code interpreter) are one-line enablements; plus MCP & custom function tools. ([OpenAI Platform][10])
* **Realtime/voice** is first-class; excellent for voice agents and interactive UIs. ([OpenAI Platform][8])
* Multi-agent handoffs pattern = clean delegation; supports non-OpenAI models via LiteLLM. ([GitHub][3])

**Cons**

* Control flow is **implicit**; complicated branching/parallelism requires custom code or extra patterns (can get messy vs a graph).
* **Sessions** ≠ per-step checkpoints; you don’t get built-in time-travel or state editing at each super-step.
* Deeper tie-in to OpenAI platform if you rely on hosted tools (milder with LiteLLM, but still).

---

# How to choose (rule-of-thumb)

* Pick **LangGraph** when you need **explicit workflows** with:
  multi-stage pipelines (RAG+tools+validators), **parallel** branches, retries with counters, **HITL approvals**, **durable** runs you can pause/resume/audit. ([LangChain][1])
* Pick the **OpenAI Agents SDK** (or Responses+Tools) when you need **fast delivery** of:
  chat/voice agents, small multi-agent systems with handoffs, strong **OpenAI hosted tools** integration, platform **tracing** and team visibility. ([OpenAI GitHub][2])

---

# “How to use both” (integration patterns)

### Pattern 1: LangGraph orchestrator → call an OpenAI Agent from a node

```python
# inside a LangGraph node
from agents import Agent, Runner

def run_research_agent(state):
    agent = Agent(name="Researcher", instructions="Do web research, cite sources.")
    result = Runner.run_sync(agent, input=state["question"])
    return {"research": result.final_output}
```

### Pattern 2: OpenAI Agent → handoff to a LangGraph subgraph as a “tool”

Expose your LangGraph subgraph behind a small function tool (HTTP or local call) and register it in the Agent’s tool list. The Agent “uses a tool,” you execute the subgraph and return the result to the Agent loop.

---

# Common build recipes

## 1) “Serious” multi-agent research system

* **LangGraph** manages: planner → parallel researchers → aggregator → grader → reporter (reducers accumulate drafts; checkpoints let you resume). ([LangChain][4])
* Inside “researcher” node, call **OpenAI Agent** with hosted **web search tool** for breadth, then return to the graph. ([OpenAI Platform][10])

## 2) Voice concierge with workflows

* **OpenAI Agents SDK** for realtime voice, TTS/STT, web/file search; **handoff** to a “booking” Agent. ([OpenAI Platform][8])
* For the booking step, call a **LangGraph** subgraph that performs validations, retries, and human approval.

---

# Minimal code starters

## LangGraph (prebuilt ReAct agent + memory + structured output)

```python
from pydantic import BaseModel
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

class Answer(BaseModel):
    answer: str
    sources: list[str]

def web_search(q: str) -> list[str]: ...
checkpointer = InMemorySaver()

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[web_search],
    prompt="Be precise. Always include sources.",
    response_format=Answer,
    checkpointer=checkpointer,
)

agent.invoke(
    {"messages": [{"role": "user", "content": "Latest on lithium prices?"}]},
    {"configurable": {"thread_id": "u-1"}}
)
```

* Prebuilt agent setup & memory details are documented in the LangGraph quickstart. ([LangChain][6])

## OpenAI Agents SDK (router + tool + guardrail + session)

```python
from agents import Agent, Runner, function_tool, InputGuardrail, GuardrailFunctionOutput, SQLiteSession
from pydantic import BaseModel

@function_tool
def get_price(commodity: str) -> str: ...

class QueryCheck(BaseModel):
    is_supported: bool
    reason: str

guard = Agent(
    name="Guard",
    instructions="Return is_supported true only for ['lithium','copper']",
    output_type=QueryCheck,
)

async def guardrail(ctx, agent, input_data):
    r = await Runner.run(guard, input_data, context=ctx.context)
    qc = r.final_output_as(QueryCheck)
    return GuardrailFunctionOutput(output_info=qc, tripwire_triggered=not qc.is_supported)

router = Agent(
    name="Router",
    instructions="Answer directly or call tools.",
    tools=[get_price],
    input_guardrails=[InputGuardrail(guardrail_function=guardrail)],
)

session = SQLiteSession("u-1")
result = Runner.run_sync(router, "lithium price today", session=session)
print(result.final_output)
```

* Shows **tools**, **guardrails**, **sessions**, and the **agent loop** model. ([OpenAI GitHub][2])

---

# Notes on platform features & docs

* **LangGraph:**

  * Prebuilt agents & quickstart (create\_react\_agent, memory, structured output). ([LangChain][6])
  * Persistence model (threads, checkpoints, replay/time-travel, store). ([LangChain][1])

* **OpenAI SDK:**

  * **Agents SDK**: primitives (Agents, Handoffs, Guardrails, Sessions) + examples & tracing. ([OpenAI GitHub][2], [GitHub][3])
  * **Responses API + Tools**: tool/function calling; web/file search & code-interpreter as hosted tools; realtime/voice guides. ([OpenAI Platform][9])

---

## Bottom line

* Reach for **LangGraph** when your agent **is a system**—a graph with branching, retries, evaluators, and durable state.
* Reach for the **OpenAI Agents SDK** (or Responses+Tools) when your agent **is a product feature**—a capable loop with tools, guardrails, and rich platform integrations (tracing, web/file search, code/voice), delivered quickly.

If you want, I can take one of your live use-cases (e.g., your Reg-K chatbot pipeline or your contract analyzer) and sketch the exact LangGraph nodes/edges and the Agents SDK pieces that fit best.

[1]: https://langchain-ai.github.io/langgraph/concepts/persistence/ "Overview"
[2]: https://openai.github.io/openai-agents-python/quickstart/ "Quickstart - OpenAI Agents SDK"
[3]: https://github.com/openai/openai-agents-python "GitHub - openai/openai-agents-python: A lightweight, powerful framework for multi-agent workflows"
[4]: https://langchain-ai.github.io/langgraph/reference/graphs/?utm_source=chatgpt.com "Graphs - GitHub Pages"
[5]: https://langchain-ai.github.io/langgraph/concepts/low_level/?utm_source=chatgpt.com "state graph node - GitHub Pages"
[6]: https://langchain-ai.github.io/langgraph/agents/agents/ "Start with a prebuilt agent"
[7]: https://openai.github.io/openai-agents-python/?utm_source=chatgpt.com "OpenAI Agents SDK"
[8]: https://platform.openai.com/docs/guides/realtime?utm_source=chatgpt.com "Realtime API"
[9]: https://platform.openai.com/docs/api-reference/responses?utm_source=chatgpt.com "API Reference"
[10]: https://platform.openai.com/docs/guides/agents?utm_source=chatgpt.com "Agents - OpenAI API"
