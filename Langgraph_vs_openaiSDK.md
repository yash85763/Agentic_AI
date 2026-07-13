The key insight: your LangGraph/LangChain/deepagents logic is just a **function that takes text and returns text**. A2A is only a transport/discovery wrapper around it. So you build the agent core once, then wrap it in whichever A2A server you want. You cannot run one process that is simultaneously "python_a2a-native" and "a2a-sdk-native" — they're different server frameworks — but you have three clean options, and I'll show all of them plus the shared core.

## Architecture

```
┌─────────────────────────────┐
│   Agent core (framework)    │   ← LangGraph / LangChain / deepagents
│   run(text) -> text         │      knows nothing about A2A
└─────────────────────────────┘
              ▲
     ┌────────┴────────┐
     │                 │
┌──────────┐     ┌──────────┐
│ a2a-sdk  │     │python_a2a│    ← thin wrappers, pick one per port
│  :9999   │     │  :5000   │
└──────────┘     └──────────┘
```

Both wrappers serve a discoverable agent card and speak the same wire protocol, so any orchestrator (using either client library) can call either endpoint.

## Step 0 — The shared agent core

This is the only place your framework choice matters. Keep it in `weather_core.py` and keep it framework-native. Below is a LangGraph version with a real tool; swap in LangChain or deepagents identically — all that matters is exposing a single `run_weather_agent(text) -> str`.

```python
# weather_core.py
import os
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    # replace with a real API call (OpenWeather, weather.gov, etc.)
    return f"It's sunny and 25°C in {location}."

_model = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
_agent = create_react_agent(_model, tools=[get_weather])

def run_weather_agent(user_text: str) -> str:
    """Framework-agnostic entrypoint. Text in, text out."""
    result = _agent.invoke({"messages": [{"role": "user", "content": user_text}]})
    return result["messages"][-1].content
```

For **deepagents** the body would be `agent = create_deep_agent(tools=[get_weather], instructions=...)` and `agent.invoke({"messages":[...]})`. For **plain LangChain** it's an `AgentExecutor.invoke({"input": user_text})["output"]`. The wrappers below never change.

---

## Option A — Wrap with `a2a-sdk` (official)

`server_a2a_sdk.py`:

```python
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message
from a2a.types import AgentCard, AgentSkill, AgentCapabilities

from weather_core import run_weather_agent   # <-- shared core

class WeatherExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        user_text = context.get_user_input()
        result = run_weather_agent(user_text)          # call LangGraph
        await event_queue.enqueue_event(new_agent_text_message(result))

    async def cancel(self, context, event_queue):
        raise Exception("cancel not supported")

skill = AgentSkill(
    id="get_weather", name="Get Weather",
    description="Get current weather for a location",
    tags=["weather"], examples=["weather in Tokyo"],
)
card = AgentCard(
    name="Weather Agent (LangGraph)",
    description="LangGraph weather agent over A2A",
    url="http://YOUR_EC2_IP:9999/",
    version="1.0.0",
    default_input_modes=["text"], default_output_modes=["text"],
    capabilities=AgentCapabilities(streaming=True),
    skills=[skill],
)

if __name__ == "__main__":
    handler = DefaultRequestHandler(
        agent_executor=WeatherExecutor(), task_store=InMemoryTaskStore()
    )
    app = A2AStarletteApplication(agent_card=card, http_handler=handler)
    uvicorn.run(app.build(), host="0.0.0.0", port=9999)
```

Card served at `/.well-known/agent-card.json`.

---

## Option B — Wrap with `python_a2a` (community)

`server_python_a2a.py`:

```python
from python_a2a import A2AServer, agent, skill, run_server, TaskStatus, TaskState
from weather_core import run_weather_agent   # <-- same shared core

@agent(
    name="Weather Agent (LangGraph)",
    description="LangGraph weather agent over A2A",
    version="1.0.0",
    url="http://YOUR_EC2_IP:5000",
)
class WeatherAgent(A2AServer):
    @skill(name="Get Weather", description="Get current weather", tags=["weather"])
    def get_weather(self, location):
        return run_weather_agent(f"weather in {location}")

    def handle_task(self, task):
        text = task.message.get("content", {}).get("text", "")
        result = run_weather_agent(text)               # call LangGraph
        task.artifacts = [{"parts": [{"type": "text", "text": result}]}]
        task.status = TaskStatus(state=TaskState.COMPLETED)
        return task

if __name__ == "__main__":
    run_server(WeatherAgent(), host="0.0.0.0", port=5000)
```

Card served at `/.well-known/agent.json`.

---

## The "both compatible" question — three real answers

**1. You don't actually need two servers.** Both `python_a2a` and `a2a-sdk` *clients* speak the same A2A JSON-RPC protocol. An orchestrator built on `python_a2a` can usually call an `a2a-sdk` server and vice versa, because compatibility lives in the wire protocol, not the library. So pick **one** server (I'd pick `a2a-sdk` for spec fidelity) and most orchestrators of either flavor can reach it. Test this first — it may be all you need.

**2. If you truly want both native endpoints, run two processes** sharing the core, on two ports:

```bash
python server_a2a_sdk.py      # :9999, card at /.well-known/agent-card.json
python server_python_a2a.py   # :5000, card at /.well-known/agent.json
```

Because both import the same `weather_core`, there's one source of truth for logic. Each gets its own systemd unit and its own security-group port. This is the most robust way to be genuinely compatible with orchestrators hardwired to one library or one card path.

**3. One process, two mounted apps** (advanced). Both official apps are Starlette/ASGI; `python_a2a` runs on Flask. You can mount them under one ASGI server with a WSGI-to-ASGI bridge, but it's fiddly and version-sensitive. Not worth it versus option 2 — two small processes are cleaner.

## Deploy on EC2

```bash
sudo apt update && sudo apt install -y python3-pip
pip3 install a2a-sdk python-a2a uvicorn \
             langgraph langchain langchain-openai deepagents
# scp weather_core.py + both server files, set OPENAI_API_KEY
```

Two systemd services (one per port), for example `/etc/systemd/system/a2a-sdk.service`:

```ini
[Unit]
Description=Weather Agent (a2a-sdk)
After=network.target
[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu
Environment=OPENAI_API_KEY=sk-...
ExecStart=/usr/bin/python3 /home/ubuntu/server_a2a_sdk.py
Restart=always
[Install]
WantedBy=multi-user.target
```

Duplicate for `python_a2a` on port 5000. Then:

```bash
sudo systemctl enable --now a2a-sdk python-a2a
```

Security group: open inbound TCP **5000 and 9999** (or, better, front both with nginx + TLS and expose only 443 with path routing). Verify both cards:

```bash
curl http://YOUR_EC2_IP:9999/.well-known/agent-card.json
curl http://YOUR_EC2_IP:5000/.well-known/agent.json
```

## Calling it (either client works against either server)

`a2a-sdk` client:

```python
resolver = A2ACardResolver(httpx_client=hc, base_url="http://YOUR_EC2_IP:9999")
card = await resolver.get_agent_card()
client = A2AClient(httpx_client=hc, agent_card=card)
# send SendMessageRequest with parts=[{"kind":"text","text":"weather in Paris"}]
```

`python_a2a` client:

```python
from python_a2a import A2AClient
client = A2AClient("http://YOUR_EC2_IP:5000")
print(client.ask("weather in Paris"))
```

## What you own as the maker

- **One framework-native core** (`run_weather_agent`) — the only place LangGraph/LangChain/deepagents lives. Everything A2A is a thin adapter over it.
- **Accurate AgentCards** on each server: correct public `url`, matching skills, correct `.well-known` path per library.
- **Bind `0.0.0.0`**, open the ports, run under systemd, add TLS.
- **Manage secrets** (`OPENAI_API_KEY`) via environment/systemd, not in code.
- **Decide 1 vs 2 servers**: try one server + cross-library client first; run two processes only if an orchestrator demands a specific library or card path.

Recommended path: build the core, ship the `a2a-sdk` server first, confirm whether your `python_a2a`-based orchestrator can already reach it. Only stand up the second process if it can't.

Want me to replace the stub `get_weather` with a real weather API (weather.gov is free, no key) and give you the full working core + both servers as downloadable files?