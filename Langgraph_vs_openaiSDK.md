Here's the full picture for building an A2A agent, deploying it on EC2, and letting other orchestrators call it.

## 1. Build the agent (maker side)

Install `python-a2a`:

```bash
pip install python-a2a
```

Create your agent server (`agent.py`):

```python
from python_a2a import A2AServer, AgentCard, AgentSkill, skill, agent, run_server, TaskStatus, TaskState

@agent(
    name="Weather Agent",
    description="Provides weather information",
    version="1.0.0",
    url="http://YOUR_EC2_PUBLIC_IP:5000"  # public endpoint
)
class WeatherAgent(A2AServer):

    @skill(
        name="Get Weather",
        description="Get current weather for a location",
        tags=["weather", "forecast"]
    )
    def get_weather(self, location):
        # your real logic / tool / model call here
        return f"It's sunny and 25°C in {location}."

    def handle_task(self, task):
        # extract the user's message text
        text = task.message.get("content", {}).get("text", "")
        location = text.replace("weather in", "").strip() or "your area"

        result = self.get_weather(location)

        task.artifacts = [{
            "parts": [{"type": "text", "text": result}]
        }]
        task.status = TaskStatus(state=TaskState.COMPLETED)
        return task

if __name__ == "__main__":
    run_server(WeatherAgent(), host="0.0.0.0", port=5000)
```

Key maker responsibilities:
- **Agent Card** — the `@agent` decorator auto-exposes a discovery document at `/.well-known/agent.json`. This is how orchestrators discover your agent's name, skills, and URL. Make sure the `url` is your reachable public address.
- **Skills** — declare each capability with `@skill` so it shows up in the card.
- **`handle_task`** — the core method that receives a task, does the work, sets artifacts, and marks the task `COMPLETED` (or `FAILED`/`INPUT_REQUIRED`).
- Bind to `0.0.0.0`, not `127.0.0.1`, so it's reachable externally.

## 2. Deploy on EC2

Launch and connect:
- Launch an EC2 instance (Amazon Linux or Ubuntu).
- **Security group**: add an inbound rule allowing TCP on your port (e.g. 5000) from the orchestrator's IP or `0.0.0.0/0` if public. This is the most common thing people forget.

On the instance:

```bash
sudo apt update && sudo apt install -y python3-pip
pip3 install python-a2a
# copy agent.py over (scp) then:
```

Run it persistently (don't just run in a shell that dies on logout). Use a systemd service:

```ini
# /etc/systemd/system/a2a-agent.service
[Unit]
Description=A2A Weather Agent
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu
ExecStart=/usr/bin/python3 /home/ubuntu/agent.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now a2a-agent
```

For production, front it with nginx + TLS (so the URL is `https://yourdomain`) rather than exposing raw port 5000. Verify from outside:

```bash
curl http://YOUR_EC2_PUBLIC_IP:5000/.well-known/agent.json
```

## 3. How another orchestrator calls it

Any A2A-compatible orchestrator just needs your agent's URL. With `python-a2a`:

```python
from python_a2a import A2AClient

client = A2AClient("http://YOUR_EC2_PUBLIC_IP:5000")

# discovery — reads the agent card
print(client.agent_card.name, client.agent_card.skills)

# send a task
response = client.ask("weather in Tokyo")
print(response)
```

If the orchestrator manages multiple agents, it can register yours in a router:

```python
from python_a2a import AgentNetwork, AIAgentRouter

network = AgentNetwork()
network.add("weather", "http://YOUR_EC2_PUBLIC_IP:5000")

agent = network.get_agent("weather")
result = agent.ask("weather in Paris")
```

The orchestrator does **not** need your code — only the reachable URL and the agent card it serves.

## What you (the maker) own vs. what the caller owns

| You (maker) | Caller / orchestrator |
|---|---|
| Implement `handle_task` and skills | Discovers you via agent card |
| Serve a valid agent card at `/.well-known/agent.json` | Sends tasks to your URL |
| Deploy on EC2, keep it running | Parses your returned artifacts |
| Open the security-group port / TLS | Handles retries, routing, chaining |
| Publish a stable, reachable public URL | — |

Your practical checklist: correct `url` in the agent card, bind to `0.0.0.0`, open the port in the security group, run under systemd, add HTTPS for anything real, and share the public URL with whoever's orchestrating.

One caveat: `python-a2a`'s exact class and method names (e.g. `TaskStatus`/`TaskState`, decorator arguments) have shifted across versions. If an import fails, check your installed version's docs — the structure above holds, but names may differ slightly. Want me to tailor this to a specific agent (a real tool, a model call, LangChain/MCP integration) rather than the weather stub?