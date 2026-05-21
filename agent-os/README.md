# AgentOS — Open Source Edition

**Agentic Data Platform · Free Stack · Self-Hosted · LLM-Agnostic**

AgentOS is a self-hosted agentic platform for data automation. Use any LLM you choose — Claude, OpenAI, DeepSeek, Llama — through a single unified interface. Every other layer of the stack is free and open source.

---

## Quick Start

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env with your LLM API keys (or skip — use Ollama for fully local)

# 2. Build the sandbox image (required, used to execute agent-generated code)
bash sandbox-image/build.sh

# 3. Start the full stack
docker compose up -d

# 4. (Optional) Pull Ollama models for local LLM workers
docker exec -it agent-os-ollama-1 ollama pull qwen2.5-coder:32b
docker exec -it agent-os-ollama-1 ollama pull llama3.3:70b

# 5. Open the apps
# - React UI:      http://localhost:3000
# - Chainlit chat: http://localhost:8001
# - Backend API:   http://localhost:8000
# - Langfuse:      http://localhost:3001
# - MinIO console: http://localhost:9001
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER                                       │
└─────────────────────────────────────────────────────────────────────┘
                ▲                                ▲
                │ SSE/WS                         │ Chat
┌──────────────────────────┐         ┌─────────────────────────┐
│  React + ECharts UI      │         │   Chainlit UI            │
└──────────────────────────┘         └─────────────────────────┘
                ▲                                ▲
                └──────── FastAPI ──────────────┘
                              │
       ┌──────────────────────┼──────────────────────┐
       ▼                      ▼                      ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  LangGraph   │    │  Cognitive FS    │    │   Langfuse        │
│  StateGraph  │    │ (soul/skills/    │    │  observability    │
│              │    │  knowledge/      │    │                   │
│  Agents:     │    │  memory)         │    └──────────────────┘
│  • Orchestr. │    └──────────────────┘
│  • Ingestion │
│  • Understnd │             ▼
│  • Transform │    ┌──────────────────┐
│  • Validate  │    │     LiteLLM       │
│  • Visualize │    │  (any provider)   │
│  • Report    │    └──────────────────┘
│  • Memory    │    ┌───────┬──────┬─────────┐
└──────────────┘    ▼       ▼      ▼         ▼
       │          Claude  OpenAI Ollama  DeepSeek
       │
       ▼
┌────────────────────────┐
│  Docker Sandbox Pool   │ ← every code-exec call lands here
│  (isolated, no net)    │
└────────────────────────┘
       │
       ▼
┌────────────────────────┐
│   MinIO + Postgres     │
└────────────────────────┘
```

---

## What's Inside

```
agent-os/
├── agent-config/           # Cognitive filesystem (user-editable MD)
│   ├── soul.md             # Agent identity & values
│   ├── skills/             # How-to skills (excel, formulas, cleaning, charts, sql)
│   ├── knowledge/          # Domain knowledge (data dict, business rules, teams)
│   ├── memory/             # Auto-updated cache + human corrections
│   └── pipelines/          # Pipeline definitions
├── backend/                # FastAPI + LangGraph + LiteLLM
│   ├── main.py             # API entry point
│   ├── cognitive_fs.py     # Cognitive filesystem loader
│   ├── agents/             # 8 specialist agents
│   ├── pipelines/          # LangGraph DAGs
│   ├── tools/              # Docker sandbox executor, DuckDB, MinIO
│   ├── streaming/          # SSE manager + event models
│   └── a2a/                # Google A2A protocol capability cards
├── sandbox-image/          # Dockerfile for safe code execution
├── frontend/               # React + TypeScript + ECharts + Tailwind
│   └── src/
│       ├── pages/          # Dashboard, Workspace, ReportViewer, Studio
│       ├── components/     # AgentFeed, EChartRenderer, DataTable, FileExplorer, MonacoEditor
│       └── hooks/          # useJobStream, useCognitiveFS
├── chainlit_app.py         # Chainlit chat interface
└── docker-compose.yml      # Full stack
```

---

## LLM Strategy

AgentOS uses [LiteLLM](https://github.com/BerriAI/litellm) as a universal LLM proxy. Switch providers by changing one string in `.env` — no code changes:

```bash
# .env
ORCHESTRATOR_MODEL=claude-opus-4-5          # or deepseek/deepseek-r1, openai/gpt-4o
WORKER_MODEL=ollama/qwen2.5-coder:32b       # free, local, great at code
VALIDATOR_MODEL=ollama/llama3.3:70b         # free, local
REPORT_MODEL=claude-sonnet-4-6              # or openai/gpt-4o
MEMORY_MODEL=ollama/llama3.3:8b             # free, local, tiny
```

### Recommended Model Mix

| Agent | Recommended | Why |
|---|---|---|
| Orchestrator | Claude Opus / DeepSeek R1 | Extended thinking, complex planning |
| Ingestion | Qwen2.5-coder (Ollama) | Code gen, free, local |
| Understanding | Claude Sonnet / GPT-4o | Schema reasoning |
| Transformation | Qwen2.5-coder (Ollama) | Pandas/SQL code, free, local |
| Validation | Llama3.3:70b (Ollama) | Rule checking, fully local |
| Visualization | Claude Sonnet / GPT-4o | Chart config quality |
| Report | Claude Opus / GPT-4o | Narrative quality |
| Memory | Llama3.3:8b (Ollama) | Simple writes, free |

Worker agents on Ollama save 70%+ of LLM costs vs. all-paid setups.

---

## Cognitive Filesystem

The agent's brain lives in editable markdown files. Update business rules, formulas, or agent behavior **without touching code**.

| File | Purpose | Edited By |
|---|---|---|
| `soul.md` | Agent identity, values | Platform admin |
| `skills/*.md` | How-to skills | Tech team |
| `knowledge/*.md` | Domain knowledge | Business users |
| `memory/corrections.md` | Mistake corrections | Any user |
| `memory/*.json` | Auto-learned cache | Memory Agent only |
| `pipelines/*.md` | Pipeline DAG definitions | Tech team |

The Studio UI (http://localhost:3000/studio) provides a Monaco editor over this filesystem with live context preview.

---

## Security

Code execution always happens inside a Docker sandbox with:

- `network_disabled=True` — no internet access
- `mem_limit=512m` — memory cap
- `nano_cpus=1_000_000_000` — CPU cap (1 core)
- Read-only mounts for source files
- Non-root `sandbox` user
- 30-second wall-clock timeout
- Auto-removal on completion

The sandbox cannot reach the host filesystem, the database, or external services.

---

## Observability

Langfuse (self-hosted) captures every:
- Pipeline trace (input, output, total cost, duration)
- Agent step (prompt, response, tokens, latency)
- Tool call (inputs, outputs, errors)
- Sandbox execution (code, stdout, exit code)
- LLM call (cost per token, per model)

Visit `http://localhost:3001` after first run.

---

## Building Sandbox Image

```bash
# One-time build
docker build -t agent-sandbox:latest ./sandbox-image

# Or use the convenience script
bash sandbox-image/build.sh
```

---

## Development

### Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev   # http://localhost:3000
```

### Chainlit
```bash
chainlit run chainlit_app.py --port 8001
```

---

## Cost Model

For a team processing ~10 jobs/day with the recommended hybrid mix:

| Component | Monthly Cost |
|---|---|
| Infrastructure (self-hosted) | $0 |
| Observability (Langfuse) | $0 |
| Charts (Apache ECharts) | $0 |
| Agent UI (Chainlit) | $0 |
| Worker LLM calls (Ollama, local) | $0 |
| Orchestrator + Report LLM (Claude/GPT) | $5 - $30 |
| **Total** | **$5 - $30** |

---

## License

MIT for all original code. Component licenses respected (see individual library docs).

---

## Build Phases (for contributors)

See `BUILD_PHASES.md` for the recommended implementation order. Each phase is independently testable.
