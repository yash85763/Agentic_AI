"""
A2A Router — FastAPI routes exposing each agent via the Google A2A protocol.
Other agents (or external systems) can delegate tasks via standard HTTP.
"""
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Any

from .capability_cards import list_capability_cards, get_capability_card, CapabilityCard

router = APIRouter(prefix="/a2a", tags=["a2a"])


class A2ATask(BaseModel):
    task_id: str = ""
    agent_id: str
    input: dict[str, Any]
    callback_url: str | None = None
    metadata: dict[str, Any] = {}


class A2ATaskResult(BaseModel):
    task_id: str
    agent_id: str
    status: str
    output: dict[str, Any] | None = None
    error: str | None = None
    started_at: datetime
    completed_at: datetime | None = None


# In-memory task store (replace with Redis/DB in production)
_tasks: dict[str, A2ATaskResult] = {}


@router.get("/.well-known/agent.json")
async def agent_manifest():
    """A2A discovery endpoint — returns all agent capability cards."""
    return {
        "platform": "AgentOS",
        "version": "2.0",
        "protocol": "a2a/1.0",
        "agents": [card.model_dump() for card in list_capability_cards()],
    }


@router.get("/agents")
async def list_agents() -> list[CapabilityCard]:
    """List all registered agent capability cards."""
    return list_capability_cards()


@router.get("/agents/{agent_id}")
async def get_agent(agent_id: str) -> CapabilityCard:
    """Get capability card for a specific agent."""
    card = get_capability_card(agent_id)
    if not card:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return card


@router.post("/{agent_id}/tasks")
async def create_task(
    agent_id: str,
    task: A2ATask,
    background_tasks: BackgroundTasks,
) -> A2ATaskResult:
    """Submit a task to a specific agent asynchronously."""
    card = get_capability_card(agent_id)
    if not card:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    task_id = task.task_id or str(uuid.uuid4())
    result = A2ATaskResult(
        task_id=task_id,
        agent_id=agent_id,
        status="accepted",
        started_at=datetime.utcnow(),
    )
    _tasks[task_id] = result

    background_tasks.add_task(_execute_agent_task, task_id, agent_id, task.input)
    return result


@router.get("/{agent_id}/tasks/{task_id}")
async def get_task_status(agent_id: str, task_id: str) -> A2ATaskResult:
    """Poll task status."""
    result = _tasks.get(task_id)
    if not result or result.agent_id != agent_id:
        raise HTTPException(status_code=404, detail="Task not found")
    return result


async def _execute_agent_task(task_id: str, agent_id: str, input_data: dict) -> None:
    """Background execution of agent task. Imports agent dynamically to avoid circular deps."""
    try:
        _tasks[task_id].status = "running"

        # Dynamic agent dispatch
        agent_module_map = {
            "orchestrator": ("agents.orchestrator", "OrchestratorAgent"),
            "ingestion": ("agents.ingestion_agent", "IngestionAgent"),
            "understanding": ("agents.understanding_agent", "UnderstandingAgent"),
            "transformation": ("agents.transformation_agent", "TransformationAgent"),
            "validation": ("agents.validation_agent", "ValidationAgent"),
            "visualization": ("agents.visualization_agent", "VisualizationAgent"),
            "report": ("agents.report_agent", "ReportAgent"),
            "memory": ("agents.memory_agent", "MemoryAgent"),
        }

        if agent_id not in agent_module_map:
            raise ValueError(f"Unknown agent: {agent_id}")

        import importlib
        module_path, class_name = agent_module_map[agent_id]
        module = importlib.import_module(module_path)
        AgentClass = getattr(module, class_name)

        agent = AgentClass()
        output = await agent.run(input_data)

        _tasks[task_id].status = "completed"
        _tasks[task_id].output = output
        _tasks[task_id].completed_at = datetime.utcnow()

    except Exception as e:
        _tasks[task_id].status = "failed"
        _tasks[task_id].error = str(e)
        _tasks[task_id].completed_at = datetime.utcnow()
