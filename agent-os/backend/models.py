"""
Pydantic v2 models and TypedDicts for AgentOS backend.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional, TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Job models
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    complete = "complete"
    failed = "failed"


class JobCreate(BaseModel):
    task_description: str = Field(..., min_length=1, description="Natural-language description of the task")
    file_ids: list[str] = Field(default_factory=list, description="List of FileRecord IDs to attach")


class JobResponse(BaseModel):
    id: str
    status: JobStatus
    task_description: str
    file_ids: list[str]
    result: Optional[dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Agent event (streamed over WebSocket / SSE)
# ---------------------------------------------------------------------------

class AgentEvent(BaseModel):
    type: str = Field(..., description="Event type, e.g. 'thought', 'tool_call', 'result', 'error'")
    job_id: str
    agent_name: str = Field(default="", description="Name of the agent that emitted this event")
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------

class FileUploadResponse(BaseModel):
    id: str
    original_name: str
    minio_path: str
    size: int
    content_type: str
    job_id: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# LangGraph pipeline state (TypedDict)
# ---------------------------------------------------------------------------

class PipelineState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""

    job_id: str
    task_description: str
    file_ids: list[str]
    user_id: str                         # tenant identifier for RLS
    cognitive_context: dict[str, Any]   # loaded from CognitiveFSLoader
    system_prompt: str
    raw_data: dict[str, Any]            # loaded/parsed file contents
    normalized_data: dict[str, Any]     # after normalization agent
    analysis_result: dict[str, Any]     # after analytics agent
    report: str                         # final markdown/text report
    errors: list[str]
    status: str                         # mirrors JobStatus values
    # Causal analysis gate (Section 4) — False by default; set True to enable
    causal_mode_enabled: bool
    # ConfidenceBundle dict attached to each output (Section 3)
    confidence_bundles: dict[str, Any]
    # Orchestrator routing decisions log (Section 6)
    routing_decisions: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# CognitiveFS models
# ---------------------------------------------------------------------------

class CognitiveFSFile(BaseModel):
    path: str = Field(..., description="Relative path inside agent-config/")
    content: str = Field(..., description="Raw file content")
    size: int = Field(..., description="File size in bytes")


class CognitiveFSTree(BaseModel):
    """Recursive directory tree returned by CognitiveFSLoader.list_files()."""

    root: str
    tree: dict[str, Any] = Field(
        ...,
        description=(
            "Nested dict where leaves are file metadata dicts with keys "
            "'path', 'size', 'modified_at'."
        ),
    )
