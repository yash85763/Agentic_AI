"""
Pydantic v2 event models for the AgentOS SSE streaming layer.

Every event that flows from an agent node to the frontend is represented
by one of the concrete model classes below.  All events share the common
AgentEvent envelope; the `data` field carries the typed payload.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Event type enumeration
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    THINKING = "thinking"
    CODE_GENERATED = "code_generated"
    CODE_EXECUTING = "code_executing"
    CODE_RESULT = "code_result"
    CHART_READY = "chart_ready"
    VALIDATION = "validation"
    REPORT_SECTION = "report_section"
    COMPLETE = "complete"
    ERROR = "error"
    AGENT_START = "agent_start"
    AGENT_COMPLETE = "agent_complete"
    PROGRESS = "progress"


# ---------------------------------------------------------------------------
# Shared envelope
# ---------------------------------------------------------------------------


class AgentEvent(BaseModel):
    """
    Top-level SSE envelope.  Every event the backend emits is an AgentEvent.

    Attributes
    ----------
    event_id:
        UUID-4 string, auto-generated if omitted.
    job_id:
        Identifies the analysis job this event belongs to.
    event_type:
        Discriminator used by the frontend to select the correct renderer.
    agent_name:
        Human-readable name of the agent/node that emitted this event.
    data:
        Typed payload — one of the concrete *Event models below.
    timestamp:
        UTC timestamp (auto-set to now if omitted).
    metadata:
        Free-form key/value bag for tracing, versioning, etc.
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str
    event_type: EventType
    agent_name: str
    data: Any
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("event_id", mode="before")
    @classmethod
    def _ensure_event_id(cls, v: Any) -> str:
        if not v:
            return str(uuid.uuid4())
        return str(v)

    def to_sse(self) -> str:
        """Serialise to SSE wire format (``event:\\ndata:\\n\\n``)."""
        payload = self.model_dump_json()
        return f"event: {self.event_type.value}\ndata: {payload}\n\n"


# ---------------------------------------------------------------------------
# Concrete payload models
# ---------------------------------------------------------------------------


class ThinkingEvent(BaseModel):
    """
    Emitted while the agent is reasoning (e.g. building a plan).

    Attributes
    ----------
    thought:
        The free-text reasoning trace.
    step:
        Optional step name / number within the agent's plan.
    """

    thought: str
    step: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class CodeGeneratedEvent(BaseModel):
    """
    Emitted once the agent has produced code (before execution).

    Attributes
    ----------
    language:
        "python" | "sql" | …
    code:
        Source code as a string.
    description:
        One-line explanation of what the code does.
    """

    language: str = "python"
    code: str
    description: Optional[str] = None
    estimated_duration_ms: Optional[int] = None


class CodeExecutingEvent(BaseModel):
    """
    Emitted when execution starts (useful for showing a spinner).
    """

    language: str = "python"
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class CodeResultEvent(BaseModel):
    """
    Emitted once code execution completes.

    Attributes
    ----------
    exit_code:
        0 = success.
    stdout / stderr:
        Captured output streams.
    duration_ms:
        Wall-clock execution time in milliseconds.
    output_files:
        {filename: base64-encoded bytes} for any files written to
        /sandbox/outputs (Python sandbox) or returned by DuckDB.
    timed_out:
        True if the sandbox killed the process due to timeout.
    """

    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: int
    output_files: dict[str, str] = Field(default_factory=dict)
    timed_out: bool = False
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class EChartsOption(BaseModel):
    """
    Minimal wrapper around an Apache ECharts option object.
    The full JSON is stored verbatim in ``option``; we only add
    a few typed fields for routing / display purposes.
    """

    title: Optional[str] = None
    chart_type: Optional[str] = None  # "bar", "line", "pie", …
    option: dict[str, Any]  # full ECharts config


class ChartReadyEvent(BaseModel):
    """
    Emitted when a chart is ready for rendering.

    The ``echarts_config`` field contains the complete ECharts option
    object that the frontend can pass directly to ``echarts.init()``
    and ``chart.setOption()``.
    """

    chart_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: Optional[str] = None
    description: Optional[str] = None
    echarts_config: dict[str, Any]  # verbatim ECharts option
    thumbnail_url: Optional[str] = None  # optional server-rendered PNG


class ChecklistItem(BaseModel):
    name: str
    passed: bool
    message: Optional[str] = None
    severity: str = "error"  # "error" | "warning" | "info"


class ValidationEvent(BaseModel):
    """
    Pass/fail checklist emitted after the Validation node.

    Attributes
    ----------
    passed:
        True only when ALL items with severity="error" have passed=True.
    items:
        Individual validation checks.
    retry_recommended:
        True when the pipeline should attempt a retry.
    """

    passed: bool
    items: list[ChecklistItem]
    retry_recommended: bool = False
    summary: Optional[str] = None

    @model_validator(mode="after")
    def _compute_passed(self) -> "ValidationEvent":
        # Auto-compute passed from items if not explicitly set
        error_items = [i for i in self.items if i.severity == "error"]
        if error_items:
            self.passed = all(i.passed for i in error_items)
        return self


class ReportSectionEvent(BaseModel):
    """
    Streamed incrementally as the report is being composed.

    Attributes
    ----------
    section_id:
        Stable identifier for ordering (e.g. "executive_summary").
    title:
        Display title.
    content:
        Markdown or HTML content.
    is_final:
        True on the last chunk of this section.
    order:
        Numeric sort key for client-side ordering.
    """

    section_id: str
    title: str
    content: str
    is_final: bool = True
    order: int = 0
    content_type: str = "markdown"  # "markdown" | "html"


class CompleteEvent(BaseModel):
    """
    Emitted when the entire pipeline finishes successfully.

    Attributes
    ----------
    report_url:
        URL to the final report artifact in MinIO / object storage.
    artifact_urls:
        URLs of any other generated artefacts (charts, parquets, …).
    duration_ms:
        Total wall-clock duration of the pipeline run.
    """

    report_url: Optional[str] = None
    artifact_urls: list[str] = Field(default_factory=list)
    duration_ms: int
    summary: Optional[str] = None


class ErrorEvent(BaseModel):
    """
    Emitted on unrecoverable pipeline failures.

    Attributes
    ----------
    message:
        Human-readable error message.
    error_type:
        Python exception class name.
    traceback:
        Full traceback string (omitted in production for security).
    node:
        Agent/node name where the error occurred.
    retryable:
        Hint to the frontend / orchestrator.
    """

    message: str
    error_type: Optional[str] = None
    traceback: Optional[str] = None
    node: Optional[str] = None
    retryable: bool = False


class AgentStartEvent(BaseModel):
    """Emitted when an agent node begins processing."""

    node_name: str
    description: Optional[str] = None
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class AgentCompleteEvent(BaseModel):
    """Emitted when an agent node finishes (success or skip)."""

    node_name: str
    duration_ms: int
    success: bool = True
    skipped: bool = False


class ProgressEvent(BaseModel):
    """
    Generic progress indicator.

    Attributes
    ----------
    pct:
        0–100 completion percentage.
    message:
        Short status message for the UI.
    step / total_steps:
        For step-based progress bars.
    """

    pct: float = Field(ge=0.0, le=100.0)
    message: str
    step: Optional[int] = None
    total_steps: Optional[int] = None


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def make_event(
    job_id: str,
    event_type: EventType,
    agent_name: str,
    data: Any,
    metadata: dict[str, Any] | None = None,
) -> AgentEvent:
    """Construct an AgentEvent with a fresh event_id and UTC timestamp."""
    return AgentEvent(
        job_id=job_id,
        event_type=event_type,
        agent_name=agent_name,
        data=data,
        metadata=metadata or {},
    )


# Convenience constructors for common event types

def thinking(job_id: str, agent: str, thought: str, step: str | None = None) -> AgentEvent:
    return make_event(job_id, EventType.THINKING, agent, ThinkingEvent(thought=thought, step=step))


def code_generated(job_id: str, agent: str, code: str, language: str = "python", description: str | None = None) -> AgentEvent:
    return make_event(job_id, EventType.CODE_GENERATED, agent, CodeGeneratedEvent(code=code, language=language, description=description))


def code_result(job_id: str, agent: str, result: dict[str, Any]) -> AgentEvent:
    payload = CodeResultEvent(
        exit_code=result.get("exit_code", -1),
        stdout=result.get("stdout", ""),
        stderr=result.get("stderr", ""),
        duration_ms=result.get("duration_ms", 0),
        timed_out=result.get("timed_out", False),
    )
    return make_event(job_id, EventType.CODE_RESULT, agent, payload)


def progress(job_id: str, agent: str, pct: float, message: str) -> AgentEvent:
    return make_event(job_id, EventType.PROGRESS, agent, ProgressEvent(pct=pct, message=message))


def error_event(job_id: str, agent: str, message: str, node: str | None = None, retryable: bool = False) -> AgentEvent:
    return make_event(job_id, EventType.ERROR, agent, ErrorEvent(message=message, node=node, retryable=retryable))
