"""
A2A Capability Cards — Google A2A protocol descriptors for each agent.
Each agent exposes a standard capability card so other agents can discover and delegate to it.
"""
from pydantic import BaseModel
from typing import Any


class InputSchema(BaseModel):
    type: str = "object"
    properties: dict[str, Any]
    required: list[str] = []


class OutputSchema(BaseModel):
    type: str = "object"
    properties: dict[str, Any]


class CapabilityCard(BaseModel):
    agent_id: str
    name: str
    description: str
    version: str = "1.0.0"
    protocol: str = "a2a/1.0"
    endpoint: str
    model: str
    skills: list[str]
    input_schema: InputSchema
    output_schema: OutputSchema
    metadata: dict[str, Any] = {}


CAPABILITY_CARDS: dict[str, CapabilityCard] = {
    "orchestrator": CapabilityCard(
        agent_id="orchestrator",
        name="Orchestrator Agent",
        description="Plans and coordinates multi-agent data pipelines. Decomposes tasks into DAGs and delegates to specialist agents.",
        endpoint="/a2a/orchestrator",
        model="claude-opus-4-5",
        skills=["dag_planning", "task_delegation", "output_validation"],
        input_schema=InputSchema(
            properties={
                "task": {"type": "string", "description": "Natural language task description"},
                "file_ids": {"type": "array", "items": {"type": "string"}},
                "cognitive_ctx": {"type": "object"},
            },
            required=["task"],
        ),
        output_schema=OutputSchema(
            properties={
                "plan": {"type": "object"},
                "status": {"type": "string"},
                "result": {"type": "object"},
            }
        ),
    ),
    "ingestion": CapabilityCard(
        agent_id="ingestion",
        name="Ingestion Agent",
        description="Reads and classifies Excel/CSV files. Extracts schema, sheet structure, and raw data manifests.",
        endpoint="/a2a/ingestion",
        model="ollama/qwen2.5-coder:32b",
        skills=["excel_parsing", "csv_parsing", "schema_extraction", "sheet_classification"],
        input_schema=InputSchema(
            properties={
                "file_ids": {"type": "array", "items": {"type": "string"}},
                "cognitive_ctx": {"type": "object"},
            },
            required=["file_ids"],
        ),
        output_schema=OutputSchema(
            properties={
                "file_manifests": {"type": "array"},
            }
        ),
    ),
    "understanding": CapabilityCard(
        agent_id="understanding",
        name="Understanding Agent",
        description="Infers column semantics, resolves aliases against data dictionary, validates schema completeness.",
        endpoint="/a2a/understanding",
        model="claude-sonnet-4-6",
        skills=["semantic_inference", "alias_resolution", "schema_validation"],
        input_schema=InputSchema(
            properties={
                "file_manifests": {"type": "array"},
                "cognitive_ctx": {"type": "object"},
            },
            required=["file_manifests"],
        ),
        output_schema=OutputSchema(
            properties={
                "enriched_manifests": {"type": "array"},
                "semantic_maps": {"type": "object"},
            }
        ),
    ),
    "transformation": CapabilityCard(
        agent_id="transformation",
        name="Transformation Agent",
        description="Generates and executes pandas/DuckDB code in Docker sandbox to transform and aggregate data.",
        endpoint="/a2a/transformation",
        model="ollama/qwen2.5-coder:32b",
        skills=["pandas_codegen", "sql_codegen", "data_transformation", "docker_execution"],
        input_schema=InputSchema(
            properties={
                "file_manifests": {"type": "array"},
                "business_rules": {"type": "string"},
                "cognitive_ctx": {"type": "object"},
            },
            required=["file_manifests"],
        ),
        output_schema=OutputSchema(
            properties={
                "team_parquets": {"type": "object"},
                "merged_path": {"type": "string"},
                "transformation_code": {"type": "string"},
            }
        ),
    ),
    "validation": CapabilityCard(
        agent_id="validation",
        name="Validation Agent",
        description="Cross-checks totals, flags anomalies, runs data quality rules in Docker sandbox.",
        endpoint="/a2a/validation",
        model="ollama/llama3.3:70b",
        skills=["total_verification", "anomaly_detection", "data_quality"],
        input_schema=InputSchema(
            properties={
                "merged_path": {"type": "string"},
                "original_manifests": {"type": "array"},
                "business_rules": {"type": "string"},
            },
            required=["merged_path"],
        ),
        output_schema=OutputSchema(
            properties={
                "passed": {"type": "boolean"},
                "checks": {"type": "array"},
                "anomalies": {"type": "array"},
                "report": {"type": "object"},
            }
        ),
    ),
    "visualization": CapabilityCard(
        agent_id="visualization",
        name="Visualization Agent",
        description="Generates Apache ECharts configurations from validated sandbox output data.",
        endpoint="/a2a/visualization",
        model="claude-sonnet-4-6",
        skills=["echarts_config", "chart_type_selection", "data_accuracy_validation"],
        input_schema=InputSchema(
            properties={
                "data_summary": {"type": "object"},
                "chart_requests": {"type": "array"},
                "cognitive_ctx": {"type": "object"},
            },
            required=["data_summary"],
        ),
        output_schema=OutputSchema(
            properties={
                "charts": {"type": "array"},
            }
        ),
    ),
    "report": CapabilityCard(
        agent_id="report",
        name="Report Agent",
        description="Assembles interactive narrative reports with charts, executive summaries, and Excel export.",
        endpoint="/a2a/report",
        model="claude-opus-4-5",
        skills=["narrative_generation", "report_assembly", "excel_export"],
        input_schema=InputSchema(
            properties={
                "validation": {"type": "object"},
                "charts": {"type": "array"},
                "data_summary": {"type": "object"},
                "cognitive_ctx": {"type": "object"},
            },
            required=["validation", "charts"],
        ),
        output_schema=OutputSchema(
            properties={
                "report": {"type": "object"},
                "excel_bytes": {"type": "string"},
            }
        ),
    ),
    "memory": CapabilityCard(
        agent_id="memory",
        name="Memory Agent",
        description="Persists schema cache, column mappings, and corrections after each pipeline run.",
        endpoint="/a2a/memory",
        model="ollama/llama3.3:8b",
        skills=["schema_caching", "mapping_persistence", "correction_logging"],
        input_schema=InputSchema(
            properties={
                "file_manifests": {"type": "array"},
                "semantic_maps": {"type": "object"},
                "corrections": {"type": "array"},
            },
            required=[],
        ),
        output_schema=OutputSchema(
            properties={
                "updated_files": {"type": "array"},
            }
        ),
    ),
}


def get_capability_card(agent_id: str) -> CapabilityCard | None:
    return CAPABILITY_CARDS.get(agent_id)


def list_capability_cards() -> list[CapabilityCard]:
    return list(CAPABILITY_CARDS.values())
