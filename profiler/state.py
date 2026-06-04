"""
state.py — LangGraph State for the Data Profiler Agent

This TypedDict is the single shared object that flows through every node.
Each node reads from it and writes back to it. LangGraph merges writes
using the Annotated reducers (list fields use operator.add to accumulate).
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict


class DataSource(TypedDict):
    """Describes where the data comes from."""
    type: str                        # "csv" | "excel" | "sql"
    path: Optional[str]              # file path for CSV/Excel
    connection_string: Optional[str] # SQLAlchemy connection string
    table_name: Optional[str]        # specific table (SQL)
    sheet_name: Optional[str]        # specific sheet (Excel)
    knowledge_file: Optional[str]    # optional knowledge.md path


class ColumnProfile(TypedDict):
    name: str
    raw_dtype: str
    inferred_type: str               # integer | float | datetime | categorical | string | free_text | boolean | empty | unknown
    total_count: int
    null_count: int
    null_pct: float
    unique_count: int
    exemplars: list[str]
    exemplar_truncated: bool
    stats: dict[str, Any]
    flags: list[str]
    pattern_detected: Optional[str]  # email | uuid | phone_us | zip_us | iso_country | currency_code
    likely_concept: str              # resolved alias / semantic meaning


class RichProfile(TypedDict):
    source: dict[str, Any]
    shape: dict[str, int]            # {rows, columns}
    columns: list[ColumnProfile]
    relationships: dict[str, str]    # FK map (SQL only)
    alias_graph: dict[str, Any]
    profiled_at: str
    sampled: bool
    sample_size: Optional[int]
    knowledge_context: Optional[str]


class LLMSummary(TypedDict):
    text: str                        # the compressed markdown summary
    token_estimate: int


class ProfilerState(TypedDict):
    """
    The full agent state. Every node reads/writes subsets of this.
    List fields use operator.add so multiple nodes can append without clobbering.
    """
    # --- Input ---
    source: DataSource

    # --- Intermediate ---
    raw_df_json: Optional[str]           # serialized DataFrame (JSON orient="split")
    row_count: int
    col_count: int
    sampled: bool
    sample_size: Optional[int]
    schema_only: bool                    # True for very wide SQL schemas

    # --- Profiling results ---
    column_profiles: list[ColumnProfile]
    relationships: dict[str, str]
    alias_graph: dict[str, Any]
    knowledge_context: Optional[str]

    # --- LLM outputs ---
    llm_summary: Optional[LLMSummary]

    # --- Final outputs ---
    rich_profile: Optional[RichProfile]
    output_path_json: Optional[str]
    output_path_summary: Optional[str]
    output_path_alias_graph: Optional[str]

    # --- Agent bookkeeping ---
    messages: Annotated[list[dict], operator.add]   # accumulated log messages
    errors: Annotated[list[str], operator.add]
    current_node: str
    completed: bool
