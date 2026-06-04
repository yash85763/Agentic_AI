"""
agent.py — Data Profiler Agent (LangGraph)

Graph topology:

  [START]
     │
     ▼
  ingest ──(fail)──► END
     │
     ▼
  profile_columns
     │
     ▼
  detect_relationships
     │
     ▼
  llm_summarise
     │
     ▼
  write_outputs
     │
     ▼
  [END]

Each node appends to state.messages and state.errors.
A failed ingest short-circuits the graph via conditional edge.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import StateGraph, START, END

from state import ProfilerState
from nodes.ingest            import ingest_node
from nodes.profile_columns   import profile_columns_node
from nodes.detect_relationships import detect_relationships_node
from nodes.llm_summarise     import llm_summarise_node
from nodes.write_outputs     import write_outputs_node

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────── #
# Conditional routing                                                         #
# ─────────────────────────────────────────────────────────────────────────── #

def route_after_ingest(state: ProfilerState) -> str:
    """If ingest failed (no DataFrame), skip to END."""
    if not state.get("raw_df_json"):
        logger.warning("Ingest failed — aborting profiling pipeline.")
        return "abort"
    return "continue"


# ─────────────────────────────────────────────────────────────────────────── #
# Graph construction                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

def build_profiler_graph() -> StateGraph:
    graph = StateGraph(ProfilerState)

    # Register nodes
    graph.add_node("ingest",                ingest_node)
    graph.add_node("profile_columns",       profile_columns_node)
    graph.add_node("detect_relationships",  detect_relationships_node)
    graph.add_node("llm_summarise",         llm_summarise_node)
    graph.add_node("write_outputs",         write_outputs_node)

    # Entry edge
    graph.add_edge(START, "ingest")

    # Conditional edge after ingest
    graph.add_conditional_edges(
        "ingest",
        route_after_ingest,
        {
            "continue": "profile_columns",
            "abort":    END,
        }
    )

    # Linear pipeline
    graph.add_edge("profile_columns",      "detect_relationships")
    graph.add_edge("detect_relationships", "llm_summarise")
    graph.add_edge("llm_summarise",        "write_outputs")
    graph.add_edge("write_outputs",        END)

    return graph.compile()


# Singleton compiled graph
profiler_graph = build_profiler_graph()


# ─────────────────────────────────────────────────────────────────────────── #
# Public API                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

def run_profiler(
    source_type: str,
    path: str | None = None,
    connection_string: str | None = None,
    table_name: str | None = None,
    sheet_name: str | None = None,
    knowledge_file: str | None = None,
) -> dict[str, Any]:
    """
    Run the Data Profiler Agent and return a results dict.

    Parameters
    ----------
    source_type : str
        One of: "csv" | "tsv" | "excel" | "xlsx" | "xls" | "sql"
    path : str, optional
        File path for CSV/Excel sources.
    connection_string : str, optional
        SQLAlchemy connection string for SQL sources.
    table_name : str, optional
        Table name for SQL sources.
    sheet_name : str or int, optional
        Sheet name or index for Excel sources (default: 0).
    knowledge_file : str, optional
        Path to a knowledge.md / skill.md file with domain context.

    Returns
    -------
    dict with keys:
        rich_profile         : RichProfile TypedDict (full data, in-memory)
        llm_summary          : str  — compressed markdown summary
        alias_graph          : dict — column → concept mapping
        output_path_json     : str  — path to saved rich JSON
        output_path_summary  : str  — path to saved LLM summary markdown
        output_path_alias_graph : str
        messages             : list[dict] — execution log
        errors               : list[str]  — any errors encountered
        completed            : bool
    """
    initial_state: ProfilerState = {
        "source": {
            "type":              source_type,
            "path":              path,
            "connection_string": connection_string,
            "table_name":        table_name,
            "sheet_name":        sheet_name,
            "knowledge_file":    knowledge_file,
        },
        "raw_df_json":       None,
        "row_count":         0,
        "col_count":         0,
        "sampled":           False,
        "sample_size":       None,
        "schema_only":       False,
        "column_profiles":   [],
        "relationships":     {},
        "alias_graph":       {},
        "knowledge_context": None,
        "llm_summary":       None,
        "rich_profile":      None,
        "output_path_json":  None,
        "output_path_summary": None,
        "output_path_alias_graph": None,
        "messages":          [],
        "errors":            [],
        "current_node":      "init",
        "completed":         False,
    }

    logger.info(f"Starting profiler: source_type={source_type}, path={path}, table={table_name}")

    final_state = profiler_graph.invoke(initial_state)

    return {
        "rich_profile":              final_state.get("rich_profile"),
        "llm_summary":               final_state.get("llm_summary", {}).get("text", ""),
        "alias_graph":               final_state.get("alias_graph", {}),
        "output_path_json":          final_state.get("output_path_json"),
        "output_path_summary":       final_state.get("output_path_summary"),
        "output_path_alias_graph":   final_state.get("output_path_alias_graph"),
        "messages":                  final_state.get("messages", []),
        "errors":                    final_state.get("errors", []),
        "completed":                 final_state.get("completed", False),
    }
