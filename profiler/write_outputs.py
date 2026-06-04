"""
nodes/write_outputs.py — Output writer node.

Saves:
  1. data_profile_rich.json       — full column-level profile
  2. data_profile_llm_summary.md  — compressed LLM-ready summary
  3. data_profile_alias_graph.json — alias/concept mapping

Also assembles the final RichProfile object returned to the calling application.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime

from state import ProfilerState, RichProfile

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "./outputs"


def write_outputs_node(state: ProfilerState) -> dict:
    """LangGraph node: write outputs to disk and assemble final RichProfile."""

    profiles      = state.get("column_profiles", [])
    alias_graph   = state.get("alias_graph", {})
    relationships = state.get("relationships", {})
    llm_summary   = state.get("llm_summary")
    source        = state.get("source", {})
    row_count     = state.get("row_count", 0)
    col_count     = state.get("col_count", 0)
    sampled       = state.get("sampled", False)
    sample_size   = state.get("sample_size")
    knowledge     = state.get("knowledge_context")

    log    = []
    errors = []

    output_dir = _resolve_output_dir(source)
    os.makedirs(output_dir, exist_ok=True)

    try:
        # ------------------------------------------------------------------ #
        # 1. Rich JSON profile                                               #
        # ------------------------------------------------------------------ #
        rich_profile = RichProfile(
            source={
                "type":              source.get("type"),
                "path":              source.get("path"),
                "connection_string": _mask(source.get("connection_string", "")),
                "table_name":        source.get("table_name"),
                "sheet_name":        source.get("sheet_name"),
            },
            shape={"rows": row_count, "columns": col_count},
            columns=profiles,
            relationships=relationships,
            alias_graph=alias_graph,
            profiled_at=datetime.utcnow().isoformat() + "Z",
            sampled=sampled,
            sample_size=sample_size,
            knowledge_context=knowledge,
        )

        json_path = os.path.join(output_dir, "data_profile_rich.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rich_profile, f, indent=2, default=str)
        log.append({"msg": f"Rich profile saved → {json_path}", "ts": _ts()})

        # ------------------------------------------------------------------ #
        # 2. LLM summary markdown                                            #
        # ------------------------------------------------------------------ #
        summary_path = os.path.join(output_dir, "data_profile_llm_summary.md")
        summary_text = llm_summary["text"] if llm_summary else "(no summary generated)"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        log.append({"msg": f"LLM summary saved → {summary_path}", "ts": _ts()})

        # ------------------------------------------------------------------ #
        # 3. Alias graph JSON                                                #
        # ------------------------------------------------------------------ #
        alias_path = os.path.join(output_dir, "data_profile_alias_graph.json")
        with open(alias_path, "w", encoding="utf-8") as f:
            json.dump(alias_graph, f, indent=2, default=str)
        log.append({"msg": f"Alias graph saved → {alias_path}", "ts": _ts()})

        log.append({"msg": "All outputs written successfully.", "ts": _ts()})

        return {
            "rich_profile":           rich_profile,
            "output_path_json":       json_path,
            "output_path_summary":    summary_path,
            "output_path_alias_graph": alias_path,
            "completed":              True,
            "current_node":           "write_outputs",
            "messages":               log,
            "errors":                 errors,
        }

    except Exception as e:
        logger.exception("write_outputs_node failed")
        errors.append(f"[write_outputs] {type(e).__name__}: {e}")
        return {
            "rich_profile":           None,
            "output_path_json":       None,
            "output_path_summary":    None,
            "output_path_alias_graph": None,
            "completed":              False,
            "current_node":           "write_outputs",
            "messages":               log,
            "errors":                 errors,
        }


def _resolve_output_dir(source: dict) -> str:
    """Derive output directory from source path, or use default."""
    path = source.get("path")
    if path:
        base_dir = os.path.dirname(os.path.abspath(path))
        return os.path.join(base_dir, "profiler_output")
    return DEFAULT_OUTPUT_DIR


def _mask(conn_str: str) -> str:
    import re
    return re.sub(r"(:)[^:@]+(@)", r"\1****\2", conn_str) if conn_str else ""


def _ts() -> str:
    return datetime.utcnow().isoformat() + "Z"
