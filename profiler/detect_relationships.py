"""
nodes/detect_relationships.py — Relationship and structure detection node.

For SQL sources: reads FK constraints from the DB schema.
For all sources: heuristically identifies potential join keys
                 (ID columns that appear in multiple tables or
                 share naming patterns across columns).
Also builds the semantic alias graph used for query routing.
"""

from __future__ import annotations

import io
import json
import logging
import re
from datetime import datetime

import pandas as pd

from state import ProfilerState

logger = logging.getLogger(__name__)


def detect_relationships_node(state: ProfilerState) -> dict:
    """LangGraph node: build relationship map and alias graph."""

    source       = state.get("source", {})
    profiles     = state.get("column_profiles", [])
    raw_df_json  = state.get("raw_df_json")

    log    = []
    errors = []

    relationships: dict[str, str] = {}
    alias_graph: dict = {}

    try:
        # ------------------------------------------------------------------ #
        # 1. SQL: extract FK constraints from DB schema                      #
        # ------------------------------------------------------------------ #
        if source.get("type") == "sql" and source.get("connection_string"):
            relationships = _extract_sql_relationships(source, log, errors)

        # ------------------------------------------------------------------ #
        # 2. Heuristic join-key candidates (file or SQL)                     #
        # ------------------------------------------------------------------ #
        if raw_df_json:
            df = pd.read_json(io.StringIO(raw_df_json), orient="split")
            candidates = _find_join_candidates(df, profiles, log)
            for k, v in candidates.items():
                if k not in relationships:
                    relationships[k] = v

        # ------------------------------------------------------------------ #
        # 3. Build alias graph                                               #
        # ------------------------------------------------------------------ #
        alias_graph = _build_alias_graph(profiles)
        log.append({"msg": f"Alias graph built for {len(alias_graph)} columns", "ts": _ts()})

        return {
            "relationships": relationships,
            "alias_graph": alias_graph,
            "current_node": "detect_relationships",
            "messages": log,
            "errors": errors,
        }

    except Exception as e:
        logger.exception("detect_relationships_node failed")
        errors.append(f"[detect_relationships] {type(e).__name__}: {e}")
        return {
            "relationships": {},
            "alias_graph": {},
            "current_node": "detect_relationships",
            "messages": log,
            "errors": errors,
        }


# ─────────────────────────────────────────────────────────────────────────── #
# SQL FK extraction                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

def _extract_sql_relationships(source: dict, log: list, errors: list) -> dict[str, str]:
    try:
        import sqlalchemy as sa
        engine = sa.create_engine(source["connection_string"])
        inspector = sa.inspect(engine)

        schema = source.get("schema", "public")
        tables = inspector.get_table_names(schema=schema)
        log.append({"msg": f"SQL schema has {len(tables)} tables", "ts": _ts()})

        relationships = {}
        for table in tables:
            try:
                fks = inspector.get_foreign_keys(table, schema=schema)
                for fk in fks:
                    from_cols = fk.get("constrained_columns", [])
                    to_table  = fk.get("referred_table", "?")
                    to_cols   = fk.get("referred_columns", [])
                    for fc, tc in zip(from_cols, to_cols):
                        key = f"{table}.{fc}"
                        val = f"{to_table}.{tc}"
                        relationships[key] = val
            except Exception as e:
                errors.append(f"FK extraction failed for table '{table}': {e}")

        log.append({"msg": f"Found {len(relationships)} FK relationships", "ts": _ts()})
        return relationships

    except Exception as e:
        errors.append(f"SQL relationship extraction failed: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────── #
# Heuristic join key detection                                               #
# ─────────────────────────────────────────────────────────────────────────── #

def _find_join_candidates(
    df: pd.DataFrame,
    profiles: list,
    log: list,
) -> dict[str, str]:
    """
    Heuristic: columns named *_id, *_key, *_code that are high-cardinality
    strings or integers are likely foreign keys or join candidates.
    """
    candidates = {}
    id_patterns = re.compile(r"(^id$|_id$|_key$|_code$|^fk_|_fk$)", re.IGNORECASE)

    id_cols = []
    for p in profiles:
        if id_patterns.search(p["name"]) or "LIKELY_ID" in " ".join(p.get("flags", [])):
            id_cols.append(p["name"])

    # If two columns share the same name pattern (e.g. orders.customer_id ~ customers.id)
    # note them as candidate joins
    for col in id_cols:
        # Guess the referenced table from the column name
        base = re.sub(r"[_]?id$|[_]?key$|[_]?code$", "", col, flags=re.IGNORECASE).strip("_")
        if base and base.lower() not in ("", col.lower()):
            candidates[col] = f"{base}(inferred)"

    if candidates:
        log.append({"msg": f"Heuristic join candidates: {list(candidates.keys())}", "ts": _ts()})

    return candidates


# ─────────────────────────────────────────────────────────────────────────── #
# Alias graph                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

def _build_alias_graph(profiles: list) -> dict:
    """
    Alias graph: column_name → {type, exemplars, likely_concept, flags, pattern}
    Used by the main agent to resolve indirect / abbreviated user queries.
    """
    graph = {}
    for p in profiles:
        graph[p["name"]] = {
            "inferred_type":    p["inferred_type"],
            "likely_concept":   p.get("likely_concept", ""),
            "pattern_detected": p.get("pattern_detected"),
            "exemplars":        p["exemplars"][:8],  # compact view for the graph
            "exemplar_truncated": p["exemplar_truncated"],
            "null_pct":         p["null_pct"],
            "unique_count":     p["unique_count"],
            "flags":            p.get("flags", []),
        }
    return graph


def _ts() -> str:
    return datetime.utcnow().isoformat() + "Z"
