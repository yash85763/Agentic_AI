"""
nodes/llm_summarise.py — LLM summarisation node.

Takes the rich column profiles + alias graph and asks the LLM to produce
a compressed, accurate, token-efficient summary for injection into
downstream agent system prompts.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from state import ProfilerState, LLMSummary
from llm_config import get_llm

logger = logging.getLogger(__name__)

# Rough token estimate: 1 token ≈ 4 chars
def _estimate_tokens(text: str) -> int:
    return len(text) // 4


SYSTEM_PROMPT = """You are a data documentation specialist. Your job is to produce a
compressed, precise, LLM-readable dataset summary that will be injected into another
AI agent's system prompt. That downstream agent will answer user questions about this
data — potentially with vague, abbreviated, or indirect query language.

Rules:
- Be terse and information-dense. Every word must earn its place.
- Never invent values, types, or relationships not present in the profile.
- Resolve abbreviations in column names where confident.
- Flag columns that are ambiguous or multi-purpose.
- Use Markdown structure for scannability.
- Target: under 2000 tokens for datasets with up to 60 columns.
  For wider datasets, group columns by type and summarise the group.
"""

USER_PROMPT_TEMPLATE = """Produce a compressed LLM-ready dataset summary from the profile below.

## Required Output Sections

### 1. Dataset Overview
One-line description, shape, source type, whether data is sampled.

### 2. Column Inventory
For each column, one concise block:
`column_name` | {type} | {null_pct}% null | {n_unique} unique
  Meaning: {resolved semantic meaning — use likely_concept; if blank, infer from name+type+exemplars}
  Exemplars: {top 5–8 values}
  Stats: {most relevant 2–3 stats for this type}
  Flags: {flags or "—"}

### 3. Semantic Aliases
List columns whose names are abbreviated, ambiguous, or domain-specific.
Format: `col_name` → {what it likely means}

### 4. Key Structural Observations
- ID / primary key columns
- Date / time columns and their granularity
- Categorical dimensions (good for grouping/filtering)
- Numeric measures (good for aggregation)
- Free-text columns (not directly queryable)
- Detected value patterns (email, UUID, currency, etc.)

### 5. Relationships
{FK map or join candidates, or "No relationships detected"}

### 6. Data Quality Notes
- High-null columns (>50%)
- Constant / single-value columns
- Any other anomalies

---
PROFILE JSON:
{profile_json}

{knowledge_section}
"""


def llm_summarise_node(state: ProfilerState) -> dict:
    """LangGraph node: call the LLM to produce a compressed summary."""

    profiles       = state.get("column_profiles", [])
    alias_graph    = state.get("alias_graph", {})
    relationships  = state.get("relationships", {})
    source         = state.get("source", {})
    knowledge      = state.get("knowledge_context")
    row_count      = state.get("row_count", 0)
    col_count      = state.get("col_count", 0)
    sampled        = state.get("sampled", False)
    sample_size    = state.get("sample_size")

    log    = []
    errors = []

    try:
        # ------------------------------------------------------------------ #
        # Build compact profile for the prompt (trim to save tokens)         #
        # ------------------------------------------------------------------ #
        compact_profile = _build_compact_profile(
            profiles, alias_graph, relationships, source,
            row_count, col_count, sampled, sample_size
        )

        knowledge_section = ""
        if knowledge:
            knowledge_section = f"\nDOMAIN KNOWLEDGE PROVIDED BY USER:\n{knowledge[:2000]}\n"

        prompt_text = USER_PROMPT_TEMPLATE.format(
            profile_json=json.dumps(compact_profile, indent=2, default=str),
            knowledge_section=knowledge_section,
        )

        log.append({"msg": "Calling LLM for summary generation", "ts": _ts()})

        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt_text),
        ])

        summary_text = response.content
        token_est    = _estimate_tokens(summary_text)

        log.append({"msg": f"LLM summary generated (~{token_est} tokens)", "ts": _ts()})

        return {
            "llm_summary": LLMSummary(text=summary_text, token_estimate=token_est),
            "current_node": "llm_summarise",
            "messages": log,
            "errors": errors,
        }

    except Exception as e:
        logger.exception("llm_summarise_node failed")
        errors.append(f"[llm_summarise] {type(e).__name__}: {e}")

        # Fallback: generate a simple template-based summary without LLM
        fallback = _fallback_summary(profiles, row_count, col_count, source)
        return {
            "llm_summary": LLMSummary(text=fallback, token_estimate=_estimate_tokens(fallback)),
            "current_node": "llm_summarise",
            "messages": log,
            "errors": errors,
        }


# ─────────────────────────────────────────────────────────────────────────── #
# Helpers                                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

def _build_compact_profile(
    profiles, alias_graph, relationships,
    source, row_count, col_count, sampled, sample_size
) -> dict:
    """Trim the full profile down to what the LLM actually needs."""

    compact_cols = []
    for p in profiles:
        compact_cols.append({
            "name":               p["name"],
            "inferred_type":      p["inferred_type"],
            "null_pct":           p["null_pct"],
            "unique_count":       p["unique_count"],
            "exemplars":          p["exemplars"][:8],
            "exemplar_truncated": p["exemplar_truncated"],
            "stats":              p.get("stats", {}),
            "flags":              p.get("flags", []),
            "pattern_detected":   p.get("pattern_detected"),
            "likely_concept":     p.get("likely_concept", ""),
        })

    return {
        "source_type":   source.get("type"),
        "total_rows":    row_count,
        "columns":       col_count,
        "sampled":       sampled,
        "sample_size":   sample_size,
        "column_profiles": compact_cols,
        "relationships": relationships,
    }


def _fallback_summary(profiles: list, row_count: int, col_count: int, source: dict) -> str:
    """Template-based summary when LLM call fails."""
    lines = [
        f"## Dataset Overview",
        f"- Shape: {row_count} rows × {col_count} columns",
        f"- Source: {source.get('type', 'unknown')}",
        "",
        "## Column Inventory",
    ]
    for p in profiles:
        lines.append(
            f"`{p['name']}` | {p['inferred_type']} | {p['null_pct']}% null | "
            f"{p['unique_count']} unique"
        )
        if p.get("likely_concept"):
            lines.append(f"  Meaning: {p['likely_concept']}")
        if p.get("exemplars"):
            lines.append(f"  Exemplars: {', '.join(p['exemplars'][:5])}")
        lines.append("")
    return "\n".join(lines)


def _ts() -> str:
    return datetime.utcnow().isoformat() + "Z"
