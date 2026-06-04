"""
nodes/profile_columns.py — Statistical column profiler node.

For every column in the DataFrame, computes:
  - Inferred semantic type
  - Null stats
  - Unique value count + exemplars
  - Type-specific statistics (numeric, datetime, categorical, free-text)
  - Data quality flags
  - Value pattern detection (email, UUID, etc.)
  - Semantic alias / likely concept
"""

from __future__ import annotations

import io
import json
import logging
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import numpy as np

from state import ProfilerState, ColumnProfile
from utils.type_inference import (
    infer_semantic_type,
    detect_value_pattern,
    resolve_column_alias,
    detect_temporal_granularity,
)
from utils.exemplars import sample_exemplars

logger = logging.getLogger(__name__)


def profile_columns_node(state: ProfilerState) -> dict:
    """LangGraph node: build a ColumnProfile for every column."""

    raw_df_json = state.get("raw_df_json")
    if not raw_df_json:
        return {
            "column_profiles": [],
            "current_node": "profile_columns",
            "errors": ["[profile_columns] No DataFrame in state — ingest may have failed."],
            "messages": [],
        }

    log = []
    errors = []

    try:
        df = pd.read_json(io.StringIO(raw_df_json), orient="split")
        log.append({"msg": f"Profiling {len(df.columns)} columns × {len(df)} rows", "ts": _ts()})

        profiles: list[ColumnProfile] = []
        for col in df.columns:
            try:
                cp = _profile_single_column(df[col], col)
                profiles.append(cp)
            except Exception as e:
                errors.append(f"[profile_columns] column '{col}': {e}")
                logger.warning(f"Failed to profile column '{col}': {e}")

        log.append({"msg": f"Profiled {len(profiles)} columns successfully", "ts": _ts()})

        return {
            "column_profiles": profiles,
            "current_node": "profile_columns",
            "messages": log,
            "errors": errors,
        }

    except Exception as e:
        logger.exception("profile_columns_node failed")
        errors.append(f"[profile_columns] {type(e).__name__}: {e}")
        return {
            "column_profiles": [],
            "current_node": "profile_columns",
            "messages": log,
            "errors": errors,
        }


# ─────────────────────────────────────────────────────────────────────────── #
# Core column profiler                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def _profile_single_column(series: pd.Series, col_name: str) -> ColumnProfile:
    non_null = series.dropna()
    n_total  = len(series)
    n_null   = int(series.isna().sum())
    n_unique = int(series.nunique(dropna=True))

    inferred_type = infer_semantic_type(series, non_null)

    # Exemplars
    unique_vals    = non_null.unique().tolist()
    exemplars, truncated = sample_exemplars(unique_vals)

    # Type-specific stats
    stats = _compute_stats(series, non_null, inferred_type)

    # Flags
    flags = _compute_flags(series, non_null, n_null, n_total, n_unique, inferred_type)

    # Pattern detection (only for string-like columns)
    pattern = None
    if inferred_type in ("string", "categorical", "unknown"):
        pattern = detect_value_pattern(series)
        if pattern:
            flags.append(f"PATTERN:{pattern.upper()}")

    # Semantic alias
    likely_concept = resolve_column_alias(col_name)

    return ColumnProfile(
        name=col_name,
        raw_dtype=str(series.dtype),
        inferred_type=inferred_type,
        total_count=n_total,
        null_count=n_null,
        null_pct=round(n_null / max(n_total, 1) * 100, 2),
        unique_count=n_unique,
        exemplars=exemplars,
        exemplar_truncated=truncated,
        stats=stats,
        flags=flags,
        pattern_detected=pattern,
        likely_concept=likely_concept,
    )


def _compute_stats(series: pd.Series, non_null: pd.Series, itype: str) -> dict[str, Any]:
    if len(non_null) == 0:
        return {}

    if itype in ("integer", "float"):
        try:
            numeric = pd.to_numeric(non_null, errors="coerce").dropna()
            return {
                "min":          _safe_float(numeric.min()),
                "max":          _safe_float(numeric.max()),
                "mean":         _safe_float(numeric.mean()),
                "median":       _safe_float(numeric.median()),
                "std":          _safe_float(numeric.std()),
                "p25":          _safe_float(numeric.quantile(0.25)),
                "p75":          _safe_float(numeric.quantile(0.75)),
                "zeros_pct":    round((numeric == 0).mean() * 100, 2),
                "negative_pct": round((numeric < 0).mean() * 100, 2),
            }
        except Exception:
            return {}

    elif itype == "datetime":
        try:
            dt = pd.to_datetime(non_null, errors="coerce").dropna()
            granularity = detect_temporal_granularity(dt)
            return {
                "min":            str(dt.min()),
                "max":            str(dt.max()),
                "range_days":     int((dt.max() - dt.min()).days),
                "granularity":    granularity,
                "parse_fail_pct": round(pd.to_datetime(non_null, errors="coerce").isna().mean() * 100, 2),
            }
        except Exception:
            return {}

    elif itype == "categorical":
        try:
            vc = non_null.value_counts()
            top10 = vc.head(10)
            return {
                "top_values":            {str(k): int(v) for k, v in top10.items()},
                "top10_coverage_pct":    round(top10.sum() / len(non_null) * 100, 2),
            }
        except Exception:
            return {}

    elif itype == "boolean":
        try:
            vc = non_null.astype(str).str.strip().str.lower().value_counts(normalize=True)
            return {str(k): round(float(v) * 100, 2) for k, v in vc.items()}
        except Exception:
            return {}

    elif itype == "free_text":
        try:
            lengths = non_null.astype(str).str.len()
            return {
                "avg_length": round(float(lengths.mean()), 1),
                "max_length": int(lengths.max()),
                "min_length": int(lengths.min()),
            }
        except Exception:
            return {}

    elif itype == "string":
        try:
            lengths = non_null.astype(str).str.len()
            return {
                "avg_length": round(float(lengths.mean()), 1),
                "max_length": int(lengths.max()),
            }
        except Exception:
            return {}

    return {}


def _compute_flags(
    series: pd.Series,
    non_null: pd.Series,
    n_null: int,
    n_total: int,
    n_unique: int,
    itype: str,
) -> list[str]:
    flags = []

    null_pct = n_null / max(n_total, 1) * 100
    if null_pct > 75:
        flags.append("HIGH_NULLS:>75%")
    elif null_pct > 50:
        flags.append("HIGH_NULLS:>50%")
    elif null_pct > 20:
        flags.append("MODERATE_NULLS:>20%")

    if n_unique == n_total and n_total > 1:
        flags.append("LIKELY_ID:all_unique")

    if n_unique == 1:
        flags.append("CONSTANT:single_value")

    if n_unique == 2 and itype not in ("boolean",):
        flags.append("BINARY_COLUMN")

    if itype == "float":
        try:
            numeric = pd.to_numeric(non_null, errors="coerce")
            if (numeric == 0).mean() > 0.90:
                flags.append("MOSTLY_ZERO")
        except Exception:
            pass

    return flags


def _safe_float(val) -> Optional[float]:
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return round(float(val), 6)
    except Exception:
        return None


def _ts() -> str:
    return datetime.utcnow().isoformat() + "Z"
