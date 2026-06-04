"""
utils/type_inference.py — Semantic column type inference and pattern detection.
"""

from __future__ import annotations

import re
import random
import pandas as pd
import numpy as np
from typing import Optional

# Strings longer than this are "free text", not categorical
LONG_STRING_THRESHOLD = 80

# Regex patterns for structured string detection
VALUE_PATTERNS: dict[str, str] = {
    "email":         r"^[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}$",
    "uuid":          r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    "phone_us":      r"^\+?1?\s?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}$",
    "zip_us":        r"^\d{5}(-\d{4})?$",
    "iso_country":   r"^[A-Z]{2,3}$",
    "currency_code": r"^(USD|EUR|GBP|JPY|CAD|AUD|CNY|INR|CHF|MXN|BRL)$",
    "ip_address":    r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$",
    "url":           r"^https?://[\w\-]+(\.[\w\-]+)+",
    "hex_color":     r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$",
}

# Domain abbreviation dictionary for alias resolution
DOMAIN_ABBREVIATIONS: dict[str, str] = {
    "id":     "identifier / primary key",
    "dt":     "date",
    "ts":     "timestamp",
    "amt":    "amount / monetary value",
    "qty":    "quantity / count",
    "pct":    "percentage",
    "avg":    "average",
    "cnt":    "count",
    "num":    "number",
    "rev":    "revenue",
    "txn":    "transaction",
    "cust":   "customer",
    "prod":   "product",
    "cat":    "category",
    "desc":   "description",
    "src":    "source",
    "dst":    "destination",
    "loc":    "location",
    "lat":    "latitude",
    "lon":    "longitude",
    "lng":    "longitude",
    "yr":     "year",
    "mo":     "month",
    "wk":     "week",
    "sku":    "stock keeping unit / product code",
    "gmv":    "gross merchandise value",
    "ltv":    "lifetime value",
    "arr":    "annual recurring revenue",
    "mrr":    "monthly recurring revenue",
    "nps":    "net promoter score",
    "fico":   "credit score",
    "acct":   "account",
    "addr":   "address",
    "usr":    "user",
    "org":    "organization",
    "dept":   "department",
    "mgr":    "manager",
    "emp":    "employee",
    "evt":    "event",
    "msg":    "message",
    "req":    "request",
    "res":    "response",
    "err":    "error",
    "cfg":    "configuration",
    "env":    "environment",
    "ver":    "version",
    "seq":    "sequence",
    "pos":    "position",
    "neg":    "negative",
    "pos":    "positive",
    "flag":   "boolean flag",
    "ind":    "indicator",
    "grp":    "group",
    "seg":    "segment",
    "cls":    "class / classification",
    "lbl":    "label",
    "val":    "value",
    "ref":    "reference",
    "ext":    "external",
    "int":    "internal",
    "max":    "maximum",
    "min":    "minimum",
}


def infer_semantic_type(series: pd.Series, non_null: pd.Series) -> str:
    """
    Infer semantic type beyond pandas dtype.
    Returns one of: integer | float | datetime | categorical | string |
                    free_text | boolean | empty | unknown
    """
    dtype = str(series.dtype)

    if len(non_null) == 0:
        return "empty"

    if dtype == "bool":
        return "boolean"

    if dtype in ("int64", "int32", "int16", "int8", "uint64", "uint32", "Int64"):
        return "integer"

    if dtype in ("float64", "float32", "Float64"):
        # Integers stored as float (e.g. 1.0, 2.0, NaN-padded integer columns)
        try:
            if non_null.apply(lambda x: float(x) == int(x)).all():
                return "integer"
        except (ValueError, OverflowError):
            pass
        return "float"

    if "datetime" in dtype or "Datetime" in dtype:
        return "datetime"

    if dtype in ("object", "string", "str"):
        # Try datetime parse on a sample
        sample = non_null.sample(min(50, len(non_null)), random_state=42)
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() > 0.85:
                return "datetime"
        except Exception:
            pass

        # Boolean-like strings
        bool_vals = {"true", "false", "yes", "no", "1", "0", "y", "n", "t", "f"}
        try:
            unique_lower = set(non_null.astype(str).str.strip().str.lower().unique())
            if unique_lower.issubset(bool_vals) and len(unique_lower) <= 4:
                return "boolean"
        except Exception:
            pass

        # Free text: long average string length
        try:
            avg_len = non_null.astype(str).str.len().mean()
            if avg_len > LONG_STRING_THRESHOLD:
                return "free_text"
        except Exception:
            pass

        # Categorical: low unique ratio OR few absolute unique values
        unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
        if unique_ratio < 0.10 or series.nunique(dropna=True) <= 50:
            return "categorical"

        # High-cardinality short strings: names, IDs, codes
        return "string"

    return "unknown"


def detect_value_pattern(series: pd.Series, sample_size: int = 100) -> Optional[str]:
    """
    Check if a string/categorical column's values match a known structural pattern.
    Returns the pattern name or None.
    """
    non_null = series.dropna().astype(str).str.strip()
    if len(non_null) == 0:
        return None

    sample = non_null.sample(min(sample_size, len(non_null)), random_state=0)

    for pattern_name, pattern in VALUE_PATTERNS.items():
        try:
            match_rate = sample.str.match(pattern, case=False, na=False).mean()
            if match_rate >= 0.80:
                return pattern_name
        except Exception:
            continue

    return None


def resolve_column_alias(col_name: str) -> str:
    """
    Attempt to resolve a column name's likely semantic meaning using
    the abbreviation dictionary and common naming patterns.
    """
    name = col_name.lower().strip()

    # Direct full match
    if name in DOMAIN_ABBREVIATIONS:
        return DOMAIN_ABBREVIATIONS[name]

    # Split on common separators and check each token
    tokens = re.split(r"[_\-\s\.]+", name)
    matched = []
    for token in tokens:
        if token in DOMAIN_ABBREVIATIONS:
            matched.append(DOMAIN_ABBREVIATIONS[token])

    if matched:
        return " / ".join(matched)

    # Prefix check (e.g. "cust123" → customer)
    for abbr, meaning in DOMAIN_ABBREVIATIONS.items():
        if name.startswith(abbr) or name.endswith(abbr):
            return meaning

    return ""


def detect_temporal_granularity(dt_series: pd.Series) -> str:
    """Given a datetime Series, estimate the time granularity of observations."""
    try:
        sorted_series = dt_series.dropna().sort_values()
        diffs = sorted_series.diff().dropna()
        if len(diffs) == 0:
            return "unknown"
        median_diff = diffs.median()
        days = median_diff.total_seconds() / 86400

        if days < 0.04:     return "sub-hourly (minutes/seconds)"
        elif days < 1:      return "sub-daily (hours)"
        elif days <= 1.5:   return "daily"
        elif days <= 8:     return "weekly"
        elif days <= 32:    return "monthly"
        elif days <= 95:    return "quarterly"
        elif days <= 370:   return "yearly"
        else:               return f"irregular (~{int(days)}d median gap)"
    except Exception:
        return "unknown"
