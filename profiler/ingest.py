"""
nodes/ingest.py — Data ingestion node.

Responsibilities:
  - Detect source type (CSV / Excel / SQL)
  - Apply size-safe loading with chunked sampling for large files
  - Serialise the DataFrame into state as JSON (orient="split")
  - Load optional knowledge.md file
"""

from __future__ import annotations

import os
import json
import logging
from datetime import datetime

import pandas as pd
import numpy as np

from state import ProfilerState, DataSource

logger = logging.getLogger(__name__)

# Thresholds
FULL_LOAD_BYTES     = 50  * 1024 * 1024   # 50 MB  → load in full
CHUNKED_LOAD_BYTES  = 500 * 1024 * 1024   # 500 MB → chunked sample only
TARGET_SAMPLE_ROWS  = 50_000


def ingest_node(state: ProfilerState) -> dict:
    """LangGraph node: load data from CSV / Excel / SQL into a sampled DataFrame."""

    source: DataSource = state["source"]
    src_type = source.get("type", "").lower()

    log = []
    errors = []
    df = None
    sampled = False
    sample_size = None
    total_rows = None
    knowledge_context = None

    try:
        # ------------------------------------------------------------------ #
        # 1. Load knowledge file (optional)                                  #
        # ------------------------------------------------------------------ #
        kf = source.get("knowledge_file")
        if kf and os.path.exists(kf):
            with open(kf, "r", encoding="utf-8") as f:
                knowledge_context = f.read()
            log.append({"msg": f"Loaded knowledge file: {kf}", "ts": _ts()})

        # ------------------------------------------------------------------ #
        # 2. Load the data                                                   #
        # ------------------------------------------------------------------ #
        if src_type in ("csv", "tsv"):
            df, sampled, sample_size, total_rows = _load_csv(source, log)

        elif src_type in ("excel", "xlsx", "xls", "xlsm", "ods"):
            df, sampled, sample_size, total_rows = _load_excel(source, log)

        elif src_type == "sql":
            df, sampled, sample_size, total_rows = _load_sql(source, log)

        else:
            raise ValueError(
                f"Unsupported source type '{src_type}'. "
                "Expected: csv | tsv | excel | xlsx | xls | xlsm | ods | sql"
            )

        if df is None or df.empty:
            raise ValueError("Loaded DataFrame is empty — check the data source.")

        # ------------------------------------------------------------------ #
        # 3. Normalise column names                                          #
        # ------------------------------------------------------------------ #
        df.columns = [str(c).strip() for c in df.columns]

        # ------------------------------------------------------------------ #
        # 4. Serialise to JSON for state transport                           #
        # ------------------------------------------------------------------ #
        # NaN/NaT → None for JSON compatibility
        df_clean = df.where(pd.notnull(df), other=None)
        raw_json = df_clean.to_json(orient="split", date_format="iso", default_handler=str)

        log.append({"msg": f"Ingested {len(df)} rows × {len(df.columns)} columns", "ts": _ts()})

        return {
            "raw_df_json": raw_json,
            "row_count": total_rows or len(df),
            "col_count": len(df.columns),
            "sampled": sampled,
            "sample_size": sample_size if sampled else len(df),
            "knowledge_context": knowledge_context,
            "current_node": "ingest",
            "messages": log,
            "errors": errors,
        }

    except Exception as e:
        logger.exception("Ingest node failed")
        errors.append(f"[ingest] {type(e).__name__}: {e}")
        return {
            "raw_df_json": None,
            "row_count": 0,
            "col_count": 0,
            "sampled": False,
            "sample_size": None,
            "knowledge_context": knowledge_context,
            "current_node": "ingest",
            "messages": log,
            "errors": errors,
        }


# ─────────────────────────────────────────────────────────────────────────── #
# Helpers                                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

def _load_csv(source: DataSource, log: list) -> tuple:
    path = source.get("path")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")

    sep = "\t" if str(path).endswith(".tsv") else ","
    size = os.path.getsize(path)
    log.append({"msg": f"CSV size: {size / 1e6:.1f} MB", "ts": _ts()})

    # Count rows without loading
    try:
        total_rows = sum(1 for _ in open(path, encoding="utf-8", errors="replace")) - 1
    except Exception:
        total_rows = None

    if size <= FULL_LOAD_BYTES:
        df = _read_csv_safe(path, sep=sep)
        return df, False, None, total_rows

    elif size <= CHUNKED_LOAD_BYTES:
        # Systematic sample
        step = max(1, (total_rows or TARGET_SAMPLE_ROWS * 2) // TARGET_SAMPLE_ROWS)
        df = _read_csv_safe(
            path, sep=sep,
            skiprows=lambda i: i != 0 and i % step != 0
        )
        log.append({"msg": f"Sampled 1-in-{step} rows → {len(df)} rows", "ts": _ts()})
        return df, True, len(df), total_rows

    else:
        # Very large: read only TARGET_SAMPLE_ROWS rows in chunks, reservoir-style
        chunks = []
        for chunk in pd.read_csv(path, sep=sep, chunksize=10_000, on_bad_lines="skip"):
            chunks.append(chunk)
            if sum(len(c) for c in chunks) >= TARGET_SAMPLE_ROWS:
                break
        df = pd.concat(chunks, ignore_index=True).head(TARGET_SAMPLE_ROWS)
        log.append({"msg": f"Large file: head-sampled {len(df)} rows", "ts": _ts()})
        return df, True, len(df), total_rows


def _read_csv_safe(path: str, sep: str = ",", **kwargs) -> pd.DataFrame:
    """Try UTF-8 first, fall back to latin-1."""
    try:
        return pd.read_csv(path, sep=sep, on_bad_lines="skip", low_memory=False, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=sep, encoding="latin-1", on_bad_lines="skip",
                           low_memory=False, **kwargs)


def _load_excel(source: DataSource, log: list) -> tuple:
    path = source.get("path")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Excel file not found: {path}")

    sheet = source.get("sheet_name", 0)
    ext = os.path.splitext(path)[1].lower()

    engine_map = {".xlsx": "openpyxl", ".xlsm": "openpyxl",
                  ".xls": "xlrd",      ".ods": "odf"}
    engine = engine_map.get(ext, "openpyxl")

    size = os.path.getsize(path)
    log.append({"msg": f"Excel size: {size / 1e6:.1f} MB, engine={engine}", "ts": _ts()})

    df = pd.read_excel(path, sheet_name=sheet, engine=engine)

    sampled = False
    sample_size = None
    total_rows = len(df)

    if len(df) > TARGET_SAMPLE_ROWS:
        step = len(df) // TARGET_SAMPLE_ROWS
        df = df.iloc[::step].reset_index(drop=True).head(TARGET_SAMPLE_ROWS)
        sampled = True
        sample_size = len(df)
        log.append({"msg": f"Excel: sampled 1-in-{step} → {len(df)} rows", "ts": _ts()})

    return df, sampled, sample_size, total_rows


def _load_sql(source: DataSource, log: list) -> tuple:
    try:
        import sqlalchemy as sa
    except ImportError:
        raise ImportError("sqlalchemy is required for SQL ingestion. pip install sqlalchemy")

    conn_str = source.get("connection_string")
    table = source.get("table_name")
    if not conn_str:
        raise ValueError("connection_string is required for SQL source.")
    if not table:
        raise ValueError("table_name is required for SQL source.")

    engine = sa.create_engine(conn_str)
    log.append({"msg": f"Connected to SQL: {_mask_conn(conn_str)}", "ts": _ts()})

    # Row count
    with engine.connect() as conn:
        total_rows = conn.execute(sa.text(f"SELECT COUNT(*) FROM {table}")).scalar()
    log.append({"msg": f"SQL table '{table}': {total_rows} rows", "ts": _ts()})

    sampled = total_rows > TARGET_SAMPLE_ROWS
    if sampled:
        # TABLESAMPLE varies by DB dialect — fall back to LIMIT
        try:
            df = pd.read_sql(
                f"SELECT * FROM {table} TABLESAMPLE SYSTEM(10) LIMIT {TARGET_SAMPLE_ROWS}",
                engine
            )
        except Exception:
            df = pd.read_sql(f"SELECT * FROM {table} LIMIT {TARGET_SAMPLE_ROWS}", engine)
        log.append({"msg": f"SQL: sampled {len(df)} of {total_rows} rows", "ts": _ts()})
    else:
        df = pd.read_sql(f"SELECT * FROM {table}", engine)

    return df, sampled, len(df) if sampled else None, total_rows


def _mask_conn(conn_str: str) -> str:
    """Mask password in connection string for logging."""
    import re
    return re.sub(r"(:)[^:@]+(@)", r"\1****\2", conn_str)


def _ts() -> str:
    return datetime.utcnow().isoformat() + "Z"
