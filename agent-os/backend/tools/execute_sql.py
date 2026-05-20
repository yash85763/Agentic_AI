"""
DuckDB SQL executor for querying Parquet files.

Each entry in parquet_paths is registered as a DuckDB view so that the
caller can reference it by name inside the query, e.g.:

    SELECT * FROM expenses LIMIT 10;

given parquet_paths={"expenses": "/data/expenses.parquet"}.
"""

import time
import logging
from typing import Any

import duckdb

logger = logging.getLogger(__name__)

# Maximum rows we will return to avoid overwhelming the caller
MAX_ROWS = 50_000


def execute_sql(
    query: str,
    parquet_paths: dict[str, str],
) -> dict[str, Any]:
    """
    Execute a DuckDB SQL query over one or more Parquet files.

    Parameters
    ----------
    query:
        SQL query string.  Table names must match the keys of parquet_paths.
    parquet_paths:
        Mapping of {table_name: parquet_file_path}.  Each path is registered
        as a DuckDB view using ``read_parquet()``.

    Returns
    -------
    {
        "columns":    list[str],
        "rows":       list[list[Any]],
        "row_count":  int,
        "duration_ms": int,
        "truncated":  bool,       # True when row_count > MAX_ROWS
        "error":      str | None,
    }
    """
    start_ts = time.monotonic()

    # Use an in-memory database for full isolation between calls
    conn = duckdb.connect(database=":memory:")

    try:
        # Register each parquet file as a named view
        for table_name, path in parquet_paths.items():
            _validate_identifier(table_name)
            # read_parquet handles both single files and glob patterns
            conn.execute(
                f"CREATE OR REPLACE VIEW {table_name} AS "
                f"SELECT * FROM read_parquet('{path}')"
            )
            logger.debug("Registered view %r -> %r", table_name, path)

        # Execute the user query
        relation = conn.execute(query)

        # Fetch column names
        columns: list[str] = [desc[0] for desc in relation.description]

        # Fetch rows, but cap at MAX_ROWS
        rows_raw = relation.fetchmany(MAX_ROWS + 1)
        truncated = len(rows_raw) > MAX_ROWS
        if truncated:
            rows_raw = rows_raw[:MAX_ROWS]

        # Convert to plain Python lists (DuckDB may return tuples)
        rows: list[list[Any]] = [list(row) for row in rows_raw]

        duration_ms = int((time.monotonic() - start_ts) * 1000)

        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "duration_ms": duration_ms,
            "truncated": truncated,
            "error": None,
        }

    except duckdb.Error as exc:
        duration_ms = int((time.monotonic() - start_ts) * 1000)
        logger.warning("DuckDB error: %s", exc)
        return {
            "columns": [],
            "rows": [],
            "row_count": 0,
            "duration_ms": duration_ms,
            "truncated": False,
            "error": str(exc),
        }
    except Exception as exc:
        duration_ms = int((time.monotonic() - start_ts) * 1000)
        logger.exception("Unexpected error in execute_sql")
        return {
            "columns": [],
            "rows": [],
            "row_count": 0,
            "duration_ms": duration_ms,
            "truncated": False,
            "error": f"Internal error: {exc}",
        }
    finally:
        conn.close()


def execute_sql_to_dataframe(
    query: str,
    parquet_paths: dict[str, str],
):
    """
    Convenience wrapper that returns a pandas DataFrame instead of a dict.

    Raises on SQL errors (no silent swallowing).
    """
    import pandas as pd

    conn = duckdb.connect(database=":memory:")
    try:
        for table_name, path in parquet_paths.items():
            _validate_identifier(table_name)
            conn.execute(
                f"CREATE OR REPLACE VIEW {table_name} AS "
                f"SELECT * FROM read_parquet('{path}')"
            )
        return conn.execute(query).df()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAFE_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
)


def _validate_identifier(name: str) -> None:
    """Reject table names that could enable SQL injection via the CREATE VIEW statement."""
    if not name:
        raise ValueError("Table name must not be empty.")
    if not all(c in _SAFE_CHARS for c in name):
        raise ValueError(
            f"Table name {name!r} contains characters outside [A-Za-z0-9_]. "
            "Please use a safe identifier."
        )
    if name[0].isdigit():
        raise ValueError(f"Table name {name!r} must not start with a digit.")
