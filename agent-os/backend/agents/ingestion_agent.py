"""
IngestionAgent - Reads uploaded Excel/CSV files from MinIO, detects sheets,
extracts schema information, and classifies sheets as data vs. metadata.

Output: a list of FileManifest dicts attached to the pipeline state.
"""

from __future__ import annotations

import io
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from langfuse.decorators import observe

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

WORKER_MODEL = os.getenv("WORKER_MODEL", "ollama/qwen2.5-coder:32b")
WORKER_FALLBACK_MODEL = os.getenv("WORKER_FALLBACK_MODEL", "openai/gpt-4o-mini")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ColumnInfo:
    name: str
    dtype: str
    non_null_count: int
    null_count: int
    unique_count: int
    sample_values: List[Any] = field(default_factory=list)
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None


@dataclass
class SheetClassification:
    sheet_name: str
    classification: str  # "data" | "metadata" | "lookup" | "summary" | "unknown"
    confidence: float    # 0.0 – 1.0
    reasoning: str = ""


@dataclass
class SheetManifest:
    sheet_name: str
    classification: SheetClassification
    row_count: int
    column_count: int
    columns: List[ColumnInfo] = field(default_factory=list)
    sample_rows: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FileManifest:
    file_name: str
    file_path: str
    file_type: str  # "excel" | "csv"
    total_rows: int
    total_columns: int
    sheets: List[SheetManifest] = field(default_factory=list)
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        def _col(c: ColumnInfo) -> Dict[str, Any]:
            return {
                "name": c.name,
                "dtype": c.dtype,
                "non_null_count": c.non_null_count,
                "null_count": c.null_count,
                "unique_count": c.unique_count,
                "sample_values": [str(v) for v in c.sample_values],
                "min_value": str(c.min_value) if c.min_value is not None else None,
                "max_value": str(c.max_value) if c.max_value is not None else None,
            }

        def _sheet(s: SheetManifest) -> Dict[str, Any]:
            return {
                "sheet_name": s.sheet_name,
                "classification": s.classification.classification,
                "classification_confidence": s.classification.confidence,
                "classification_reasoning": s.classification.reasoning,
                "row_count": s.row_count,
                "column_count": s.column_count,
                "columns": [_col(c) for c in s.columns],
                "sample_rows": s.sample_rows,
            }

        return {
            "file_name": self.file_name,
            "file_path": self.file_path,
            "file_type": self.file_type,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "sheets": [_sheet(s) for s in self.sheets],
            "raw_metadata": self.raw_metadata,
        }


# ---------------------------------------------------------------------------
# IngestionAgent
# ---------------------------------------------------------------------------


class IngestionAgent(BaseAgent):
    """Reads and parses uploaded files; emits FileManifest per file."""

    DEFAULT_MODEL = WORKER_MODEL

    def __init__(
        self,
        cognitive_context: Dict[str, Any],
        langfuse_handler: Any,
        model: str = WORKER_MODEL,
    ) -> None:
        super().__init__(
            model=model,
            cognitive_context=cognitive_context,
            langfuse_handler=langfuse_handler,
        )
        self._fallback_model = WORKER_FALLBACK_MODEL

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @observe(name="ingestion_agent.ingest_file")
    def ingest_file(
        self,
        file_path: str,
        minio_client: Any = None,
    ) -> FileManifest:
        """Read a file from the local filesystem or MinIO and extract a manifest.

        Args:
            file_path: Local path *or* MinIO object key (``bucket/object``).
            minio_client: Configured ``minio.Minio`` client.  If ``None`` the
                file is read directly from the local filesystem.

        Returns:
            :class:`FileManifest` describing the file.
        """
        raw_bytes: Optional[bytes] = None

        if minio_client is not None:
            # Parse "bucket/object" from the path
            parts = file_path.split("/", 1)
            bucket = parts[0]
            obj_key = parts[1] if len(parts) > 1 else file_path
            logger.info("Downloading '%s' from MinIO bucket '%s'", obj_key, bucket)
            try:
                response = minio_client.get_object(bucket, obj_key)
                raw_bytes = response.read()
                response.close()
                response.release_conn()
            except Exception as exc:
                logger.warning(
                    "MinIO download failed (%s) — falling back to local path.", exc
                )

        suffix = Path(file_path).suffix.lower()
        file_name = Path(file_path).name

        try:
            if suffix in (".xlsx", ".xls", ".xlsm"):
                file_type = "excel"
                manifest = self._process_excel(file_name, file_path, raw_bytes)
            elif suffix == ".csv":
                file_type = "csv"
                manifest = self._process_csv(file_name, file_path, raw_bytes)
            else:
                # Best-effort: try CSV
                logger.warning(
                    "Unknown extension '%s' — attempting CSV parse.", suffix
                )
                file_type = "csv"
                manifest = self._process_csv(file_name, file_path, raw_bytes)
        except Exception as exc:
            logger.error(
                "Failed to process file '%s': %s", file_path, exc, exc_info=True
            )
            # Return a minimal manifest so the pipeline can continue.
            return FileManifest(
                file_name=file_name,
                file_path=file_path,
                file_type="unknown",
                total_rows=0,
                total_columns=0,
                raw_metadata={"error": str(exc)},
            )

        logger.info(
            "Ingested '%s': %d sheets, %d total rows",
            file_name,
            len(manifest.sheets),
            manifest.total_rows,
        )
        return manifest

    @observe(name="ingestion_agent.classify_sheets")
    def classify_sheets(
        self,
        workbook_info: Dict[str, Any],
    ) -> List[SheetClassification]:
        """Use the LLM to classify sheets as data / metadata / lookup / summary.

        Args:
            workbook_info: Dict with keys ``file_name`` and ``sheets`` (list
                of sheet preview dicts containing column names and row counts).

        Returns:
            One :class:`SheetClassification` per sheet.
        """
        sheet_previews = workbook_info.get("sheets", [])
        if not sheet_previews:
            return []

        system_prompt = (
            "You are a data analyst. Classify each worksheet in an Excel workbook "
            "as one of: 'data', 'metadata', 'lookup', 'summary', or 'unknown'.\n\n"
            "Rules:\n"
            "- 'data': Contains raw transactional/operational records (many rows).\n"
            "- 'summary': Aggregated totals, pivot tables, KPI dashboards.\n"
            "- 'lookup': Reference/code tables (small, used as foreign-key lookup).\n"
            "- 'metadata': Instructions, cover pages, data dictionaries.\n"
            "- 'unknown': Cannot be determined from available info.\n\n"
            "Return ONLY valid JSON: "
            '{"classifications": [{"sheet_name": "...", "classification": "...", '
            '"confidence": 0.95, "reasoning": "..."}]}'
        )

        user_prompt = (
            f"File: {workbook_info.get('file_name', 'unknown')}\n\n"
            "Sheets:\n"
            + json.dumps(sheet_previews, indent=2)
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self._call_llm(
                messages,
                temperature=0.1,
                max_tokens=2048,
                fallback_model=self._fallback_model,
            )
            data = self._extract_json(response)
            return [
                SheetClassification(
                    sheet_name=c["sheet_name"],
                    classification=c.get("classification", "unknown"),
                    confidence=float(c.get("confidence", 0.5)),
                    reasoning=c.get("reasoning", ""),
                )
                for c in data.get("classifications", [])
            ]
        except Exception as exc:
            logger.warning("classify_sheets LLM call failed: %s", exc)
            # Fall back: classify all sheets as "unknown"
            return [
                SheetClassification(
                    sheet_name=s.get("sheet_name", "Sheet"),
                    classification="unknown",
                    confidence=0.0,
                    reasoning="Classification failed — LLM error.",
                )
                for s in sheet_previews
            ]

    @observe(name="ingestion_agent.extract_raw_data")
    def extract_raw_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract column metadata and sample rows from a DataFrame.

        Args:
            df: The sheet/table as a pandas DataFrame.

        Returns:
            Dict with ``columns``, ``sample_rows``, ``row_count``, and ``dtypes``.
        """
        columns: List[Dict[str, Any]] = []
        for col in df.columns:
            series = df[col]
            col_info: Dict[str, Any] = {
                "name": str(col),
                "dtype": str(series.dtype),
                "non_null_count": int(series.count()),
                "null_count": int(series.isnull().sum()),
                "unique_count": int(series.nunique()),
                "sample_values": [str(v) for v in series.dropna().head(5).tolist()],
            }
            if pd.api.types.is_numeric_dtype(series):
                col_info["min_value"] = (
                    float(series.min()) if not series.empty else None
                )
                col_info["max_value"] = (
                    float(series.max()) if not series.empty else None
                )
            columns.append(col_info)

        sample_rows = (
            df.head(5)
            .fillna("")
            .astype(str)
            .to_dict(orient="records")
        )

        return {
            "columns": columns,
            "sample_rows": sample_rows,
            "row_count": len(df),
            "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
        }

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    @observe(name="ingestion_agent.run")
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process all uploaded files and populate ``state["file_manifests"]``.

        Expected state keys:
            - ``uploaded_files`` (list of str): Local paths or MinIO keys.
            - ``minio_client`` (optional): Configured MinIO client.
            - ``job_id`` (str): Pipeline job identifier.
            - ``redis_client`` (optional): Redis client for event streaming.

        Returns:
            State with ``file_manifests`` list added/updated.
        """
        job_id = state.get("job_id", self._job_id())
        redis_client = state.get("redis_client")
        minio_client = state.get("minio_client")
        uploaded_files: List[str] = state.get("uploaded_files", [])

        self._emit_event(
            "ingestion_started",
            {"file_count": len(uploaded_files)},
            job_id,
            redis_client,
        )

        file_manifests: List[Dict[str, Any]] = []
        for idx, file_path in enumerate(uploaded_files):
            logger.info(
                "Ingesting file %d/%d: %s", idx + 1, len(uploaded_files), file_path
            )
            self._emit_event(
                "ingesting_file",
                {"file": file_path, "index": idx, "total": len(uploaded_files)},
                job_id,
                redis_client,
            )
            manifest = self.ingest_file(file_path, minio_client)
            file_manifests.append(manifest.to_dict())
            self._emit_event(
                "file_ingested",
                {
                    "file": manifest.file_name,
                    "sheets": len(manifest.sheets),
                    "total_rows": manifest.total_rows,
                },
                job_id,
                redis_client,
            )

        state["file_manifests"] = file_manifests
        self._emit_event(
            "ingestion_completed",
            {"files_processed": len(file_manifests)},
            job_id,
            redis_client,
        )
        logger.info("IngestionAgent completed — %d files processed.", len(file_manifests))
        return state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_excel(
        self,
        file_name: str,
        file_path: str,
        raw_bytes: Optional[bytes],
    ) -> FileManifest:
        """Parse an Excel workbook and return a FileManifest."""
        source: Any = io.BytesIO(raw_bytes) if raw_bytes else file_path
        xls = pd.ExcelFile(source)
        sheet_names: List[str] = xls.sheet_names

        # Build lightweight preview for LLM classification
        previews: List[Dict[str, Any]] = []
        dfs: Dict[str, pd.DataFrame] = {}
        for name in sheet_names:
            try:
                df = pd.read_excel(source, sheet_name=name, nrows=100)
                dfs[name] = df
                previews.append(
                    {
                        "sheet_name": name,
                        "row_count_preview": len(df),
                        "column_names": df.columns.tolist(),
                    }
                )
            except Exception as exc:
                logger.warning("Could not read sheet '%s': %s", name, exc)
                previews.append(
                    {"sheet_name": name, "row_count_preview": 0, "column_names": []}
                )

        classifications_list = self.classify_sheets(
            {"file_name": file_name, "sheets": previews}
        )
        classifications: Dict[str, SheetClassification] = {
            c.sheet_name: c for c in classifications_list
        }

        sheets: List[SheetManifest] = []
        total_rows = 0
        total_columns = 0

        for name in sheet_names:
            df = dfs.get(name)
            if df is None or df.empty:
                continue

            # Re-read full sheet for non-metadata sheets
            classification = classifications.get(
                name,
                SheetClassification(name, "unknown", 0.0),
            )
            if classification.classification not in ("metadata",):
                try:
                    df_full = pd.read_excel(source, sheet_name=name)
                except Exception:
                    df_full = df
            else:
                df_full = df

            raw = self.extract_raw_data(df_full)
            columns = [
                ColumnInfo(
                    name=c["name"],
                    dtype=c["dtype"],
                    non_null_count=c["non_null_count"],
                    null_count=c["null_count"],
                    unique_count=c["unique_count"],
                    sample_values=c["sample_values"],
                    min_value=c.get("min_value"),
                    max_value=c.get("max_value"),
                )
                for c in raw["columns"]
            ]

            sheet_manifest = SheetManifest(
                sheet_name=name,
                classification=classification,
                row_count=raw["row_count"],
                column_count=len(columns),
                columns=columns,
                sample_rows=raw["sample_rows"],
            )
            sheets.append(sheet_manifest)
            total_rows += raw["row_count"]
            total_columns = max(total_columns, len(columns))

        return FileManifest(
            file_name=file_name,
            file_path=file_path,
            file_type="excel",
            total_rows=total_rows,
            total_columns=total_columns,
            sheets=sheets,
            raw_metadata={"sheet_count": len(sheet_names)},
        )

    def _process_csv(
        self,
        file_name: str,
        file_path: str,
        raw_bytes: Optional[bytes],
    ) -> FileManifest:
        """Parse a CSV file and return a FileManifest."""
        source: Any = io.BytesIO(raw_bytes) if raw_bytes else file_path
        try:
            df = pd.read_csv(source)
        except UnicodeDecodeError:
            # Retry with latin-1
            if raw_bytes:
                source = io.BytesIO(raw_bytes)
            df = pd.read_csv(source, encoding="latin-1")

        raw = self.extract_raw_data(df)
        columns = [
            ColumnInfo(
                name=c["name"],
                dtype=c["dtype"],
                non_null_count=c["non_null_count"],
                null_count=c["null_count"],
                unique_count=c["unique_count"],
                sample_values=c["sample_values"],
                min_value=c.get("min_value"),
                max_value=c.get("max_value"),
            )
            for c in raw["columns"]
        ]
        classification = SheetClassification(
            sheet_name="default",
            classification="data",
            confidence=1.0,
            reasoning="CSV files contain a single data sheet by convention.",
        )
        sheet = SheetManifest(
            sheet_name="default",
            classification=classification,
            row_count=raw["row_count"],
            column_count=len(columns),
            columns=columns,
            sample_rows=raw["sample_rows"],
        )
        return FileManifest(
            file_name=file_name,
            file_path=file_path,
            file_type="csv",
            total_rows=raw["row_count"],
            total_columns=len(columns),
            sheets=[sheet],
            raw_metadata={},
        )
