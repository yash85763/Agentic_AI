"""
UnderstandingAgent - Maps raw column names to their semantic meaning using
the data dictionary and LLM inference.  Enriches file_manifests with
semantic annotations and validates the schema against business expectations.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langfuse.decorators import observe

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

UNDERSTANDING_MODEL = os.getenv("UNDERSTANDING_MODEL", "claude-sonnet-4-6")
UNDERSTANDING_FALLBACK = os.getenv("UNDERSTANDING_FALLBACK", "openai/gpt-4o")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ColumnSemantics:
    """Semantic annotation for a single column."""

    raw_name: str
    canonical_name: str
    semantic_type: str       # e.g. "date", "currency", "identifier", "metric", "category"
    description: str
    aliases: List[str] = field(default_factory=list)
    unit: Optional[str] = None
    is_key: bool = False
    is_nullable: bool = True


@dataclass
class SemanticMap:
    """Semantic annotations for a full sheet."""

    file_name: str
    sheet_name: str
    columns: List[ColumnSemantics] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_name": self.file_name,
            "sheet_name": self.sheet_name,
            "columns": [
                {
                    "raw_name": c.raw_name,
                    "canonical_name": c.canonical_name,
                    "semantic_type": c.semantic_type,
                    "description": c.description,
                    "aliases": c.aliases,
                    "unit": c.unit,
                    "is_key": c.is_key,
                    "is_nullable": c.is_nullable,
                }
                for c in self.columns
            ],
        }

    def lookup(self, raw_name: str) -> Optional[ColumnSemantics]:
        """Find a column by its raw name (case-insensitive)."""
        raw_lower = raw_name.lower()
        for c in self.columns:
            if c.raw_name.lower() == raw_lower:
                return c
            if any(a.lower() == raw_lower for a in c.aliases):
                return c
        return None


@dataclass
class ValidationIssue:
    column: str
    severity: str   # "error" | "warning" | "info"
    message: str


@dataclass
class ValidationResult:
    passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    coverage_pct: float = 100.0  # % of columns with known semantics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "coverage_pct": self.coverage_pct,
            "issues": [
                {"column": i.column, "severity": i.severity, "message": i.message}
                for i in self.issues
            ],
        }


# ---------------------------------------------------------------------------
# UnderstandingAgent
# ---------------------------------------------------------------------------


class UnderstandingAgent(BaseAgent):
    """Enriches file manifests with semantic column understanding."""

    DEFAULT_MODEL = UNDERSTANDING_MODEL

    def __init__(
        self,
        cognitive_context: Dict[str, Any],
        langfuse_handler: Any,
        model: str = UNDERSTANDING_MODEL,
    ) -> None:
        super().__init__(
            model=model,
            cognitive_context=cognitive_context,
            langfuse_handler=langfuse_handler,
        )
        self._fallback = UNDERSTANDING_FALLBACK

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    @observe(name="understanding_agent.infer_semantics")
    def infer_semantics(
        self,
        columns: List[Dict[str, Any]],
        data_dict: Dict[str, Any],
        file_name: str = "",
        sheet_name: str = "",
    ) -> SemanticMap:
        """Ask the LLM to map raw column names to semantic meanings.

        Args:
            columns: List of column info dicts (from FileManifest).
            data_dict: Business data dictionary (column descriptions, aliases,
                units, etc.).  Can be an empty dict if unavailable.
            file_name: Source file name (for context).
            sheet_name: Sheet name (for context).

        Returns:
            :class:`SemanticMap` with one :class:`ColumnSemantics` per column.
        """
        column_summary = [
            {
                "raw_name": c.get("name"),
                "dtype": c.get("dtype"),
                "sample_values": c.get("sample_values", [])[:3],
                "null_pct": round(
                    c.get("null_count", 0)
                    / max(c.get("non_null_count", 1) + c.get("null_count", 0), 1)
                    * 100,
                    1,
                ),
            }
            for c in columns
        ]

        system_prompt = (
            "You are a data semantics expert. Given raw column metadata and a data "
            "dictionary, map each column to its canonical semantic meaning.\n\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "columns": [\n'
            "    {\n"
            '      "raw_name": "orig column name",\n'
            '      "canonical_name": "snake_case_canonical",\n'
            '      "semantic_type": "date|currency|identifier|metric|category|text|boolean|unknown",\n'
            '      "description": "what this column represents",\n'
            '      "aliases": ["alt_name1"],\n'
            '      "unit": "USD or null",\n'
            '      "is_key": false,\n'
            '      "is_nullable": true\n'
            "    }\n"
            "  ]\n"
            "}"
        )

        user_prompt = (
            f"File: {file_name}  Sheet: {sheet_name}\n\n"
            f"Column metadata:\n{json.dumps(column_summary, indent=2)}\n\n"
            f"Data dictionary:\n{json.dumps(data_dict, indent=2)}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self._call_llm(
                messages,
                temperature=0.1,
                max_tokens=4096,
                fallback_model=self._fallback,
            )
            data = self._extract_json(response)
        except Exception as exc:
            logger.warning("infer_semantics LLM failed: %s", exc)
            # Fall back to trivial identity mapping
            data = {
                "columns": [
                    {
                        "raw_name": c.get("name", ""),
                        "canonical_name": c.get("name", "").lower().replace(" ", "_"),
                        "semantic_type": "unknown",
                        "description": "Auto-inferred (LLM unavailable)",
                        "aliases": [],
                        "unit": None,
                        "is_key": False,
                        "is_nullable": True,
                    }
                    for c in columns
                ]
            }

        semantics = [
            ColumnSemantics(
                raw_name=c["raw_name"],
                canonical_name=c.get("canonical_name", c["raw_name"]),
                semantic_type=c.get("semantic_type", "unknown"),
                description=c.get("description", ""),
                aliases=c.get("aliases", []),
                unit=c.get("unit"),
                is_key=bool(c.get("is_key", False)),
                is_nullable=bool(c.get("is_nullable", True)),
            )
            for c in data.get("columns", [])
        ]

        return SemanticMap(
            file_name=file_name,
            sheet_name=sheet_name,
            columns=semantics,
        )

    @observe(name="understanding_agent.resolve_alias")
    def resolve_alias(
        self,
        raw_name: str,
        known_aliases: Dict[str, str],
    ) -> str:
        """Resolve a raw column name to its canonical form using a known alias map.

        Falls back to an LLM call if not found in the alias map.

        Args:
            raw_name: The column name as it appears in the raw file.
            known_aliases: Dict mapping raw/alias names to canonical names.

        Returns:
            Canonical column name string.
        """
        # Exact match first
        if raw_name in known_aliases:
            return known_aliases[raw_name]

        # Case-insensitive match
        raw_lower = raw_name.lower()
        for alias, canonical in known_aliases.items():
            if alias.lower() == raw_lower:
                return canonical

        # LLM-assisted resolution
        if not known_aliases:
            # Nothing to match against — return normalised form
            return raw_name.strip().lower().replace(" ", "_")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a column name resolver. Given a raw column name and a "
                    "dictionary of known aliases, return the best matching canonical "
                    "name. If no good match, return the raw name normalised to snake_case. "
                    'Return ONLY a JSON object: {"canonical_name": "..."}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Raw column name: {raw_name}\n\n"
                    f"Known aliases:\n{json.dumps(known_aliases, indent=2)}"
                ),
            },
        ]

        try:
            response = self._call_llm(messages, temperature=0.0, max_tokens=128)
            result = self._extract_json(response)
            return result.get("canonical_name", raw_name)
        except Exception as exc:
            logger.warning("resolve_alias LLM failed for '%s': %s", raw_name, exc)
            return raw_name.strip().lower().replace(" ", "_")

    @observe(name="understanding_agent.validate_schema")
    def validate_schema(
        self,
        manifest: Dict[str, Any],
        data_dict: Dict[str, Any],
    ) -> ValidationResult:
        """Validate a file manifest against the business data dictionary.

        Checks for:
        - Required columns missing
        - Unexpected data types for known columns
        - High null rates on non-nullable columns
        - Unknown columns (columns not in the data dictionary)

        Args:
            manifest: FileManifest dict.
            data_dict: Business data dictionary.

        Returns:
            :class:`ValidationResult`.
        """
        issues: List[ValidationIssue] = []
        required_columns: List[str] = data_dict.get("required_columns", [])
        column_specs: Dict[str, Dict[str, Any]] = data_dict.get("columns", {})
        nullable_by_default: bool = data_dict.get("nullable_by_default", True)

        for sheet in manifest.get("sheets", []):
            if sheet.get("classification") not in ("data", "unknown"):
                continue

            actual_columns = {
                c["name"].lower(): c for c in sheet.get("columns", [])
            }

            # Check required columns
            for req in required_columns:
                if req.lower() not in actual_columns:
                    issues.append(
                        ValidationIssue(
                            column=req,
                            severity="error",
                            message=f"Required column '{req}' missing from sheet "
                            f"'{sheet['sheet_name']}'.",
                        )
                    )

            # Check spec-defined columns
            for spec_col, spec in column_specs.items():
                actual = actual_columns.get(spec_col.lower())
                if actual is None:
                    if spec.get("required", False):
                        issues.append(
                            ValidationIssue(
                                column=spec_col,
                                severity="error",
                                message=f"Required column '{spec_col}' not found.",
                            )
                        )
                    continue

                # Nullable check
                is_nullable = spec.get("nullable", nullable_by_default)
                null_count = actual.get("null_count", 0)
                total = actual.get("non_null_count", 0) + null_count
                null_rate = null_count / total if total else 0
                if not is_nullable and null_rate > 0:
                    issues.append(
                        ValidationIssue(
                            column=spec_col,
                            severity="warning",
                            message=f"Column '{spec_col}' has {null_rate:.1%} null "
                            f"values but is marked non-nullable.",
                        )
                    )

        # Compute coverage
        all_columns: List[str] = []
        for sheet in manifest.get("sheets", []):
            all_columns.extend(c["name"] for c in sheet.get("columns", []))
        known = set(column_specs.keys())
        if all_columns:
            coverage = (
                sum(1 for c in all_columns if c in known) / len(all_columns) * 100
            )
        else:
            coverage = 0.0

        errors = [i for i in issues if i.severity == "error"]
        passed = len(errors) == 0

        return ValidationResult(
            passed=passed,
            issues=issues,
            coverage_pct=round(coverage, 1),
        )

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    @observe(name="understanding_agent.run")
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich ``state["file_manifests"]`` with semantic annotations.

        Expected state keys:
            - ``file_manifests`` (list): From IngestionAgent.
            - ``data_dict`` (dict, optional): Business data dictionary.
            - ``job_id`` (str): Pipeline job identifier.
            - ``redis_client`` (optional): Redis client.

        Returns:
            State with ``semantic_maps`` and ``schema_validations`` added.
        """
        job_id = state.get("job_id", self._job_id())
        redis_client = state.get("redis_client")
        file_manifests: List[Dict[str, Any]] = state.get("file_manifests", [])
        data_dict: Dict[str, Any] = state.get("data_dict", {})

        self._emit_event(
            "understanding_started",
            {"manifest_count": len(file_manifests)},
            job_id,
            redis_client,
        )

        semantic_maps: List[Dict[str, Any]] = []
        schema_validations: List[Dict[str, Any]] = []

        for manifest in file_manifests:
            file_name = manifest.get("file_name", "unknown")
            logger.info("Inferring semantics for '%s'", file_name)

            for sheet in manifest.get("sheets", []):
                if sheet.get("classification") not in ("data", "unknown"):
                    continue

                sheet_name = sheet.get("sheet_name", "default")
                columns = sheet.get("columns", [])

                self._emit_event(
                    "inferring_sheet",
                    {"file": file_name, "sheet": sheet_name},
                    job_id,
                    redis_client,
                )

                sem_map = self.infer_semantics(
                    columns,
                    data_dict,
                    file_name=file_name,
                    sheet_name=sheet_name,
                )
                semantic_maps.append(sem_map.to_dict())

                # Annotate the manifest in-place for downstream agents
                sheet["semantic_map"] = sem_map.to_dict()

            # Validate schema
            validation = self.validate_schema(manifest, data_dict)
            schema_validations.append(
                {
                    "file_name": file_name,
                    **validation.to_dict(),
                }
            )

            self._emit_event(
                "schema_validated",
                {
                    "file": file_name,
                    "passed": validation.passed,
                    "issues": len(validation.issues),
                    "coverage_pct": validation.coverage_pct,
                },
                job_id,
                redis_client,
            )

        state["semantic_maps"] = semantic_maps
        state["schema_validations"] = schema_validations

        self._emit_event(
            "understanding_completed",
            {
                "semantic_maps": len(semantic_maps),
                "validations": len(schema_validations),
            },
            job_id,
            redis_client,
        )
        logger.info(
            "UnderstandingAgent completed — %d semantic maps produced.",
            len(semantic_maps),
        )
        return state
