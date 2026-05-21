"""
ValidationAgent - Cross-checks totals, runs business-rule validations,
and flags anomalies in the merged dataset using Docker sandbox execution.
"""

from __future__ import annotations

import json
import logging
import os
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List

from langfuse.decorators import observe

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

VALIDATOR_MODEL = os.getenv("VALIDATOR_MODEL", "ollama/llama3.3:70b")
VALIDATOR_FALLBACK = os.getenv("VALIDATOR_FALLBACK", "openai/gpt-4o-mini")


@dataclass
class ValidationCheck:
    name: str
    passed: bool
    expected: Any = None
    actual: Any = None
    message: str = ""


@dataclass
class Anomaly:
    row_id: Any
    column: str
    severity: str  # "info" | "warning" | "critical"
    description: str
    value: Any = None


@dataclass
class ValidationReport:
    passed: bool
    checks: List[ValidationCheck] = field(default_factory=list)
    anomalies: List[Anomaly] = field(default_factory=list)
    row_count: int = 0
    total_value: float = 0.0
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "row_count": self.row_count,
            "total_value": self.total_value,
            "summary": self.summary,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "expected": c.expected,
                    "actual": c.actual,
                    "message": c.message,
                }
                for c in self.checks
            ],
            "anomalies": [
                {
                    "row_id": a.row_id,
                    "column": a.column,
                    "severity": a.severity,
                    "description": a.description,
                    "value": a.value,
                }
                for a in self.anomalies
            ],
        }


class ValidationAgent(BaseAgent):
    """Validates the merged dataset against business rules and data quality checks."""

    DEFAULT_MODEL: str = VALIDATOR_MODEL

    def __init__(
        self,
        cognitive_context: Dict[str, Any] = None,
        langfuse_handler: Any = None,
        model: str = None,
    ) -> None:
        super().__init__(
            model=model or VALIDATOR_MODEL,
            cognitive_context=cognitive_context or {},
            langfuse_handler=langfuse_handler,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @observe(name="validation.run")
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run validation on the merged dataset.

        Expected state keys:
            - merged_path: parquet path of consolidated data
            - file_manifests: list of original file manifests (for cross-check)
            - cognitive_ctx: cognitive context (business rules etc.)
            - job_id: pipeline job id

        Adds to state:
            - validation: ValidationReport dict
        """
        merged_path = state.get("merged_path")
        manifests = state.get("file_manifests", [])
        job_id = state.get("job_id", self._job_id())
        redis_client = state.get("redis_client")

        if not merged_path or not os.path.exists(merged_path):
            report = ValidationReport(
                passed=False,
                summary="No merged dataset path provided",
            )
            state["validation"] = report.to_dict()
            return state

        self._emit_event(
            "agent_start",
            {"agent": "validation", "merged_path": merged_path},
            job_id,
            redis_client,
        )

        business_rules = self._extract_business_rules()
        code = self.generate_validation_code(merged_path, manifests, business_rules)

        self._emit_event(
            "code_generated",
            {"agent": "validation", "code": code},
            job_id,
            redis_client,
        )

        result = self.run_validations(code, merged_path)
        report = self._parse_report(result, manifests)

        state["validation"] = report.to_dict()

        self._emit_event(
            "validation",
            report.to_dict(),
            job_id,
            redis_client,
        )

        self._emit_event(
            "agent_complete",
            {"agent": "validation", "passed": report.passed},
            job_id,
            redis_client,
        )
        return state

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------

    def generate_validation_code(
        self,
        merged_path: str,
        manifests: List[Dict[str, Any]],
        business_rules: str,
    ) -> str:
        """Use the LLM to generate sandbox-safe validation code."""
        manifest_summary = json.dumps(
            [
                {
                    "file_id": m.get("file_id"),
                    "row_count": m.get("row_count"),
                    "columns": m.get("columns", []),
                    "total_value": m.get("total_value"),
                }
                for m in manifests
            ],
            indent=2,
        )

        system = textwrap.dedent(
            f"""
            You are the Validation Agent of an open-source agentic data platform.

            Generate Python code that runs in a sandboxed Docker container with
            pandas, numpy, scipy, and duckdb available.

            Your code MUST:
            1. Load the merged parquet at: /data/merged.parquet
            2. Run validation checks and write results to /out/validation.json
            3. Print only the JSON to stdout (single line)
            4. Never hallucinate numbers — every value must come from the data

            Validation checks to perform:
            - Row count matches sum of source row counts (within tolerance)
            - Numeric totals are reasonable (no negatives unless expected)
            - No duplicate primary keys
            - Required columns are non-null
            - Categorical values are within expected sets
            - Statistical outliers (>3 sigma) flagged as anomalies
            - Apply business rules below

            Business rules:
            {business_rules}

            Source manifest summary:
            {manifest_summary}

            Output JSON schema:
            {{
              "passed": bool,
              "row_count": int,
              "total_value": float,
              "summary": str,
              "checks": [{{"name": str, "passed": bool, "expected": any, "actual": any, "message": str}}],
              "anomalies": [{{"row_id": any, "column": str, "severity": "info"|"warning"|"critical", "description": str, "value": any}}]
            }}

            Return ONLY the Python code. No markdown, no commentary.
            """
        ).strip()

        user = f"Generate validation code for the merged dataset at {merged_path}."

        response = self._call_llm(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=3000,
            fallback_model=VALIDATOR_FALLBACK,
        )
        code = self._extract_text(response)

        # Strip markdown fences if any
        for fence in ("```python", "```"):
            if code.startswith(fence):
                code = code[len(fence):]
        if code.endswith("```"):
            code = code[:-3]
        return code.strip()

    # ------------------------------------------------------------------
    # Sandbox execution
    # ------------------------------------------------------------------

    def run_validations(self, code: str, merged_path: str) -> Dict[str, Any]:
        """Execute validation code in Docker sandbox and return JSON result."""
        try:
            from tools.execute_python import SandboxExecutor
            executor = SandboxExecutor()
            result = executor.execute(
                code=code,
                input_files={"merged.parquet": merged_path},
                timeout=60,
            )

            if result.get("exit_code") != 0:
                logger.warning(
                    "Validation sandbox returned non-zero exit: %s",
                    result.get("stderr"),
                )
                return {
                    "passed": False,
                    "summary": f"Sandbox error: {result.get('stderr', 'unknown')[:200]}",
                    "checks": [],
                    "anomalies": [],
                    "row_count": 0,
                    "total_value": 0.0,
                }

            stdout = result.get("stdout", "").strip()
            try:
                return json.loads(stdout)
            except json.JSONDecodeError:
                # Try to find JSON in stdout
                idx = stdout.find("{")
                if idx >= 0:
                    try:
                        return json.loads(stdout[idx:])
                    except Exception:
                        pass
                return {
                    "passed": False,
                    "summary": "Validation output was not valid JSON",
                    "checks": [],
                    "anomalies": [],
                    "row_count": 0,
                    "total_value": 0.0,
                }
        except Exception as exc:
            logger.error("Validation execution failed: %s", exc, exc_info=True)
            return {
                "passed": False,
                "summary": f"Execution failed: {exc}",
                "checks": [],
                "anomalies": [],
                "row_count": 0,
                "total_value": 0.0,
            }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def check_totals(
        self,
        merged_total: float,
        manifest_totals: List[float],
        tolerance: float = 0.01,
    ) -> ValidationCheck:
        """Cross-check sum-of-sources against merged total."""
        expected = sum(manifest_totals)
        diff = abs(expected - merged_total)
        rel = diff / max(abs(expected), 1e-9)
        passed = rel <= tolerance
        return ValidationCheck(
            name="totals_match",
            passed=passed,
            expected=expected,
            actual=merged_total,
            message=(
                f"Diff {diff:.2f} ({rel * 100:.3f}%) within tolerance {tolerance * 100:.2f}%"
                if passed
                else f"Total mismatch: {diff:.2f} ({rel * 100:.3f}%) exceeds tolerance"
            ),
        )

    def flag_anomalies(self, df_summary: Dict[str, Any]) -> List[Anomaly]:
        """Generate a basic list of anomalies from a dataframe summary dict."""
        anomalies: List[Anomaly] = []
        for col, stats in df_summary.get("columns", {}).items():
            null_pct = stats.get("null_pct", 0)
            if null_pct > 0.2:
                anomalies.append(
                    Anomaly(
                        row_id="*",
                        column=col,
                        severity="warning",
                        description=f"Column {col} has {null_pct * 100:.1f}% nulls",
                        value=null_pct,
                    )
                )
        return anomalies

    def _parse_report(
        self,
        raw: Dict[str, Any],
        manifests: List[Dict[str, Any]],
    ) -> ValidationReport:
        """Convert raw sandbox JSON output into a structured ValidationReport."""
        checks = [
            ValidationCheck(
                name=c.get("name", "unnamed"),
                passed=bool(c.get("passed", False)),
                expected=c.get("expected"),
                actual=c.get("actual"),
                message=c.get("message", ""),
            )
            for c in raw.get("checks", [])
        ]
        anomalies = [
            Anomaly(
                row_id=a.get("row_id"),
                column=a.get("column", ""),
                severity=a.get("severity", "info"),
                description=a.get("description", ""),
                value=a.get("value"),
            )
            for a in raw.get("anomalies", [])
        ]

        # Add cross-check using manifests if total_value present
        manifest_totals = [
            m.get("total_value", 0) or 0
            for m in manifests
            if m.get("total_value") is not None
        ]
        merged_total = raw.get("total_value", 0) or 0
        if manifest_totals and merged_total:
            checks.append(self.check_totals(merged_total, manifest_totals))

        passed = bool(raw.get("passed", False)) and all(c.passed for c in checks)

        return ValidationReport(
            passed=passed,
            checks=checks,
            anomalies=anomalies,
            row_count=int(raw.get("row_count", 0)),
            total_value=float(merged_total),
            summary=raw.get("summary", ""),
        )

    def _extract_business_rules(self) -> str:
        """Pull the business rules block from cognitive context."""
        ctx = self.cognitive_context
        return ctx.get("business_rules") or ctx.get("knowledge", {}).get(
            "business-rules.md", ""
        ) or ""
