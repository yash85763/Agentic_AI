"""
CausalAgent — gated causal analysis with mandatory assumption declaration.

Causal analysis is OPT-IN. The agent enforces three gates before making
any causal claim:
  1. causal_mode_enabled == True in pipeline state
  2. Dataset row count >= 100
  3. DoWhy identification returns a valid estimand

Every output carries a ConfidenceBundle. causal_status defaults to
correlational_only and is only upgraded to 'causal' if all three gates pass.
Outputs include a visible label: 'CAUSAL — assumptions verified' or
'CORRELATIONAL — not causal'.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langfuse.decorators import observe

from agents.base_agent import BaseAgent
from models.confidence import (
    AnswerType,
    CausalStatus,
    ConfidenceBundle,
    ContractStatus,
    SourceReference,
    TrustTier,
)

logger = logging.getLogger(__name__)

CAUSAL_MODEL = os.getenv("CAUSAL_MODEL", "claude-sonnet-4-6")
MIN_ROWS_FOR_CAUSAL = 100
MAX_TOOLS = 4  # per Section 6.1 — specialist agents have <= 4 tools


# ---------------------------------------------------------------------------
# Assumption declaration model
# ---------------------------------------------------------------------------


@dataclass
class CausalAssumptions:
    """Declared assumptions for a causal analysis run.

    These MUST be shown to the user (via SSE) before computation runs.
    They are written to the ConfidenceBundle.assumptions field regardless.
    """

    cause_variables: List[str]
    effect_variables: List[str]
    confounders: List[str]
    time_lag_assumption: str
    dataset_row_count: int

    def to_statement_list(self) -> List[str]:
        return [
            f"Cause variables (treated as independent): {', '.join(self.cause_variables) or 'none identified'}",
            f"Effect variables (treated as dependent): {', '.join(self.effect_variables) or 'none identified'}",
            f"Declared confounders: {', '.join(self.confounders) or 'none declared'}",
            f"Time lag assumption: {self.time_lag_assumption}",
            f"Dataset row count: {self.dataset_row_count} "
            f"(minimum required for causal claims: {MIN_ROWS_FOR_CAUSAL})",
        ]

    def to_event_dict(self) -> Dict[str, Any]:
        return {
            "cause_variables": self.cause_variables,
            "effect_variables": self.effect_variables,
            "confounders": self.confounders,
            "time_lag_assumption": self.time_lag_assumption,
            "dataset_row_count": self.dataset_row_count,
            "minimum_rows_required": MIN_ROWS_FOR_CAUSAL,
        }


# ---------------------------------------------------------------------------
# CausalAgent
# ---------------------------------------------------------------------------


class CausalAgent(BaseAgent):
    """Performs causal or correlational analysis on structured financial data.

    The agent has at most MAX_TOOLS (4) tools:
      1. declare_assumptions — LLM-assisted assumption identification
      2. compute_correlations — Pearson correlation via pandas sandbox
      3. run_dowhy — DoWhy causal identification and estimation
      4. run (orchestrates the above)

    Never call run_dowhy directly — it is gated by run().
    """

    DEFAULT_MODEL = CAUSAL_MODEL

    def __init__(
        self,
        cognitive_context: Dict[str, Any],
        langfuse_handler: Any,
        model: str = CAUSAL_MODEL,
    ) -> None:
        super().__init__(
            model=model,
            cognitive_context=cognitive_context,
            langfuse_handler=langfuse_handler,
        )

    # ------------------------------------------------------------------
    # Tool 1: Assumption declaration
    # ------------------------------------------------------------------

    @observe(name="causal_agent.declare_assumptions")
    def declare_assumptions(
        self,
        data_summary: Dict[str, Any],
        task: str,
        job_id: str,
        redis_client: Any = None,
    ) -> CausalAssumptions:
        """Use the LLM to identify causal assumptions before any computation.

        Emits 'causal_assumptions_declared' SSE event so the frontend can
        show the assumptions to the user before computation proceeds.
        """
        columns = data_summary.get("columns", [])
        row_count = data_summary.get("row_count", 0)

        system_prompt = (
            "You are a causal inference specialist. Given a dataset summary and task, "
            "identify the causal assumptions needed.\n\n"
            "Return ONLY valid JSON — no prose, no markdown fences.\n\n"
            'Schema: {"cause_variables":["col1"],"effect_variables":["col2"],'
            '"confounders":["col3"],"time_lag_assumption":"no lag assumed",'
            '"reasoning":"why these assumptions are appropriate"}'
        )

        user_prompt = (
            f"Task: {task}\n\n"
            f"Dataset columns: {', '.join(str(c) for c in columns[:30])}\n"
            f"Row count: {row_count}\n\n"
            "Declare the minimal causal assumptions. Be conservative — only claim "
            "causality where it is scientifically plausible given the column names."
        )

        try:
            response = self._call_llm(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": user_prompt}],
                temperature=0.1,
                max_tokens=1024,
            )
            data = self._extract_json(response)
        except Exception as exc:
            logger.warning("Assumption declaration LLM call failed: %s", exc)
            data = {
                "cause_variables": [],
                "effect_variables": [],
                "confounders": [],
                "time_lag_assumption": "unknown — LLM call failed",
            }

        assumptions = CausalAssumptions(
            cause_variables=data.get("cause_variables", []),
            effect_variables=data.get("effect_variables", []),
            confounders=data.get("confounders", []),
            time_lag_assumption=data.get("time_lag_assumption", "unknown"),
            dataset_row_count=row_count,
        )

        # Emit to SSE stream BEFORE computation — user must see these
        self._emit_event(
            "causal_assumptions_declared",
            {
                "message": (
                    "Review causal assumptions before computation proceeds. "
                    "These assumptions determine whether results are causal or correlational."
                ),
                "assumptions": assumptions.to_event_dict(),
            },
            job_id,
            redis_client,
        )

        return assumptions

    # ------------------------------------------------------------------
    # Tool 2: Correlation analysis (no gating — always available)
    # ------------------------------------------------------------------

    @observe(name="causal_agent.compute_correlations")
    def compute_correlations(
        self,
        merged_path: str,
        numeric_columns: List[str],
    ) -> Dict[str, Any]:
        """Run Pearson correlation analysis. Explicitly CORRELATIONAL — not causal."""
        from tools.execute_python import execute_python

        if not merged_path or not numeric_columns:
            return {"causal_status": "correlational_only", "correlations": {}}

        cols_repr = repr(numeric_columns[:10])  # limit to 10 cols for safety
        code = f"""
import pandas as pd
import json, sys

df = pd.read_parquet('/sandbox/inputs/merged.parquet')
cols = {cols_repr}
available = [c for c in cols if c in df.columns]

if len(available) < 2:
    print(json.dumps({{"error": "fewer than 2 numeric columns available",
                       "available": df.columns.tolist()}}))
    sys.exit(0)

corr = df[available].corr(method='pearson').round(4).to_dict()
print(json.dumps({{"correlations": corr, "row_count": len(df), "columns": available}}))
"""
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                execute_python(code, {"merged.parquet": merged_path})
            )
        finally:
            loop.close()

        if result.get("ast_rejected"):
            return {"causal_status": "correlational_only",
                    "error": f"Code rejected: {result.get('rejection_reason')}"}

        if result["exit_code"] != 0:
            return {"causal_status": "correlational_only", "error": result["stderr"]}

        try:
            import json
            parsed = json.loads(result["stdout"])
            return {**parsed, "causal_status": "correlational_only"}
        except Exception:
            return {"causal_status": "correlational_only", "error": "Parse failed"}

    # ------------------------------------------------------------------
    # Tool 3: DoWhy causal identification (gated — only called from run())
    # ------------------------------------------------------------------

    @observe(name="causal_agent.run_dowhy")
    def run_dowhy(
        self,
        merged_path: str,
        assumptions: CausalAssumptions,
    ) -> Dict[str, Any]:
        """Run DoWhy causal identification and linear regression estimation.

        Returns causal_status='causal' only when DoWhy finds a valid estimand
        and the estimation succeeds. Returns 'correlational_only' on any failure.
        """
        from tools.execute_python import execute_python

        cause_repr = repr(assumptions.cause_variables)
        effect_repr = repr(assumptions.effect_variables)
        confounder_repr = repr(assumptions.confounders)

        code = f"""
import pandas as pd
import json, sys

try:
    import dowhy
    from dowhy import CausalModel
except ImportError:
    print(json.dumps({{"error": "dowhy not installed in sandbox",
                       "causal_status": "correlational_only"}}))
    sys.exit(0)

df = pd.read_parquet('/sandbox/inputs/merged.parquet')

cause_vars  = {cause_repr}
effect_vars = {effect_repr}
confounders = {confounder_repr}

if not cause_vars or not effect_vars:
    print(json.dumps({{"error": "no cause/effect variables declared",
                       "causal_status": "correlational_only"}}))
    sys.exit(0)

treatment = cause_vars[0]
outcome   = effect_vars[0]

# Validate columns exist
missing = [c for c in [treatment, outcome] + confounders if c not in df.columns]
if missing:
    print(json.dumps({{"error": f"columns not in data: {{missing}}",
                       "causal_status": "correlational_only"}}))
    sys.exit(0)

# Build DOT causal graph
confounder_edges = ''.join(
    f'{{c}} -> {{treatment}}; {{c}} -> {{outcome}};'
    for c in confounders
)
graph_dot = f'digraph {{{{{{treatment}} -> {{outcome}}; {{confounder_edges}}}}}}'

try:
    model = CausalModel(
        data=df,
        treatment=treatment,
        outcome=outcome,
        graph=graph_dot,
    )
    estimand = model.identify_effect(proceed_when_unidentifiable=False)
    if estimand is None:
        print(json.dumps({{"error": "no valid estimand found",
                           "causal_status": "correlational_only"}}))
        sys.exit(0)

    estimate = model.estimate_effect(
        estimand,
        method_name="backdoor.linear_regression",
    )
    print(json.dumps({{
        "causal_status": "causal",
        "treatment": treatment,
        "outcome": outcome,
        "estimand": str(estimand),
        "effect_estimate": float(estimate.value),
        "method": "backdoor.linear_regression",
        "confounders_controlled": confounders,
    }}))
except Exception as exc:
    print(json.dumps({{"error": str(exc), "causal_status": "correlational_only"}}))
"""
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                execute_python(code, {"merged.parquet": merged_path})
            )
        finally:
            loop.close()

        if result.get("ast_rejected"):
            return {"causal_status": "correlational_only",
                    "error": f"Code rejected: {result.get('rejection_reason')}"}

        if result["exit_code"] != 0:
            return {"causal_status": "correlational_only", "error": result["stderr"]}

        try:
            import json
            return json.loads(result["stdout"])
        except Exception:
            return {"causal_status": "correlational_only", "error": "Output parse failed"}

    # ------------------------------------------------------------------
    # Tool 4: Main run — orchestrates gates and analysis
    # ------------------------------------------------------------------

    @observe(name="causal_agent.run")
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate causal analysis through the three-gate system.

        Gates (must all pass for causal analysis):
          Gate 1: causal_mode_enabled == True
          Gate 2: row_count >= MIN_ROWS_FOR_CAUSAL (100)
          Gate 3: DoWhy returns valid estimand

        Outputs always carry a ConfidenceBundle. causal_status is
        correlational_only unless all three gates pass.
        """
        job_id = self._job_id()
        redis_client = state.get("redis_client")
        task = state.get("task", "")
        merged_path = state.get("merged_path", "")
        data_summary = state.get("analysis_result", {})
        row_count = data_summary.get("row_count", 0)
        causal_mode_enabled = state.get("causal_mode_enabled", False)

        # Build initial ConfidenceBundle
        sources = [
            SourceReference(id=fid, trust_tier=TrustTier.user_uploaded, citation=fid)
            for fid in state.get("file_ids", [])
        ]
        bundle = ConfidenceBundle(
            answer_type=AnswerType.analytical,
            factual_confidence=0.9,
            analytical_confidence=0.7,
            causal_status=CausalStatus.correlational_only,
            retrieval_coverage=1.0 if merged_path else 0.0,
            contract_status=ContractStatus.skipped,
            reproducible=True,
            sources=sources,
        )
        bundle.pipeline_run_id = job_id

        numeric_cols: List[str] = data_summary.get("numeric_columns", [])

        # ------------------------------------------------------------------
        # Gate 1: causal_mode_enabled
        # ------------------------------------------------------------------
        if not causal_mode_enabled:
            self._emit_event(
                "causal_skipped",
                {
                    "reason": "causal_mode_enabled=False — running correlation analysis only.",
                    "label": "CORRELATIONAL — not causal",
                    "how_to_enable": "Set causal_mode_enabled=True in the pipeline state.",
                },
                job_id,
                redis_client,
            )
            bundle.assumptions = [
                "Causal mode disabled — correlation analysis only.",
                "Set causal_mode_enabled=True in the pipeline state to enable causal analysis.",
            ]
            correlations = self.compute_correlations(merged_path, numeric_cols)
            bundle.evaluate_human_review()
            state["causal_analysis"] = {
                "correlations": correlations,
                "label": "CORRELATIONAL — not causal",
                "confidence": bundle.to_display_summary(),
            }
            return state

        # ------------------------------------------------------------------
        # Gate 2: Dataset size
        # ------------------------------------------------------------------
        if row_count < MIN_ROWS_FOR_CAUSAL:
            bundle.mark_causal_small_dataset(row_count)
            warning = bundle.warnings[-1]
            self._emit_event(
                "causal_downgraded",
                {
                    "reason": warning,
                    "row_count": row_count,
                    "minimum_required": MIN_ROWS_FOR_CAUSAL,
                    "label": "CORRELATIONAL — not causal",
                },
                job_id,
                redis_client,
            )
            correlations = self.compute_correlations(merged_path, numeric_cols)
            bundle.evaluate_human_review()
            state["causal_analysis"] = {
                "correlations": correlations,
                "label": "CORRELATIONAL — not causal",
                "confidence": bundle.to_display_summary(),
            }
            return state

        # ------------------------------------------------------------------
        # Gate 3a: Declare assumptions BEFORE computation
        # ------------------------------------------------------------------
        assumptions_obj = self.declare_assumptions(
            data_summary, task, job_id, redis_client
        )
        bundle.assumptions = assumptions_obj.to_statement_list()

        # ------------------------------------------------------------------
        # Gate 3b: Run DoWhy
        # ------------------------------------------------------------------
        self._emit_event(
            "causal_computation_started",
            {"message": "Running DoWhy causal identification…"},
            job_id,
            redis_client,
        )

        dowhy_result = self.run_dowhy(merged_path, assumptions_obj)

        if dowhy_result.get("causal_status") == "causal":
            bundle.causal_status = CausalStatus.causal
            bundle.analytical_confidence = 0.85
            label = "CAUSAL — assumptions verified"
            self._emit_event(
                "causal_identification_succeeded",
                {
                    "estimand": dowhy_result.get("estimand", ""),
                    "effect_estimate": dowhy_result.get("effect_estimate"),
                    "label": label,
                    "note": "Review declared assumptions before acting on this result.",
                },
                job_id,
                redis_client,
            )
        else:
            error = dowhy_result.get("error", "DoWhy identification failed")
            bundle.causal_status = CausalStatus.correlational_only
            bundle.add_warning(f"CORRELATIONAL — not causal. DoWhy: {error}")
            label = "CORRELATIONAL — not causal"
            self._emit_event(
                "causal_identification_failed",
                {
                    "reason": error,
                    "downgraded_to": "correlational_only",
                    "label": label,
                },
                job_id,
                redis_client,
            )

        bundle.evaluate_human_review()
        state["causal_analysis"] = {
            "result": dowhy_result,
            "label": label,
            "confidence": bundle.to_display_summary(),
        }
        return state
