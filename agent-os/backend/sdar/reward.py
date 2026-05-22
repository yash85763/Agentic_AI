"""
Reward Computation Layer for SDAR (Self-Distilled Agentic Reinforcement Learning).

Maps a completed pipeline run's outcome signals to a single scalar reward R ∈ [0, 1].
This reward is the training signal for GRPO and the feedback signal for UCB selection.

Reward decomposition (weights sum to 1.0):
  validation_passed:           0.40 — did output totals validate against source files?
  columns_matched_fraction:    0.20 — partial credit for partial column matches
  anomaly_detection_quality:   0.10 — quality of anomaly detection vs. business rules
  first_pass_success:          0.15 — clean run without transform retries
  user_accepted:               0.15 — delayed signal from user feedback

The user_accepted signal starts at 0.5 (neutral/pending) and is updated when
the user either downloads the report without correction or submits a correction.

Why these weights:
  Validation is the ground truth we can measure mechanically. Column matching
  gives partial credit for partially correct runs (drives GRPO gradient more
  than binary). User acceptance captures quality signals the validator misses.
  First-pass success penalises brittleness — retries mean the LLM's first code
  attempt failed, which is an outcome we want to discourage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reward weights
# ---------------------------------------------------------------------------

_W_VALIDATION  = 0.40
_W_COL_MATCH   = 0.20
_W_ANOMALY     = 0.10
_W_FIRST_PASS  = 0.15
_W_USER_ACCEPT = 0.15

assert abs(sum([_W_VALIDATION, _W_COL_MATCH, _W_ANOMALY, _W_FIRST_PASS, _W_USER_ACCEPT]) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Signals and result types
# ---------------------------------------------------------------------------


@dataclass
class RewardSignals:
    """All observable signals collected after a pipeline run completes.

    Immediate signals are available as soon as the pipeline finishes.
    Delayed signals (user_accepted) are updated asynchronously when the user
    provides feedback via the report UI.
    """

    # --- Immediate (available at pipeline completion) ---

    validation_passed: bool = False
    # Fraction of amount-column totals that matched source-file totals [0, 1]
    columns_matched_fraction: float = 0.0
    # Number of anomalies the validation agent flagged
    anomalies_flagged: int = 0
    # Ground-truth anomaly count from business rules (0 = unknown)
    anomalies_expected: int = 0
    # Number of transform retries (0 = first-pass success)
    retry_count: int = 0

    # --- Delayed (updated by user feedback endpoint) ---

    # None = pending, True = user accepted, False = user corrected output
    user_accepted: Optional[bool] = None

    # --- Context (for UCB record-keeping) ---

    skills_used: list[str] = field(default_factory=list)
    task_type: str = "general"
    job_id: str = ""


@dataclass
class RewardBreakdown:
    """Scalar reward with per-component decomposition for explainability and debugging."""

    total: float
    validation_component: float
    column_match_component: float
    anomaly_component: float
    first_pass_component: float
    user_accept_component: float
    signals: RewardSignals

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": round(self.total, 4),
            "components": {
                "validation":    round(self.validation_component, 4),
                "column_match":  round(self.column_match_component, 4),
                "anomaly":       round(self.anomaly_component, 4),
                "first_pass":    round(self.first_pass_component, 4),
                "user_accept":   round(self.user_accept_component, 4),
            },
            "signals": {
                "validation_passed":          self.signals.validation_passed,
                "columns_matched_fraction":   round(self.signals.columns_matched_fraction, 4),
                "anomalies_flagged":          self.signals.anomalies_flagged,
                "anomalies_expected":         self.signals.anomalies_expected,
                "retry_count":                self.signals.retry_count,
                "user_accepted":              self.signals.user_accepted,
                "skills_used":                self.signals.skills_used,
                "task_type":                  self.signals.task_type,
                "job_id":                     self.signals.job_id,
            },
        }


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_reward(signals: RewardSignals) -> RewardBreakdown:
    """Compute scalar reward R ∈ [0, 1] from pipeline outcome signals.

    user_accepted=None is treated as 0.5 (neutral/pending). This means the
    immediate reward is slightly conservative — it improves when positive
    feedback arrives and decreases on corrections.

    The reward is strictly bounded to [0, 1] even if individual components
    would push beyond due to edge-case inputs.
    """
    # Component 1: Validation (binary — pass or fail)
    v = _W_VALIDATION * (1.0 if signals.validation_passed else 0.0)

    # Component 2: Column match (continuous partial credit)
    col = _W_COL_MATCH * max(0.0, min(1.0, signals.columns_matched_fraction))

    # Component 3: Anomaly detection quality
    if signals.anomalies_expected > 0:
        # Ground truth known: precision of anomaly detection
        anom_score = min(signals.anomalies_flagged, signals.anomalies_expected) / signals.anomalies_expected
    elif signals.anomalies_flagged > 0:
        # No ground truth: positive signal for finding anomalies (capped at 3)
        anom_score = min(signals.anomalies_flagged / 3.0, 1.0)
    else:
        # No anomalies expected, none found → neutral 0.5
        anom_score = 0.5
    anom = _W_ANOMALY * anom_score

    # Component 4: First-pass success (penalises retries)
    if signals.retry_count == 0:
        fp_score = 1.0
    elif signals.retry_count == 1:
        fp_score = 0.35
    else:
        fp_score = 0.0
    fp = _W_FIRST_PASS * fp_score

    # Component 5: User acceptance (delayed signal)
    if signals.user_accepted is None:
        ua_score = 0.5   # pending — neutral
    elif signals.user_accepted:
        ua_score = 1.0
    else:
        ua_score = 0.0   # user corrected the output
    ua = _W_USER_ACCEPT * ua_score

    total = max(0.0, min(1.0, v + col + anom + fp + ua))

    logger.debug(
        "Reward job=%s total=%.3f  val=%.3f col=%.3f anom=%.3f fp=%.3f ua=%.3f",
        signals.job_id, total, v, col, anom, fp, ua,
    )

    return RewardBreakdown(
        total=total,
        validation_component=v,
        column_match_component=col,
        anomaly_component=anom,
        first_pass_component=fp,
        user_accept_component=ua,
        signals=signals,
    )


def update_reward_for_user_feedback(
    breakdown: RewardBreakdown,
    user_accepted: bool,
) -> RewardBreakdown:
    """Recompute reward with the delayed user_accepted signal.

    Call this when the user provides explicit feedback (accepts or corrects
    the output). The updated reward replaces the pending neutral value.
    """
    updated_signals = RewardSignals(
        validation_passed=breakdown.signals.validation_passed,
        columns_matched_fraction=breakdown.signals.columns_matched_fraction,
        anomalies_flagged=breakdown.signals.anomalies_flagged,
        anomalies_expected=breakdown.signals.anomalies_expected,
        retry_count=breakdown.signals.retry_count,
        user_accepted=user_accepted,
        skills_used=breakdown.signals.skills_used,
        task_type=breakdown.signals.task_type,
        job_id=breakdown.signals.job_id,
    )
    return compute_reward(updated_signals)


# ---------------------------------------------------------------------------
# Signal extraction from pipeline result
# ---------------------------------------------------------------------------


def extract_signals_from_pipeline_state(
    job_id: str,
    task_type: str,
    skills_used: list[str],
    pipeline_result: dict[str, Any],
) -> RewardSignals:
    """Build RewardSignals from a completed pipeline result dict.

    Called by tasks.py immediately after pipeline completion.
    Designed to be robust to partial/missing result structures.
    """
    validation = pipeline_result.get("validation", {}) or {}

    passed = bool(validation.get("passed", False))

    # Extract column match fraction from validation check results
    checks = validation.get("checks", []) or validation.get("items", []) or []
    col_checks = [
        c for c in checks
        if isinstance(c, dict) and c.get("check") in (
            "sum_conservation", "total_match", "column_sum"
        )
    ]
    if col_checks:
        n_passed = sum(
            1 for c in col_checks
            if c.get("status") in ("pass", "passed", "ok")
        )
        col_frac = n_passed / len(col_checks)
    else:
        # No explicit checks recorded — use binary validation result
        col_frac = 1.0 if passed else 0.3   # partial credit even on overall failure

    # Anomaly count
    anomalies = validation.get("anomalies", []) or []
    flagged = len(anomalies) if isinstance(anomalies, list) else 0

    # Retry count
    retry_count = int(pipeline_result.get("retry_count", 0))

    return RewardSignals(
        validation_passed=passed,
        columns_matched_fraction=col_frac,
        anomalies_flagged=flagged,
        anomalies_expected=0,
        retry_count=retry_count,
        user_accepted=None,
        skills_used=skills_used,
        task_type=task_type,
        job_id=job_id,
    )
