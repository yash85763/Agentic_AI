"""
ConfidenceBundle — standardised trust and provenance envelope for all pipeline outputs.

Every pipeline output (chart, table, report section, causal analysis) must carry
a ConfidenceBundle. It answers: how was this produced, from what sources, with
what confidence, and what should the consumer do with it.

Downstream consumers use requires_human_review and contract_status to decide
whether to act on an output or route it for human inspection first.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class AnswerType(str, Enum):
    analytical = "analytical"
    causal = "causal"
    factual = "factual"
    descriptive = "descriptive"


class CausalStatus(str, Enum):
    causal = "causal"
    correlational_only = "correlational_only"
    not_applicable = "not_applicable"


class ContractStatus(str, Enum):
    passed = "pass"
    failed = "fail"
    skipped = "skipped"


class TrustTier(str, Enum):
    system_generated = "system-generated"      # cognitive FS, agent-generated rules
    human_verified = "human-verified"           # human-reviewed and approved
    user_uploaded = "user-uploaded"             # uploaded by user, not reviewed
    externally_fetched = "externally-fetched"   # from external APIs / web


# ---------------------------------------------------------------------------
# Source reference
# ---------------------------------------------------------------------------


class SourceReference(BaseModel):
    id: str = Field(..., description="Unique identifier for this source")
    trust_tier: TrustTier
    citation: str = Field(..., description="Human-readable source description")
    file_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Thresholds for automatic human-review triggers
# ---------------------------------------------------------------------------

_LOW_COVERAGE_THRESHOLD = 0.70
_CAUSAL_MIN_ROWS = 100
_CONFLICT_KEYWORDS = ("conflict", "mismatch", "discrepancy", "inconsistent")


# ---------------------------------------------------------------------------
# ConfidenceBundle
# ---------------------------------------------------------------------------


class ConfidenceBundle(BaseModel):
    """
    Standardised trust and provenance envelope.

    Attach one of these to every final output. The frontend displays it as a
    visual trust indicator showing: source count, coverage, contract status,
    and human-review recommendation.
    """

    # What kind of claim is this?
    answer_type: AnswerType

    # Confidence scores [0, 1]
    factual_confidence: float = Field(..., ge=0.0, le=1.0)
    analytical_confidence: float = Field(..., ge=0.0, le=1.0)

    # Causal status — MUST default to correlational_only
    # Only set to 'causal' when DoWhy identification returns a valid estimand
    causal_status: CausalStatus = CausalStatus.not_applicable

    # What fraction of expected source data was found and used [0, 1]
    retrieval_coverage: float = Field(..., ge=0.0, le=1.0)

    # Did output totals validate against source file totals?
    contract_status: ContractStatus = ContractStatus.skipped

    # Can this exact output be reproduced from the same inputs?
    reproducible: bool = True

    # All data sources that contributed to this output
    sources: List[SourceReference] = Field(default_factory=list)

    # Explicit assumptions declared before analysis
    assumptions: List[str] = Field(default_factory=list)

    # Conflicts, missing data, or unusual conditions
    warnings: List[str] = Field(default_factory=list)

    # Human review recommendation
    requires_human_review: bool = False
    human_review_reason: Optional[str] = None

    # Traceability
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    pipeline_run_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )

    # ---------------------------------------------------------------------------
    # Auto-evaluation
    # ---------------------------------------------------------------------------

    def evaluate_human_review(self, anomaly_threshold_exceeded: bool = False) -> None:
        """Auto-set requires_human_review per the rules in Section 3.2.

        Call this after all fields are populated. It is idempotent.
        """
        reasons: list[str] = []

        # Rule 1: contract validation failed
        if self.contract_status == ContractStatus.failed:
            reasons.append(
                "Contract validation failed: output totals do not match source file totals."
            )

        # Rule 2: any untrusted source
        untrusted = [
            s for s in self.sources
            if s.trust_tier in (TrustTier.user_uploaded, TrustTier.externally_fetched)
        ]
        if untrusted:
            tiers = ", ".join(s.trust_tier.value for s in untrusted[:3])
            reasons.append(
                f"Output includes untrusted sources ({tiers})."
            )

        # Rule 3: low retrieval coverage
        if self.retrieval_coverage < _LOW_COVERAGE_THRESHOLD:
            pct = self.retrieval_coverage * 100
            reasons.append(
                f"Low retrieval coverage ({pct:.0f}%) — "
                "more than 30% of expected data was not found."
            )

        # Rule 4: causal claim with small dataset (row count checked in CausalAgent)
        # This flag is set externally via add_warning() before evaluate_human_review()

        # Rule 5: conflicting warnings
        if any(kw in w.lower() for w in self.warnings for kw in _CONFLICT_KEYWORDS):
            reasons.append("Conflicts detected between data sources.")

        # Rule 6: anomaly threshold exceeded
        if anomaly_threshold_exceeded:
            reasons.append(
                "One or more values exceed the anomaly threshold defined in business-rules.md."
            )

        if reasons:
            self.requires_human_review = True
            self.human_review_reason = " | ".join(reasons)
        else:
            self.requires_human_review = False
            self.human_review_reason = None

    def add_warning(self, warning: str) -> None:
        """Append a warning string (convenience method)."""
        self.warnings.append(warning)

    def mark_causal_small_dataset(self, row_count: int) -> None:
        """Mark that causal analysis was requested but dataset is too small."""
        self.causal_status = CausalStatus.correlational_only
        self.add_warning(
            f"Dataset has {row_count} rows (minimum {_CAUSAL_MIN_ROWS} required "
            "for causal claims) — downgraded to correlational analysis."
        )

    # Frontend-facing summary
    def to_display_summary(self) -> dict:
        return {
            "answer_type": self.answer_type,
            "factual_confidence": round(self.factual_confidence, 2),
            "analytical_confidence": round(self.analytical_confidence, 2),
            "causal_status": self.causal_status,
            "retrieval_coverage": round(self.retrieval_coverage, 2),
            "contract_status": self.contract_status,
            "source_count": len(self.sources),
            "warning_count": len(self.warnings),
            "requires_human_review": self.requires_human_review,
            "human_review_reason": self.human_review_reason,
            "reproducible": self.reproducible,
            "pipeline_run_id": self.pipeline_run_id,
        }

    model_config = {"use_enum_values": False}
