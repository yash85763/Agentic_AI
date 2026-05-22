"""
Training Data Exporter for SDAR.

Manages the collection and export of training trajectories for future
GRPO fine-tuning of the student model.

Data collection strategy:
  - One TrajectoryRecord per completed pipeline run
  - Record captures: task description, skills used, model response, reward
  - JSONL export format compatible with GRPO training scripts
  - Readiness threshold: 500 exported trajectories with reward std > 0.15

The teacher/student training loop (SDAR Phase 3) is deferred until GPU
infra is available. This module handles only data collection and export.

Storage:
  - training_trajectories PostgreSQL table (one row per trajectory)
  - Exported JSONL at EXPORT_PATH (default: /tmp/sdar_trajectories.jsonl)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

EXPORT_PATH = os.getenv("SDAR_EXPORT_PATH", "/tmp/sdar_trajectories.jsonl")

# Readiness thresholds
MIN_TRAJECTORIES = 500
MIN_REWARD_STD = 0.15
MIN_TASK_TYPES = 3   # need diversity across at least 3 task types


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryRecord:
    """A single training example for SDAR GRPO fine-tuning.

    In SDAR terminology:
        task_description   → prompt given to both teacher and student
        skills_context     → privileged skill demonstrations (teacher's c+)
        agent_response     → model completion to score
        reward             → scalar R ∈ [0,1] from compute_reward()
        reward_breakdown   → per-component dict for debugging

    The student model learns to match teacher quality without c+ at inference
    time. During training, c+ is provided to teacher and withheld from student.
    """
    job_id: str
    task_description: str
    task_type: str
    skills_used: list[str]
    skills_context: str           # assembled demonstrations (teacher's c+)
    agent_response: str           # model output from pipeline run
    reward: float
    reward_breakdown: dict[str, Any]
    validation_passed: bool
    retry_count: int
    user_accepted: Optional[bool] = None
    created_at: str = ""          # ISO timestamp, filled at insert time
    exported: bool = False

    def to_grpo_example(self) -> dict[str, Any]:
        """Format as a GRPO training example dict."""
        return {
            "prompt": self.task_description,
            "privileged_context": self.skills_context,
            "response": self.agent_response,
            "reward": self.reward,
            "metadata": {
                "job_id": self.job_id,
                "task_type": self.task_type,
                "skills_used": self.skills_used,
                "reward_breakdown": self.reward_breakdown,
                "validation_passed": self.validation_passed,
                "retry_count": self.retry_count,
                "user_accepted": self.user_accepted,
            },
        }


@dataclass
class TrainingReadinessReport:
    """Summary of training data collection status."""
    total_trajectories: int
    exported_trajectories: int
    mean_reward: float
    std_reward: float
    task_type_counts: dict[str, int]
    is_ready: bool
    readiness_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_trajectories": self.total_trajectories,
            "exported_trajectories": self.exported_trajectories,
            "mean_reward": round(self.mean_reward, 4),
            "std_reward": round(self.std_reward, 4),
            "task_type_counts": self.task_type_counts,
            "is_ready": self.is_ready,
            "readiness_reasons": self.readiness_reasons,
            "thresholds": {
                "min_trajectories": MIN_TRAJECTORIES,
                "min_reward_std": MIN_REWARD_STD,
                "min_task_types": MIN_TASK_TYPES,
            },
        }


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def record_trajectory(
    job_id: str,
    task_description: str,
    task_type: str,
    skills_used: list[str],
    skills_context: str,
    agent_response: str,
    reward: float,
    reward_breakdown: dict[str, Any],
    validation_passed: bool,
    retry_count: int,
    user_accepted: Optional[bool],
    session: Any,
) -> None:
    """Insert a trajectory record into the training_trajectories table.

    Silently logs and returns on DB errors to avoid breaking the pipeline.
    Called after every completed pipeline run.
    """
    from sqlalchemy import text

    try:
        session.execute(
            text("""
                INSERT INTO training_trajectories (
                    id, job_id, task_description, task_type,
                    skills_used, skills_context, agent_response,
                    reward, reward_breakdown,
                    validation_passed, retry_count, user_accepted,
                    exported, created_at
                ) VALUES (
                    gen_random_uuid(),
                    :job_id::uuid,
                    :task_description,
                    :task_type,
                    :skills_used::jsonb,
                    :skills_context,
                    :agent_response,
                    :reward,
                    :reward_breakdown::jsonb,
                    :validation_passed,
                    :retry_count,
                    :user_accepted,
                    false,
                    NOW()
                )
            """),
            {
                "job_id": job_id,
                "task_description": task_description,
                "task_type": task_type,
                "skills_used": json.dumps(skills_used),
                "skills_context": skills_context,
                "agent_response": agent_response,
                "reward": float(reward),
                "reward_breakdown": json.dumps(reward_breakdown),
                "validation_passed": bool(validation_passed),
                "retry_count": int(retry_count),
                "user_accepted": user_accepted,
            },
        )
        session.commit()
        logger.info("Recorded trajectory for job=%s task_type=%s reward=%.3f", job_id, task_type, reward)
    except Exception as exc:
        logger.error("Failed to record trajectory for job=%s: %s", job_id, exc)
        try:
            session.rollback()
        except Exception:
            pass


def export_training_data(
    session: Any,
    output_path: str = EXPORT_PATH,
    batch_size: int = 1000,
) -> int:
    """Export unexported trajectories to a JSONL file for GRPO training.

    Marks exported rows as exported=True after writing. Returns count exported.
    Appends to existing file so incremental exports are safe.

    Args:
        session: Sync SQLAlchemy session.
        output_path: Path to write JSONL output.
        batch_size: Rows to fetch per DB batch.

    Returns:
        Number of trajectories exported.
    """
    from sqlalchemy import text

    total_exported = 0
    offset = 0

    try:
        with open(output_path, "a", encoding="utf-8") as f:
            while True:
                rows = session.execute(
                    text("""
                        SELECT id, job_id, task_description, task_type,
                               skills_used, skills_context, agent_response,
                               reward, reward_breakdown,
                               validation_passed, retry_count, user_accepted,
                               created_at
                        FROM training_trajectories
                        WHERE exported = false
                        ORDER BY created_at ASC
                        LIMIT :limit OFFSET :offset
                    """),
                    {"limit": batch_size, "offset": offset},
                ).fetchall()

                if not rows:
                    break

                row_ids = []
                for row in rows:
                    record = TrajectoryRecord(
                        job_id=str(row.job_id),
                        task_description=row.task_description,
                        task_type=row.task_type,
                        skills_used=row.skills_used if isinstance(row.skills_used, list) else json.loads(row.skills_used or "[]"),
                        skills_context=row.skills_context or "",
                        agent_response=row.agent_response or "",
                        reward=float(row.reward),
                        reward_breakdown=row.reward_breakdown if isinstance(row.reward_breakdown, dict) else json.loads(row.reward_breakdown or "{}"),
                        validation_passed=bool(row.validation_passed),
                        retry_count=int(row.retry_count),
                        user_accepted=row.user_accepted,
                        created_at=str(row.created_at),
                        exported=False,
                    )
                    f.write(json.dumps(record.to_grpo_example()) + "\n")
                    row_ids.append(str(row.id))

                # Mark batch as exported
                if row_ids:
                    placeholders = ", ".join(f"'{rid}'" for rid in row_ids)
                    session.execute(
                        text(f"UPDATE training_trajectories SET exported = true WHERE id IN ({placeholders})")
                    )
                    session.commit()

                total_exported += len(rows)
                offset += batch_size

                if len(rows) < batch_size:
                    break

    except Exception as exc:
        logger.error("Training data export failed: %s", exc)
        try:
            session.rollback()
        except Exception:
            pass
        raise

    logger.info("Exported %d trajectories to %s", total_exported, output_path)
    return total_exported


def get_training_readiness_report(session: Any) -> TrainingReadinessReport:
    """Check if enough high-quality trajectories have been collected.

    Readiness requires ALL of:
    - 500+ total trajectories
    - Reward std > 0.15 (enough diversity to drive GRPO gradient)
    - At least 3 distinct task types represented

    Returns a TrainingReadinessReport with current status and blocking reasons.
    """
    from sqlalchemy import text

    try:
        agg_row = session.execute(
            text("""
                SELECT
                    COUNT(*)           AS total,
                    COUNT(*) FILTER (WHERE exported = true) AS exported,
                    COALESCE(AVG(reward), 0)   AS mean_reward,
                    COALESCE(STDDEV(reward), 0) AS std_reward
                FROM training_trajectories
            """),
        ).fetchone()

        task_rows = session.execute(
            text("""
                SELECT task_type, COUNT(*) AS cnt
                FROM training_trajectories
                GROUP BY task_type
                ORDER BY cnt DESC
            """),
        ).fetchall()

    except Exception as exc:
        logger.error("Failed to compute training readiness: %s", exc)
        return TrainingReadinessReport(
            total_trajectories=0,
            exported_trajectories=0,
            mean_reward=0.0,
            std_reward=0.0,
            task_type_counts={},
            is_ready=False,
            readiness_reasons=["DB query failed — cannot assess readiness"],
        )

    total = int(agg_row.total) if agg_row else 0
    exported = int(agg_row.exported) if agg_row else 0
    mean_reward = float(agg_row.mean_reward) if agg_row else 0.0
    std_reward = float(agg_row.std_reward) if agg_row else 0.0
    task_type_counts = {row.task_type: int(row.cnt) for row in (task_rows or [])}
    n_task_types = len(task_type_counts)

    reasons: list[str] = []
    if total < MIN_TRAJECTORIES:
        reasons.append(f"Need {MIN_TRAJECTORIES} trajectories, have {total}")
    if std_reward < MIN_REWARD_STD:
        reasons.append(
            f"Reward std {std_reward:.3f} < threshold {MIN_REWARD_STD} — "
            "not enough outcome diversity for GRPO gradient"
        )
    if n_task_types < MIN_TASK_TYPES:
        reasons.append(
            f"Only {n_task_types} task type(s) — need {MIN_TASK_TYPES} for diverse training"
        )

    is_ready = len(reasons) == 0

    return TrainingReadinessReport(
        total_trajectories=total,
        exported_trajectories=exported,
        mean_reward=mean_reward,
        std_reward=std_reward,
        task_type_counts=task_type_counts,
        is_ready=is_ready,
        readiness_reasons=reasons,
    )
