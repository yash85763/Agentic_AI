"""
Celery tasks for AgentOS backend.

Tasks
-----
run_pipeline           – Execute the expense-consolidation pipeline for a job.
monthly_report_trigger – Scheduled task fired on the 1st of each month.

Configuration is pulled from environment variables (same as main.py):
    REDIS_URL          – Redis broker & result-backend URL
    DATABASE_URL       – Sync PostgreSQL URL (psycopg2)
    AGENT_CONFIG_ROOT  – Path to agent-config/ directory
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any

from celery import Celery
from celery.schedules import crontab
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Celery application
# ---------------------------------------------------------------------------

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "agentos",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Keep results for 24 hours
    result_expires=86400,
    # Retry policy defaults
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Beat schedule
    beat_schedule={
        "monthly-report-trigger": {
            "task": "tasks.monthly_report_trigger",
            "schedule": crontab(day_of_month="1", hour="6", minute="0"),
        },
        "weekly-training-export": {
            "task": "tasks.export_training_data",
            "schedule": crontab(day_of_week="sunday", hour="2", minute="0"),
        },
    },
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_sync_db_session():
    """Return a synchronous SQLAlchemy session for use inside Celery tasks."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/agentos",
    )
    # Convert asyncpg URL to sync psycopg2 URL if needed
    sync_url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    engine = create_engine(sync_url, pool_pre_ping=True)
    Session = sessionmaker(bind=engine)
    return Session()


def _update_job_status(
    session,
    job_id: str,
    status: str,
    result: dict[str, Any] | None = None,
) -> None:
    """Update job status (and optionally result) in the DB."""
    from sqlalchemy import text

    if result is not None:
        session.execute(
            text(
                "UPDATE jobs SET status = :status, result = :result, "
                "updated_at = NOW() WHERE id = :job_id"
            ),
            {"status": status, "result": json.dumps(result), "job_id": job_id},
        )
    else:
        session.execute(
            text(
                "UPDATE jobs SET status = :status, updated_at = NOW() "
                "WHERE id = :job_id"
            ),
            {"status": status, "job_id": job_id},
        )
    session.commit()


def _publish_event(job_id: str, event_type: str, agent_name: str, data: dict[str, Any]) -> None:
    """Publish an AgentEvent to Redis pub/sub so WebSocket clients receive it."""
    import redis as redis_lib

    r = redis_lib.from_url(REDIS_URL, decode_responses=True)
    event = {
        "type": event_type,
        "job_id": job_id,
        "agent_name": agent_name,
        "data": data,
        "timestamp": datetime.utcnow().isoformat(),
    }
    channel = f"job:{job_id}:events"
    r.publish(channel, json.dumps(event))


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    name="tasks.run_pipeline",
    max_retries=2,
    default_retry_delay=10,
    soft_time_limit=1800,   # 30 min soft limit
    time_limit=2100,        # 35 min hard limit
)
def run_pipeline(
    self,
    job_id: str,
    file_ids: list[str],
    task_description: str,
) -> dict[str, Any]:
    """
    Execute the expense-consolidation pipeline for *job_id*.

    Steps
    -----
    1. Mark job as ``running``.
    2. Load cognitive context (soul, knowledge, skills, memory).
    3. Fetch file metadata from DB and download files from MinIO.
    4. Run the LangGraph expense_consolidation pipeline.
    5. Persist result and mark job ``complete`` (or ``failed``).

    Parameters
    ----------
    job_id:
        UUID string of the Job row.
    file_ids:
        List of FileRecord UUIDs attached to this job.
    task_description:
        Natural-language task sent by the user.
    """
    session = None
    try:
        session = _get_sync_db_session()
        _update_job_status(session, job_id, "running")
        _publish_event(job_id, "status", "orchestrator", {"status": "running"})

        # ------------------------------------------------------------------
        # 1. Load cognitive context
        # ------------------------------------------------------------------
        from cognitive_fs import CognitiveFSLoader

        agent_config_root = os.getenv(
            "AGENT_CONFIG_ROOT",
            os.path.join(os.path.dirname(__file__), "..", "agent-config"),
        )
        loader = CognitiveFSLoader(agent_config_root)
        context = loader.load_context(task_description, session=session)
        # Flatten demonstrations dict to a single text block for trajectory recording
        if context.get("demonstrations"):
            context["demonstrations_text"] = "\n\n---\n\n".join(
                context["demonstrations"].values()
            )
        system_prompt = loader.assemble_system_prompt(context)

        _publish_event(
            job_id, "thought", "orchestrator",
            {"message": "Cognitive context loaded", "skills": list(context["skills"].keys())},
        )

        # ------------------------------------------------------------------
        # 2. Fetch file metadata
        # ------------------------------------------------------------------
        from sqlalchemy import text as sa_text

        file_rows = []
        if file_ids:
            placeholders = ", ".join(f"'{fid}'" for fid in file_ids)
            rows = session.execute(
                sa_text(f"SELECT id, original_name, minio_path, content_type FROM file_records WHERE id IN ({placeholders})")
            ).fetchall()
            file_rows = [dict(r._mapping) for r in rows]

        _publish_event(
            job_id, "thought", "orchestrator",
            {"message": f"Found {len(file_rows)} file(s) for processing"},
        )

        # ------------------------------------------------------------------
        # 3. Run the LangGraph pipeline
        # ------------------------------------------------------------------
        try:
            from pipelines.expense_consolidation import run as run_expense_pipeline

            pipeline_result = run_expense_pipeline(
                job_id=job_id,
                task_description=task_description,
                system_prompt=system_prompt,
                cognitive_context=context,
                file_records=file_rows,
                event_publisher=lambda etype, agent, data: _publish_event(job_id, etype, agent, data),
            )
        except ImportError:
            # Pipeline module not yet implemented – return a stub result
            logger.warning("expense_consolidation pipeline not found, returning stub result")
            pipeline_result = {
                "status": "complete",
                "report": "Pipeline module not yet implemented.",
                "summary": {"files_processed": len(file_rows), "task": task_description},
            }

        # ------------------------------------------------------------------
        # 4. Compute reward and record SDAR training data
        # ------------------------------------------------------------------
        try:
            from sdar import (
                UCBSkillSelector,
                classify_task_type,
                compute_reward,
                extract_signals_from_pipeline_state,
            )
            from sdar.training_exporter import record_trajectory

            skills_used = list(context.get("skills", {}).keys())
            task_type = classify_task_type(task_description)

            signals = extract_signals_from_pipeline_state(
                job_id=job_id,
                task_type=task_type,
                skills_used=skills_used,
                pipeline_result=pipeline_result,
            )
            reward_breakdown = compute_reward(signals)
            reward_value = reward_breakdown.total

            # Record per-skill reward for UCB learning
            UCBSkillSelector().record_reward(
                job_id=job_id,
                skills_used=skills_used,
                task_type=task_type,
                reward=reward_value,
                reward_breakdown=reward_breakdown.to_dict(),
                session=session,
            )

            # Record full trajectory for GRPO training data collection
            record_trajectory(
                job_id=job_id,
                task_description=task_description,
                task_type=task_type,
                skills_used=skills_used,
                skills_context=context.get("demonstrations_text", ""),
                agent_response=str(pipeline_result.get("report", "")),
                reward=reward_value,
                reward_breakdown=reward_breakdown.to_dict(),
                validation_passed=signals.validation_passed,
                retry_count=signals.retry_count,
                user_accepted=None,
                session=session,
            )

            _publish_event(
                job_id, "thought", "orchestrator",
                {
                    "message": f"Reward computed: {reward_value:.3f}",
                    "task_type": task_type,
                    "reward_breakdown": reward_breakdown.to_dict()["components"],
                },
            )
        except Exception as exc:
            logger.warning("SDAR reward/trajectory recording failed (non-fatal): %s", exc)

        # ------------------------------------------------------------------
        # 5. Persist result
        # ------------------------------------------------------------------
        _update_job_status(session, job_id, "complete", result=pipeline_result)
        _publish_event(job_id, "result", "orchestrator", pipeline_result)

        logger.info("run_pipeline: job %s complete", job_id)
        return pipeline_result

    except Exception as exc:
        logger.exception("run_pipeline: job %s failed – %s", job_id, exc)
        if session:
            try:
                _update_job_status(
                    session, job_id, "failed",
                    result={"error": str(exc), "type": type(exc).__name__},
                )
                _publish_event(
                    job_id, "error", "orchestrator",
                    {"error": str(exc), "type": type(exc).__name__},
                )
            except Exception:
                pass
        raise self.retry(exc=exc)
    finally:
        if session:
            session.close()


# ---------------------------------------------------------------------------
# Scheduled tasks
# ---------------------------------------------------------------------------


@celery_app.task(name="tasks.export_training_data")
def export_training_data() -> dict[str, Any]:
    """Weekly task: export unexported SDAR training trajectories to JSONL.

    Runs every Sunday at 02:00 UTC. Appends to the JSONL file at
    SDAR_EXPORT_PATH (default: /tmp/sdar_trajectories.jsonl).
    Also logs training readiness so operators know when to start fine-tuning.
    """
    session = None
    try:
        session = _get_sync_db_session()
        from sdar.training_exporter import (
            export_training_data as _export,
            get_training_readiness_report,
        )

        count = _export(session)
        report = get_training_readiness_report(session)

        logger.info(
            "export_training_data: exported=%d  ready=%s  total=%d  std_reward=%.3f",
            count, report.is_ready, report.total_trajectories, report.std_reward,
        )

        if not report.is_ready:
            logger.info("Training not ready: %s", report.readiness_reasons)

        return {"exported": count, "readiness": report.to_dict()}

    except Exception as exc:
        logger.exception("export_training_data task failed: %s", exc)
        raise
    finally:
        if session:
            session.close()


@celery_app.task(name="tasks.monthly_report_trigger")
def monthly_report_trigger() -> dict[str, Any]:
    """
    Scheduled task that fires on the 1st of each month at 06:00 UTC.

    Creates a new Job for the monthly consolidation report and enqueues
    *run_pipeline* to process it.
    """
    import uuid as _uuid

    session = None
    try:
        session = _get_sync_db_session()
        now = datetime.utcnow()
        job_id = str(_uuid.uuid4())
        task_description = (
            f"Generate monthly expense consolidation report for "
            f"{now.strftime('%B %Y')}. "
            "Aggregate all expense files uploaded during the current month, "
            "normalise categories, and produce a summary report."
        )

        # Insert job row directly (sync)
        from sqlalchemy import text as sa_text

        session.execute(
            sa_text(
                "INSERT INTO jobs (id, status, task_description, file_ids, created_at, updated_at) "
                "VALUES (:id, 'pending', :task, '[]', NOW(), NOW())"
            ),
            {"id": job_id, "task": task_description},
        )
        session.commit()

        # Enqueue pipeline task
        run_pipeline.delay(
            job_id=job_id,
            file_ids=[],
            task_description=task_description,
        )

        logger.info("monthly_report_trigger: created job %s", job_id)
        return {"job_id": job_id, "triggered_at": now.isoformat()}

    except Exception as exc:
        logger.exception("monthly_report_trigger failed: %s", exc)
        raise
    finally:
        if session:
            session.close()
