"""
Abstract base class for all AgentOS pipelines.

Every concrete pipeline (e.g. ExpenseConsolidationPipeline) must:
  1. Inherit from BasePipeline.
  2. Implement the ``_run_pipeline`` coroutine.

BasePipeline provides:
  - Standardised ``run()`` entry point with error handling.
  - Job-status helpers (update_job_status, emit_progress).
  - Exponential-backoff retry logic (up to 3 attempts by default).
  - Structured PipelineResult dataclass.
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """
    Returned by BasePipeline.run() regardless of success or failure.

    Attributes
    ----------
    job_id:
        The job this result belongs to.
    success:
        True if the pipeline completed without unrecoverable errors.
    data:
        Pipeline-specific output (report URL, artefact paths, …).
    errors:
        List of error strings collected during the run.
    duration_ms:
        Total wall-clock time in milliseconds.
    retries:
        Number of retry attempts that were made before success/failure.
    """

    job_id: str
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    duration_ms: int = 0
    retries: int = 0


# ---------------------------------------------------------------------------
# BasePipeline
# ---------------------------------------------------------------------------


class BasePipeline(ABC):
    """
    Abstract pipeline base class.

    Parameters
    ----------
    max_retries:
        Maximum number of retry attempts on transient failures (default 3).
    retry_base_delay:
        Base delay in seconds for the first back-off interval (default 2 s).
        The actual delay doubles each attempt: 2 s, 4 s, 8 s, …
    sse_manager:
        Optional SSEManager instance for emitting progress events.
        When None, progress events are only logged.
    db_session:
        Optional async DB session for persisting job status.
    """

    MAX_RETRIES: int = 3
    RETRY_BASE_DELAY: float = 2.0  # seconds

    def __init__(
        self,
        max_retries: int = MAX_RETRIES,
        retry_base_delay: float = RETRY_BASE_DELAY,
        sse_manager=None,
        db_session=None,
    ):
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self._sse = sse_manager
        self._db = db_session

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        job_id: str,
        file_ids: list[str],
        task: str,
        cognitive_context: dict[str, Any],
    ) -> PipelineResult:
        """
        Execute the pipeline with retry logic.

        Parameters
        ----------
        job_id:
            Unique job identifier (UUID string).
        file_ids:
            List of file identifiers (MinIO object names / DB IDs) that
            the pipeline should process.
        task:
            Natural-language description of what the pipeline should do.
        cognitive_context:
            Additional context from previous pipeline runs / agent memory
            (team IDs, fiscal year, preferred currency, etc.).

        Returns
        -------
        PipelineResult
        """
        start_ts = time.monotonic()
        result = PipelineResult(job_id=job_id, success=False)
        attempt = 0

        await self.update_job_status(job_id, "running")

        while attempt < self.max_retries:
            attempt += 1
            try:
                logger.info(
                    "[%s] %s attempt %d/%d",
                    job_id, self.__class__.__name__, attempt, self.max_retries,
                )
                await self.emit_progress(
                    job_id,
                    pct=0.0 if attempt == 1 else 10.0 * attempt,
                    message=f"Starting pipeline (attempt {attempt}/{self.max_retries})",
                )

                pipeline_data = await self._run_pipeline(
                    job_id=job_id,
                    file_ids=file_ids,
                    task=task,
                    cognitive_context=cognitive_context,
                )

                result.success = True
                result.data = pipeline_data
                result.retries = attempt - 1
                break  # success — exit retry loop

            except _RetryableError as exc:
                logger.warning(
                    "[%s] Retryable error on attempt %d: %s",
                    job_id, attempt, exc,
                )
                result.errors.append(str(exc))

                if attempt >= self.max_retries:
                    logger.error(
                        "[%s] Max retries (%d) reached. Giving up.",
                        job_id, self.max_retries,
                    )
                    break

                delay = self.retry_base_delay * (2 ** (attempt - 1))
                logger.info("[%s] Retrying in %.1f s …", job_id, delay)
                await asyncio.sleep(delay)

            except Exception as exc:
                # Non-retryable — fail immediately
                tb = traceback.format_exc()
                logger.error(
                    "[%s] Non-retryable error: %s\n%s", job_id, exc, tb
                )
                result.errors.append(str(exc))
                break

        result.duration_ms = int((time.monotonic() - start_ts) * 1000)
        result.retries = attempt - 1

        status = "completed" if result.success else "failed"
        await self.update_job_status(
            job_id,
            status,
            meta={
                "duration_ms": result.duration_ms,
                "retries": result.retries,
                "errors": result.errors,
            },
        )

        if result.success:
            await self.emit_progress(job_id, pct=100.0, message="Pipeline complete")
        else:
            await self._emit_error(job_id, "; ".join(result.errors) or "Unknown error")

        return result

    # ------------------------------------------------------------------
    # Abstract method — subclasses implement this
    # ------------------------------------------------------------------

    @abstractmethod
    async def _run_pipeline(
        self,
        job_id: str,
        file_ids: list[str],
        task: str,
        cognitive_context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute the pipeline body.

        Must return a dict with pipeline-specific results.
        Raise ``_RetryableError`` for transient failures that should be
        retried, or any other exception for permanent failures.
        """
        ...

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    async def update_job_status(
        self,
        job_id: str,
        status: str,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """
        Persist the job status to the database.

        Falls back to logging if no DB session is configured.
        """
        logger.info("[%s] Job status → %s  %s", job_id, status, meta or "")

        if self._db is None:
            return

        try:
            # Assumes a SQLAlchemy async session and a `jobs` table model
            # imported by the concrete subclass.
            from sqlalchemy import text

            await self._db.execute(
                text(
                    "UPDATE jobs SET status = :status, updated_at = NOW(), "
                    "meta = :meta WHERE id = :job_id"
                ),
                {
                    "status": status,
                    "job_id": job_id,
                    "meta": meta,
                },
            )
            await self._db.commit()
        except Exception as exc:
            logger.error("[%s] Failed to update job status in DB: %s", job_id, exc)

    async def emit_progress(
        self,
        job_id: str,
        pct: float,
        message: str,
        agent_name: str = "pipeline",
    ) -> None:
        """
        Emit a ProgressEvent over SSE and log it.

        No-ops gracefully when SSEManager is not configured.
        """
        logger.debug("[%s] Progress %.0f%%: %s", job_id, pct, message)

        if self._sse is None:
            return

        try:
            from streaming.event_models import (
                AgentEvent,
                EventType,
                ProgressEvent,
            )

            event = AgentEvent(
                job_id=job_id,
                event_type=EventType.PROGRESS,
                agent_name=agent_name,
                data=ProgressEvent(pct=pct, message=message),
            )
            await self._sse.publish_event(job_id, event)
        except Exception as exc:
            logger.warning("[%s] Failed to emit progress event: %s", job_id, exc)

    async def _emit_error(
        self,
        job_id: str,
        message: str,
        agent_name: str = "pipeline",
        retryable: bool = False,
    ) -> None:
        """Emit an ErrorEvent over SSE."""
        if self._sse is None:
            return
        try:
            from streaming.event_models import (
                AgentEvent,
                ErrorEvent,
                EventType,
            )

            event = AgentEvent(
                job_id=job_id,
                event_type=EventType.ERROR,
                agent_name=agent_name,
                data=ErrorEvent(message=message, retryable=retryable),
            )
            await self._sse.publish_event(job_id, event)
        except Exception as exc:
            logger.warning("[%s] Failed to emit error event: %s", job_id, exc)

    async def emit_event(self, job_id: str, event) -> None:
        """
        Publish an arbitrary AgentEvent.

        Provided for use by subclass nodes that construct events directly.
        """
        if self._sse is None:
            return
        try:
            await self._sse.publish_event(job_id, event)
        except Exception as exc:
            logger.warning("[%s] Failed to emit event: %s", job_id, exc)

    # ------------------------------------------------------------------
    # Retry helpers for use by subclass nodes
    # ------------------------------------------------------------------

    @staticmethod
    def retryable(message: str) -> "_RetryableError":
        """Raise this from _run_pipeline to trigger a retry."""
        return _RetryableError(message)

    async def with_retry(
        self,
        coro_fn,
        *args,
        label: str = "operation",
        max_attempts: int = 3,
        base_delay: float = 1.0,
        **kwargs,
    ):
        """
        Run ``coro_fn(*args, **kwargs)`` with independent retry logic.

        Useful for individual steps within _run_pipeline that should
        retry without failing the whole pipeline.
        """
        for attempt in range(1, max_attempts + 1):
            try:
                return await coro_fn(*args, **kwargs)
            except Exception as exc:
                if attempt == max_attempts:
                    logger.error(
                        "%s failed after %d attempts: %s", label, max_attempts, exc
                    )
                    raise
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "%s attempt %d/%d failed: %s. Retrying in %.1f s",
                    label, attempt, max_attempts, exc, delay,
                )
                await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# Internal sentinel exception
# ---------------------------------------------------------------------------


class _RetryableError(Exception):
    """Raised by pipeline nodes to signal a transient failure."""
