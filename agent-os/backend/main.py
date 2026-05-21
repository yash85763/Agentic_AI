"""
AgentOS FastAPI backend – entry point.

Routers
-------
/api/health      – liveness / readiness checks
/api/jobs        – CRUD + SSE stream for pipeline jobs
/api/files       – upload / list / delete files (MinIO-backed)
/api/cognitive   – read / write agent-config files

WebSocket
---------
/ws/{job_id}     – real-time event stream for a specific job

Background
----------
Job creation triggers run_pipeline Celery task automatically.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import aiofiles
import redis.asyncio as aioredis
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/agentos"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "agentos-files"
    minio_secure: bool = False

    # Langfuse (optional – disabled when keys are blank)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # Agent-config root (volume-mounted in Docker)
    agent_config_root: str = str(
        Path(__file__).resolve().parent.parent / "agent-config"
    )

    # App
    debug: bool = False
    max_upload_bytes: int = 100 * 1024 * 1024  # 100 MB


settings = Settings()


# ---------------------------------------------------------------------------
# Global clients (initialised in lifespan)
# ---------------------------------------------------------------------------

_redis_client: Optional[aioredis.Redis] = None
_minio_client: Optional[Any] = None  # minio.Minio


def get_redis() -> aioredis.Redis:
    if _redis_client is None:
        raise RuntimeError("Redis not initialised")
    return _redis_client


def get_minio():
    if _minio_client is None:
        raise RuntimeError("MinIO not initialised")
    return _minio_client


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _redis_client, _minio_client

    # -- Database -----------------------------------------------------------
    from database import init_db
    await init_db(settings.database_url)
    logger.info("Database initialised")

    # -- Redis --------------------------------------------------------------
    _redis_client = aioredis.from_url(
        settings.redis_url, encoding="utf-8", decode_responses=True
    )
    await _redis_client.ping()
    logger.info("Redis connected")

    # -- MinIO --------------------------------------------------------------
    try:
        from minio import Minio
        _minio_client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        if not _minio_client.bucket_exists(settings.minio_bucket):
            _minio_client.make_bucket(settings.minio_bucket)
        logger.info("MinIO connected, bucket: %s", settings.minio_bucket)
    except Exception as exc:
        logger.warning("MinIO not available: %s – file uploads will fail", exc)
        _minio_client = None

    # -- Langfuse -----------------------------------------------------------
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        try:
            from langfuse import Langfuse
            lf = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            app.state.langfuse = lf
            logger.info("Langfuse initialised")
        except Exception as exc:
            logger.warning("Langfuse init failed: %s", exc)
    else:
        logger.info("Langfuse not configured – tracing disabled")

    yield

    # -- Shutdown -----------------------------------------------------------
    if _redis_client:
        await _redis_client.aclose()
    logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AgentOS Backend",
    version="0.1.0",
    description="Backend API for the AgentOS expense-consolidation platform",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# DB dependency
# ---------------------------------------------------------------------------

from database import get_session  # noqa: E402 – after init


# ===========================================================================
# ROUTER: /api/health
# ===========================================================================

from fastapi import APIRouter

health_router = APIRouter(prefix="/api/health", tags=["health"])


@health_router.get("")
async def health_check():
    """Liveness probe."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@health_router.get("/ready")
async def readiness_check():
    """Readiness probe – verifies DB and Redis."""
    checks: dict[str, str] = {}

    # Redis
    try:
        r = get_redis()
        await r.ping()
        checks["redis"] = "ok"
    except Exception as exc:
        checks["redis"] = f"error: {exc}"

    # MinIO
    try:
        mc = get_minio()
        mc.bucket_exists(settings.minio_bucket)
        checks["minio"] = "ok"
    except Exception as exc:
        checks["minio"] = f"error: {exc}"

    all_ok = all(v == "ok" for v in checks.values())
    return {"status": "ready" if all_ok else "degraded", "checks": checks}


# ===========================================================================
# ROUTER: /api/jobs
# ===========================================================================

jobs_router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class JobCreateRequest(BaseModel):
    task_description: str
    file_ids: list[str] = []


@jobs_router.post("", status_code=201)
async def create_job(
    body: JobCreateRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    """Create a new pipeline job and enqueue it for processing."""
    from database import Job as JobModel

    job = JobModel(
        id=str(uuid.uuid4()),
        status="pending",
        task_description=body.task_description,
        file_ids=body.file_ids,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    # Enqueue Celery task (fire-and-forget)
    background_tasks.add_task(
        _enqueue_pipeline,
        job_id=job.id,
        file_ids=body.file_ids,
        task_description=body.task_description,
    )

    return _job_to_dict(job)


def _enqueue_pipeline(job_id: str, file_ids: list[str], task_description: str) -> None:
    """Synchronous helper called from BackgroundTasks to send task to Celery."""
    try:
        from tasks import run_pipeline
        run_pipeline.delay(
            job_id=job_id,
            file_ids=file_ids,
            task_description=task_description,
        )
        logger.info("Enqueued pipeline for job %s", job_id)
    except Exception as exc:
        logger.error("Failed to enqueue pipeline for job %s: %s", job_id, exc)


@jobs_router.get("")
async def list_jobs(
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
):
    """List all jobs, optionally filtered by status."""
    from database import Job as JobModel

    stmt = select(JobModel).order_by(JobModel.created_at.desc()).limit(limit).offset(offset)
    if status:
        stmt = stmt.where(JobModel.status == status)
    result = await session.execute(stmt)
    jobs = result.scalars().all()
    return [_job_to_dict(j) for j in jobs]


@jobs_router.get("/{job_id}")
async def get_job(job_id: str, session: AsyncSession = Depends(get_session)):
    """Get a single job by ID."""
    from database import Job as JobModel

    job = await session.get(JobModel, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_to_dict(job)


@jobs_router.delete("/{job_id}", status_code=204)
async def delete_job(job_id: str, session: AsyncSession = Depends(get_session)):
    """Delete a job record."""
    from database import Job as JobModel

    job = await session.get(JobModel, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    await session.delete(job)
    await session.commit()


@jobs_router.get("/{job_id}/stream")
async def stream_job_events(job_id: str):
    """
    Server-Sent Events (SSE) stream for a job.

    Subscribes to Redis pub/sub channel ``job:{job_id}:events`` and
    forwards events to the browser as ``text/event-stream``.
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        r = aioredis.from_url(settings.redis_url, encoding="utf-8", decode_responses=True)
        pubsub = r.pubsub()
        channel = f"job:{job_id}:events"
        await pubsub.subscribe(channel)
        try:
            # Send a heartbeat immediately so the browser knows it's connected
            yield f"data: {json.dumps({'type': 'connected', 'job_id': job_id})}\n\n"
            async for message in pubsub.listen():
                if message["type"] == "message":
                    yield f"data: {message['data']}\n\n"
                    data = json.loads(message["data"])
                    if data.get("type") in ("result", "error"):
                        break
        except asyncio.CancelledError:
            pass
        finally:
            await pubsub.unsubscribe(channel)
            await r.aclose()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _job_to_dict(job) -> dict:
    return {
        "id": job.id,
        "status": job.status,
        "task_description": job.task_description,
        "file_ids": job.file_ids or [],
        "result": job.result,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
    }


# ===========================================================================
# ROUTER: /api/files
# ===========================================================================

files_router = APIRouter(prefix="/api/files", tags=["files"])


@files_router.post("/upload", status_code=201)
async def upload_file(
    file: UploadFile = File(...),
    job_id: Optional[str] = Query(None, description="Associate file with a job"),
    session: AsyncSession = Depends(get_session),
):
    """
    Upload a file to MinIO and record it in the database.

    Accepts any file up to ``settings.max_upload_bytes``.
    """
    from database import FileRecord

    content = await file.read()
    file_size = len(content)
    if file_size > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {file_size} bytes (max {settings.max_upload_bytes})",
        )

    file_id = str(uuid.uuid4())
    original_name = file.filename or "unknown"
    content_type = file.content_type or "application/octet-stream"
    minio_path = f"uploads/{file_id}/{original_name}"

    # Upload to MinIO
    mc = get_minio()
    import io
    mc.put_object(
        settings.minio_bucket,
        minio_path,
        io.BytesIO(content),
        length=file_size,
        content_type=content_type,
    )

    # Persist metadata
    record = FileRecord(
        id=file_id,
        original_name=original_name,
        minio_path=minio_path,
        size=file_size,
        content_type=content_type,
        job_id=job_id,
    )
    session.add(record)
    await session.commit()
    await session.refresh(record)

    return {
        "id": record.id,
        "original_name": record.original_name,
        "minio_path": record.minio_path,
        "size": record.size,
        "content_type": record.content_type,
        "job_id": record.job_id,
        "created_at": record.created_at.isoformat() if record.created_at else None,
    }


@files_router.get("")
async def list_files(
    job_id: Optional[str] = Query(None),
    session: AsyncSession = Depends(get_session),
):
    """List all uploaded files, optionally filtered by job_id."""
    from database import FileRecord
    from sqlalchemy import select

    stmt = select(FileRecord).order_by(FileRecord.created_at.desc())
    if job_id:
        stmt = stmt.where(FileRecord.job_id == job_id)
    result = await session.execute(stmt)
    records = result.scalars().all()
    return [
        {
            "id": r.id,
            "original_name": r.original_name,
            "minio_path": r.minio_path,
            "size": r.size,
            "content_type": r.content_type,
            "job_id": r.job_id,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in records
    ]


@files_router.delete("/{file_id}", status_code=204)
async def delete_file(file_id: str, session: AsyncSession = Depends(get_session)):
    """Delete a file from MinIO and remove its DB record."""
    from database import FileRecord

    record = await session.get(FileRecord, file_id)
    if not record:
        raise HTTPException(status_code=404, detail="File not found")

    # Remove from MinIO (best-effort)
    try:
        mc = get_minio()
        mc.remove_object(settings.minio_bucket, record.minio_path)
    except Exception as exc:
        logger.warning("Could not delete MinIO object %s: %s", record.minio_path, exc)

    await session.delete(record)
    await session.commit()


# ===========================================================================
# ROUTER: /api/cognitive
# ===========================================================================

cognitive_router = APIRouter(prefix="/api/cognitive", tags=["cognitive-fs"])


class PutFileRequest(BaseModel):
    content: str


class UpdateMemoryRequest(BaseModel):
    key: str
    value: dict[str, Any]


def _get_loader():
    from cognitive_fs import CognitiveFSLoader
    return CognitiveFSLoader(settings.agent_config_root)


@cognitive_router.get("/files")
async def list_cognitive_files():
    """Return the full agent-config directory tree."""
    loader = _get_loader()
    return loader.list_files()


@cognitive_router.get("/files/{path:path}")
async def get_cognitive_file(path: str):
    """Read a file from agent-config by its relative path."""
    loader = _get_loader()
    try:
        content = loader.get_file(path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"path": path, "content": content}


@cognitive_router.put("/files/{path:path}")
async def put_cognitive_file(path: str, body: PutFileRequest):
    """Write (create or overwrite) a file in agent-config."""
    loader = _get_loader()
    try:
        loader.put_file(path, body.content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"path": path, "size": len(body.content), "status": "written"}


@cognitive_router.post("/memory")
async def update_memory(body: UpdateMemoryRequest):
    """Merge data into memory/schema-cache.json or memory/column-mappings.json."""
    loader = _get_loader()
    try:
        loader.update_memory(body.key, body.value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"key": body.key, "status": "updated"}


@cognitive_router.get("/context")
async def get_context(task: str = Query(..., description="Task description for context loading")):
    """Load and return the full cognitive context for a given task."""
    loader = _get_loader()
    context = loader.load_context(task)
    system_prompt = loader.assemble_system_prompt(context)
    return {
        "context": context,
        "system_prompt": system_prompt,
    }


# ===========================================================================
# WebSocket: /ws/{job_id}
# ===========================================================================

@app.websocket("/ws/{job_id}")
async def websocket_job_stream(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint that streams AgentEvents for *job_id* in real time.

    Subscribes to the Redis pub/sub channel ``job:{job_id}:events`` and
    forwards each published message to the connected client.
    The connection is closed when a ``result`` or ``error`` event is received.
    """
    await websocket.accept()
    r = aioredis.from_url(settings.redis_url, encoding="utf-8", decode_responses=True)
    pubsub = r.pubsub()
    channel = f"job:{job_id}:events"
    await pubsub.subscribe(channel)

    try:
        # Send connected acknowledgement
        await websocket.send_json({"type": "connected", "job_id": job_id})

        async for message in pubsub.listen():
            if message["type"] == "message":
                raw = message["data"]
                await websocket.send_text(raw)
                try:
                    data = json.loads(raw)
                    if data.get("type") in ("result", "error"):
                        break
                except json.JSONDecodeError:
                    pass
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected for job %s", job_id)
    except Exception as exc:
        logger.error("WebSocket error for job %s: %s", job_id, exc)
        try:
            await websocket.send_json({"type": "error", "job_id": job_id, "data": {"error": str(exc)}})
        except Exception:
            pass
    finally:
        await pubsub.unsubscribe(channel)
        await r.aclose()
        try:
            await websocket.close()
        except Exception:
            pass


# ===========================================================================
# Register routers
# ===========================================================================

app.include_router(health_router)
app.include_router(jobs_router)
app.include_router(files_router)
app.include_router(cognitive_router)


# ===========================================================================
# Dev entry-point
# ===========================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info",
    )
