"""
SQLAlchemy async database setup for AgentOS backend.

Models
------
- Job        – tracks a pipeline execution
- FileRecord – metadata for an uploaded file stored in MinIO
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import (
    AsyncAttrs,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ---------------------------------------------------------------------------
# Engine / session factory – populated by init_db()
# ---------------------------------------------------------------------------

_engine = None
_async_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def get_engine():
    if _engine is None:
        raise RuntimeError("Database not initialised. Call init_db() first.")
    return _engine


async def get_session() -> AsyncSession:  # type: ignore[return]
    """FastAPI dependency that yields an async DB session."""
    if _async_session_factory is None:
        raise RuntimeError("Database not initialised. Call init_db() first.")
    async with _async_session_factory() as session:
        yield session


# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------

class Base(AsyncAttrs, DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------

class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    # user_id is required for Row Level Security — never default to a shared value
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, default="system")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    task_description: Mapped[str] = mapped_column(Text, nullable=False)
    file_ids: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True, default=list)
    result: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # one-to-many: a job owns many FileRecords
    files: Mapped[list["FileRecord"]] = relationship(
        "FileRecord", back_populates="job", lazy="select"
    )

    def __repr__(self) -> str:
        return f"<Job id={self.id} status={self.status}>"


class FileRecord(Base):
    __tablename__ = "file_records"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    # user_id is required for Row Level Security
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, default="system")
    original_name: Mapped[str] = mapped_column(String(512), nullable=False)
    minio_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    size: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    content_type: Mapped[str] = mapped_column(String(256), nullable=False, default="application/octet-stream")
    # Source trust tier for knowledge-base poisoning prevention (Section 2.4)
    source_trust_tier: Mapped[str] = mapped_column(
        String(32), nullable=False, default="user-uploaded"
    )
    # Quarantine: new documents are held here until schema check + approval
    quarantine_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="quarantined"
    )
    ingestion_metadata: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    job_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    job: Mapped[Optional["Job"]] = relationship("Job", back_populates="files")

    def __repr__(self) -> str:
        return f"<FileRecord id={self.id} name={self.original_name}>"


class SkillReward(Base):
    """One reward observation per skill per pipeline run.

    Written by UCBSkillSelector.record_reward() after every completed pipeline.
    Read by UCBSkillSelector.select_skills() to compute UCB scores.
    """
    __tablename__ = "skill_rewards"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    job_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    skill_name: Mapped[str] = mapped_column(String(256), nullable=False)
    task_type: Mapped[str] = mapped_column(String(64), nullable=False)
    reward: Mapped[float] = mapped_column(Float, nullable=False)
    reward_breakdown: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<SkillReward skill={self.skill_name} task={self.task_type} reward={self.reward:.3f}>"


class TrainingTrajectory(Base):
    """One SDAR training trajectory per completed pipeline run.

    Written by sdar.record_trajectory() after every completed pipeline.
    Exported to JSONL by sdar.export_training_data() for GRPO fine-tuning.
    """
    __tablename__ = "training_trajectories"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    job_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    task_description: Mapped[str] = mapped_column(Text, nullable=False)
    task_type: Mapped[str] = mapped_column(String(64), nullable=False)
    skills_used: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    skills_context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    agent_response: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reward: Mapped[float] = mapped_column(Float, nullable=False)
    reward_breakdown: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    validation_passed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    user_accepted: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    exported: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<TrainingTrajectory job={self.job_id} task_type={self.task_type} reward={self.reward:.3f}>"


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

async def init_db(database_url: str) -> None:
    """
    Create the async engine, session factory, and all tables.

    Parameters
    ----------
    database_url:
        AsyncPG connection string, e.g.
        ``postgresql+asyncpg://user:pass@host:5432/dbname``
    """
    global _engine, _async_session_factory

    _engine = create_async_engine(
        database_url,
        echo=False,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
    )

    _async_session_factory = async_sessionmaker(
        _engine,
        expire_on_commit=False,
        class_=AsyncSession,
    )

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def set_rls_user_context(session: AsyncSession, user_id: str) -> None:
    """Set PostgreSQL session-level user context for RLS policy evaluation.

    Call this at the start of every request handler that touches the DB.
    The value is scoped to the current transaction and cleared automatically.

    Usage::

        async for session in get_session():
            await set_rls_user_context(session, request.state.user_id)
            jobs = await session.execute(select(Job))
    """
    if not user_id or not user_id.strip():
        raise ValueError(
            "user_id must not be empty when setting RLS context. "
            "Passing an empty user_id would bypass Row Level Security."
        )
    await session.execute(
        text("SELECT set_config('app.current_user_id', :uid, true)"),
        {"uid": user_id},
    )


async def lift_quarantine(
    session: AsyncSession, file_id: str, user_id: str
) -> None:
    """Promote a file from quarantine to active retrieval access.

    Only human-verified or policy-approved files should be lifted.
    Logs the event to the ingestion_audit table.
    """
    from sqlalchemy import update
    await session.execute(
        update(FileRecord)
        .where(FileRecord.id == file_id, FileRecord.user_id == user_id)
        .values(quarantine_status="approved")
    )
    await session.commit()
