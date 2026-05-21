"""
Server-Sent Events (SSE) manager backed by Redis pub/sub.

Architecture
------------
- Each analysis job has a Redis channel: ``job:{job_id}:events``
- Events are published as JSON-serialised AgentEvent objects
- The last 100 events per job are stored in a Redis list
  ``job:{job_id}:history`` so that clients that reconnect can
  replay missed events before subscribing to live updates
- FastAPI endpoint uses ``job_event_stream`` as an async generator

Usage (FastAPI)
---------------
::

    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
    from streaming.sse_manager import SSEManager

    app = FastAPI()
    sse = SSEManager()

    @app.get("/api/jobs/{job_id}/stream")
    async def stream_job(job_id: str, request: Request):
        return StreamingResponse(
            sse.job_event_stream(job_id, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import AsyncGenerator

import redis.asyncio as aioredis

from streaming.event_models import AgentEvent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHANNEL_PREFIX = "job"
EVENT_CHANNEL_SUFFIX = "events"
HISTORY_SUFFIX = "history"
MAX_HISTORY = 100
KEEPALIVE_INTERVAL = 15  # seconds between SSE comment keepalives
RECONNECT_RETRY_MS = 3000  # SSE retry hint sent to client


def _channel(job_id: str) -> str:
    return f"{CHANNEL_PREFIX}:{job_id}:{EVENT_CHANNEL_SUFFIX}"


def _history_key(job_id: str) -> str:
    return f"{CHANNEL_PREFIX}:{job_id}:{HISTORY_SUFFIX}"


# ---------------------------------------------------------------------------
# SSEManager
# ---------------------------------------------------------------------------


class SSEManager:
    """
    Manages publish/subscribe for job event streams over Redis.

    Parameters
    ----------
    redis_url:
        Redis DSN (default: REDIS_URL env var, falling back to
        ``redis://localhost:6379/0``).
    history_ttl:
        TTL in seconds for the history list (default 24 h).
    """

    def __init__(
        self,
        redis_url: str | None = None,
        history_ttl: int = 86_400,
    ):
        self._redis_url = redis_url or os.getenv(
            "REDIS_URL", "redis://localhost:6379/0"
        )
        self._history_ttl = history_ttl
        # Shared async Redis client for publishing / history ops
        self._redis: aioredis.Redis | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = await aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis

    async def _get_pubsub(self) -> aioredis.client.PubSub:
        """Create a *new* dedicated PubSub connection (one per subscriber)."""
        client = await aioredis.from_url(
            self._redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        return client.pubsub(ignore_subscribe_messages=True)

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    async def publish_event(self, job_id: str, event: AgentEvent) -> None:
        """
        Publish *event* to the job's Redis channel and append it to
        the history list (capped at MAX_HISTORY entries).
        """
        r = await self._get_redis()
        payload = event.model_dump_json()
        channel = _channel(job_id)
        history_key = _history_key(job_id)

        # Pipeline: publish + rpush + ltrim + expire atomically
        pipe = r.pipeline()
        pipe.publish(channel, payload)
        pipe.rpush(history_key, payload)
        pipe.ltrim(history_key, -MAX_HISTORY, -1)
        pipe.expire(history_key, self._history_ttl)
        await pipe.execute()

        logger.debug(
            "Published event %s [%s] on channel %s",
            event.event_id,
            event.event_type.value,
            channel,
        )

    # ------------------------------------------------------------------
    # Retrieve history
    # ------------------------------------------------------------------

    async def get_event_history(self, job_id: str) -> list[AgentEvent]:
        """Return the stored history (up to MAX_HISTORY) for *job_id*."""
        r = await self._get_redis()
        raw_events = await r.lrange(_history_key(job_id), 0, -1)
        events: list[AgentEvent] = []
        for raw in raw_events:
            try:
                events.append(AgentEvent.model_validate_json(raw))
            except Exception as exc:
                logger.warning("Failed to deserialise history event: %s", exc)
        return events

    async def clear_history(self, job_id: str) -> None:
        """Delete the history list for *job_id*."""
        r = await self._get_redis()
        await r.delete(_history_key(job_id))

    # ------------------------------------------------------------------
    # SSE streaming
    # ------------------------------------------------------------------

    async def job_event_stream(
        self,
        job_id: str,
        request=None,
    ) -> AsyncGenerator[str, None]:
        """
        Async generator that yields SSE-formatted strings.

        Steps
        -----
        1. Send the SSE ``retry`` directive.
        2. Replay historical events (so reconnecting clients catch up).
        3. Subscribe to live Redis channel and yield new events.
        4. Send periodic keepalive comments to prevent proxy timeouts.
        5. Detect client disconnect via the optional FastAPI *request*
           object and exit cleanly.

        Parameters
        ----------
        job_id:
            The job to stream events for.
        request:
            A FastAPI ``Request`` object (optional).  When provided, the
            generator will stop as soon as the client disconnects.
        """

        # SSE: tell the client how long to wait before reconnecting
        yield f"retry: {RECONNECT_RETRY_MS}\n\n"

        # ----------------------------------------------------------------
        # 1. Replay history
        # ----------------------------------------------------------------
        try:
            history = await self.get_event_history(job_id)
        except Exception as exc:
            logger.warning("Could not fetch history for %s: %s", job_id, exc)
            history = []

        for event in history:
            yield _format_sse(event)
            logger.debug("Replayed history event %s", event.event_id)

        # ----------------------------------------------------------------
        # 2. Subscribe to live channel
        # ----------------------------------------------------------------
        pubsub = await self._get_pubsub()
        channel = _channel(job_id)
        await pubsub.subscribe(channel)
        logger.info("SSE client subscribed to channel %s", channel)

        keepalive_task: asyncio.Task | None = None
        message_queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def _reader():
            """Background task: read from pubsub and push to queue."""
            try:
                async for message in pubsub.listen():
                    if message is None:
                        continue
                    if message.get("type") != "message":
                        continue
                    raw = message.get("data", "")
                    await message_queue.put(raw)
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.error("PubSub reader error on %s: %s", channel, exc)
            finally:
                await message_queue.put(None)  # sentinel

        async def _keepalive():
            """Send a comment every KEEPALIVE_INTERVAL seconds."""
            try:
                while True:
                    await asyncio.sleep(KEEPALIVE_INTERVAL)
                    await message_queue.put(": keepalive\n\n")
            except asyncio.CancelledError:
                pass

        reader_task = asyncio.create_task(_reader())
        keepalive_task = asyncio.create_task(_keepalive())

        try:
            while True:
                # Check for client disconnect (FastAPI Request)
                if request is not None and await request.is_disconnected():
                    logger.info("SSE client disconnected from job %s", job_id)
                    break

                try:
                    raw = await asyncio.wait_for(message_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                if raw is None:
                    # Sentinel: reader finished
                    break

                # Keepalive comments are already formatted
                if raw.startswith(":"):
                    yield raw
                    continue

                # Parse and format the event
                try:
                    event = AgentEvent.model_validate_json(raw)
                    yield _format_sse(event)
                    # Stop streaming after terminal events
                    if event.event_type.value in ("complete", "error"):
                        logger.info(
                            "Terminal event %s received for job %s — closing stream",
                            event.event_type.value, job_id,
                        )
                        break
                except Exception as exc:
                    logger.warning(
                        "Failed to parse event from channel %s: %s", channel, exc
                    )
                    # Forward the raw data rather than silently dropping it
                    yield f"data: {raw}\n\n"

        finally:
            reader_task.cancel()
            if keepalive_task:
                keepalive_task.cancel()
            try:
                await pubsub.unsubscribe(channel)
                await pubsub.aclose()
            except Exception:
                pass
            logger.info("SSE stream closed for job %s", job_id)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the shared Redis connection."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_sse(event: AgentEvent) -> str:
    """Convert an AgentEvent to an SSE-formatted string."""
    payload = event.model_dump_json()
    lines = [
        f"id: {event.event_id}",
        f"event: {event.event_type.value}",
        f"data: {payload}",
        "",  # blank line terminates the SSE message
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# FastAPI router factory (optional convenience)
# ---------------------------------------------------------------------------


def create_sse_router(sse_manager: SSEManager):
    """
    Create a FastAPI APIRouter with a single SSE endpoint.

    Mount with::

        from streaming.sse_manager import create_sse_router, SSEManager
        sse = SSEManager()
        app.include_router(create_sse_router(sse), prefix="/api")
    """
    try:
        from fastapi import APIRouter, Request
        from fastapi.responses import StreamingResponse
    except ImportError:
        raise ImportError(
            "fastapi is required to use create_sse_router(). "
            "Install it with: pip install fastapi"
        )

    router = APIRouter()

    @router.get(
        "/jobs/{job_id}/stream",
        summary="Stream job events as SSE",
        response_class=StreamingResponse,
    )
    async def stream_job_events(job_id: str, request: Request):
        return StreamingResponse(
            sse_manager.job_event_stream(job_id, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return router


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_manager: SSEManager | None = None


def get_sse_manager() -> SSEManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = SSEManager()
    return _default_manager
