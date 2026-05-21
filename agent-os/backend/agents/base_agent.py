"""
BaseAgent - Abstract base class for all AgentOS agents.

Every concrete agent inherits from this class and gains:
- litellm-based LLM calls with automatic retry
- Langfuse observability via @observe decorator
- Redis pub/sub event emission for real-time streaming
- Structured logging
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import litellm
from langfuse.decorators import observe

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all AgentOS agents.

    Subclasses must implement :meth:`run`.
    """

    # Override in subclasses to set the default model.
    DEFAULT_MODEL: str = "openai/gpt-4o-mini"

    def __init__(
        self,
        model: str,
        cognitive_context: Dict[str, Any],
        langfuse_handler: Any,
    ) -> None:
        """Initialise a base agent.

        Args:
            model: litellm model string (e.g. ``"claude-opus-4-5"`` or
                ``"ollama/qwen2.5-coder:32b"``).
            cognitive_context: Shared context dict passed down from the
                orchestrator (job_id, user_id, business_rules, …).
            langfuse_handler: A Langfuse ``CallbackHandler`` (or ``None``
                when tracing is disabled).  Stored for use by subclasses that
                call LLM chains directly.
        """
        self.model = model
        self.cognitive_context = cognitive_context
        self.langfuse_handler = langfuse_handler
        self.agent_name: str = self.__class__.__name__
        self._call_count: int = 0

        # litellm global settings
        litellm.drop_params = True  # silently ignore unsupported params
        litellm.set_verbose = os.getenv("LITELLM_VERBOSE", "false").lower() == "true"

        logger.info(
            "Agent '%s' initialised with model='%s'",
            self.agent_name,
            self.model,
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's primary task.

        Args:
            state: The current pipeline state dict shared among all agents.

        Returns:
            Updated pipeline state dict.
        """

    # ------------------------------------------------------------------
    # LLM call helper
    # ------------------------------------------------------------------

    @observe(name="base_agent._call_llm")
    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        max_retries: int = 3,
        fallback_model: Optional[str] = None,
    ) -> Any:
        """Call the configured LLM via litellm with automatic retry.

        Args:
            messages: OpenAI-style message list.
            tools: Optional list of tool/function definitions (OpenAI format).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.
            max_retries: Number of retry attempts on transient errors.
            fallback_model: If the primary model fails all retries, attempt
                this model once before raising.

        Returns:
            litellm ``ModelResponse`` object.

        Raises:
            litellm.APIError: If all retries (and fallback) are exhausted.
        """
        self._call_count += 1
        call_id = f"{self.agent_name}-call-{self._call_count}"

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                logger.debug(
                    "[%s] LLM call attempt %d/%d model=%s",
                    call_id,
                    attempt,
                    max_retries,
                    self.model,
                )
                response = litellm.completion(**kwargs)
                logger.debug(
                    "[%s] LLM call succeeded (attempt %d)",
                    call_id,
                    attempt,
                )
                return response
            except (
                litellm.RateLimitError,
                litellm.APIConnectionError,
                litellm.Timeout,
            ) as exc:
                last_error = exc
                wait = 2 ** attempt  # exponential back-off: 2, 4, 8 …
                logger.warning(
                    "[%s] Transient error on attempt %d/%d: %s — retrying in %ds",
                    call_id,
                    attempt,
                    max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
            except Exception as exc:
                logger.error(
                    "[%s] Non-retryable LLM error: %s",
                    call_id,
                    exc,
                    exc_info=True,
                )
                raise

        # Primary model exhausted — try fallback
        if fallback_model:
            logger.warning(
                "[%s] Primary model '%s' exhausted — falling back to '%s'",
                call_id,
                self.model,
                fallback_model,
            )
            kwargs["model"] = fallback_model
            try:
                return litellm.completion(**kwargs)
            except Exception as exc:
                logger.error(
                    "[%s] Fallback model '%s' also failed: %s",
                    call_id,
                    fallback_model,
                    exc,
                    exc_info=True,
                )
                raise exc from last_error

        raise RuntimeError(
            f"[{call_id}] All {max_retries} attempts failed for model '{self.model}'"
        ) from last_error

    # ------------------------------------------------------------------
    # Redis pub/sub event emitter
    # ------------------------------------------------------------------

    def _emit_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        job_id: str,
        redis_client: Any,
        *,
        channel_prefix: str = "agentos:jobs",
    ) -> None:
        """Publish a structured event to a Redis pub/sub channel.

        The channel name is ``{channel_prefix}:{job_id}``.  All subscribers
        (e.g. the FastAPI SSE endpoint) receive the event as a JSON string.

        Event envelope schema::

            {
                "event_id":    "<uuid4>",
                "job_id":      "<job_id>",
                "agent":       "<agent_name>",
                "event_type":  "<event_type>",
                "timestamp":   <unix_float>,
                "data":        { ... }
            }

        Args:
            event_type: Short label such as ``"progress"``, ``"result"``,
                ``"error"``, ``"started"``, ``"completed"``.
            data: Arbitrary payload dict attached to the event.
            job_id: Pipeline job identifier used to derive the channel name.
            redis_client: A ``redis.Redis`` (or ``redis.asyncio.Redis``) client
                instance.  Pass ``None`` to skip publishing (useful in tests).
            channel_prefix: Redis channel prefix (default ``agentos:jobs``).
        """
        if redis_client is None:
            logger.debug(
                "Redis client is None — skipping event emission (%s)", event_type
            )
            return

        channel = f"{channel_prefix}:{job_id}"
        envelope: Dict[str, Any] = {
            "event_id": str(uuid.uuid4()),
            "job_id": job_id,
            "agent": self.agent_name,
            "event_type": event_type,
            "timestamp": time.time(),
            "data": data,
        }
        try:
            payload = json.dumps(envelope, default=str)
            redis_client.publish(channel, payload)
            logger.debug(
                "Emitted event '%s' to channel '%s'", event_type, channel
            )
        except Exception as exc:
            # Never let event emission crash the pipeline.
            logger.warning(
                "Failed to emit event '%s' to Redis channel '%s': %s",
                event_type,
                channel,
                exc,
            )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def _extract_text(self, response: Any) -> str:
        """Extract the assistant text content from a litellm response."""
        try:
            return response.choices[0].message.content or ""
        except (AttributeError, IndexError) as exc:
            logger.warning("Could not extract text from LLM response: %s", exc)
            return ""

    def _extract_json(self, response: Any) -> Any:
        """Extract and parse JSON from a litellm response.

        Strips markdown code fences (``` or ```json) before parsing.
        """
        raw = self._extract_text(response).strip()
        # Strip common markdown fences
        for fence in ("```json", "```"):
            if raw.startswith(fence):
                raw = raw[len(fence):]
                break
        if raw.endswith("```"):
            raw = raw[:-3]
        return json.loads(raw.strip())

    def _job_id(self) -> str:
        """Return the job_id from cognitive_context, defaulting to a new UUID."""
        return self.cognitive_context.get("job_id", str(uuid.uuid4()))
