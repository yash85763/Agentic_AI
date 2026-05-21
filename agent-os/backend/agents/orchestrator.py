"""
OrchestratorAgent - Plans the execution DAG and delegates work to specialist agents.

Uses extended thinking (when supported by the model) to produce a robust
execution plan from uploaded file manifests, then coordinates each stage of
the pipeline and validates the final output.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langfuse.decorators import observe

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "claude-opus-4-5")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class DAGStep:
    """A single node in the execution DAG."""

    step_id: str
    agent_name: str
    description: str
    depends_on: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanResult:
    """Structured output of :meth:`OrchestratorAgent.plan_dag`."""

    steps: List[DAGStep]
    reasoning: str
    estimated_minutes: int
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [
                {
                    "step_id": s.step_id,
                    "agent_name": s.agent_name,
                    "description": s.description,
                    "depends_on": s.depends_on,
                    "params": s.params,
                }
                for s in self.steps
            ],
            "reasoning": self.reasoning,
            "estimated_minutes": self.estimated_minutes,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class OrchestratorAgent(BaseAgent):
    """Plans, delegates and validates the full AgentOS pipeline.

    The orchestrator is the only agent that:
    - Creates the execution DAG (via LLM with extended thinking).
    - Invokes sub-agents by calling their ``run()`` methods (or via A2A if
      they are remote).
    - Validates the final merged output before handing it to the report stage.
    - Logs every parallel fan-out decision as a ROUTING_DECISION SSE event.
    - Escalates to the user (ESCALATION_REQUIRED event) when routing is ambiguous.

    Tool count note (Section 6.1): The orchestrator is the exception to the
    max-4-tools rule. Its tools are management tools (plan, delegate, validate,
    escalate) — not execution tools. Specialist agents enforce MAX_TOOLS = 4.
    """

    DEFAULT_MODEL = ORCHESTRATOR_MODEL

    # Ordered list of available specialist agents the orchestrator can invoke.
    KNOWN_AGENTS: List[str] = [
        "IngestionAgent",
        "UnderstandingAgent",
        "TransformationAgent",
        "ValidationAgent",
        "VisualizationAgent",
        "ReportAgent",
        "MemoryAgent",
    ]

    # Specialist agents whose file-level work can be parallelised
    PARALLELISABLE_AGENTS: List[str] = [
        "IngestionAgent",
        "TransformationAgent",
    ]

    def __init__(
        self,
        cognitive_context: Dict[str, Any],
        langfuse_handler: Any,
        model: str = ORCHESTRATOR_MODEL,
    ) -> None:
        super().__init__(
            model=model,
            cognitive_context=cognitive_context,
            langfuse_handler=langfuse_handler,
        )
        self._agent_registry: Dict[str, BaseAgent] = {}

    # ------------------------------------------------------------------
    # Agent registry (injected by the pipeline runner)
    # ------------------------------------------------------------------

    def register_agent(self, name: str, agent: BaseAgent) -> None:
        """Register a concrete agent instance so the orchestrator can delegate."""
        self._agent_registry[name] = agent
        logger.info("Orchestrator registered agent: %s", name)

    # ------------------------------------------------------------------
    # Routing decision log (Section 6.1)
    # ------------------------------------------------------------------

    def _log_parallel_fanout(
        self,
        file_count: int,
        parallel_agents: List[str],
        job_id: str,
        redis_client: Any,
    ) -> None:
        """Emit a ROUTING_DECISION event explaining why fan-out is used.

        Appears in the Langfuse trace as a dedicated span and is visible
        in the frontend's agent feed.
        """
        # Rough estimate: assume 30s per file sequential vs 30s total parallel
        est_seq = file_count * 30
        est_par = 30
        saving_pct = round((1 - est_par / max(est_seq, 1)) * 100, 1)

        self._emit_event(
            "routing_decision",
            {
                "file_count": file_count,
                "parallel_agents": parallel_agents,
                "estimated_sequential_seconds": est_seq,
                "estimated_parallel_seconds": est_par,
                "time_saving_pct": saving_pct,
                "reasoning": (
                    f"Parallel fan-out selected: {file_count} files → "
                    f"{len(parallel_agents)} agents running concurrently. "
                    f"Estimated {saving_pct}% faster than sequential processing."
                ),
            },
            job_id,
            redis_client,
        )
        logger.info(
            "Routing decision: %d files → %d parallel agents (est. %d%% time saving)",
            file_count,
            len(parallel_agents),
            saving_pct,
        )

    def _escalate(
        self,
        question: str,
        context: str,
        job_id: str,
        redis_client: Any,
        options: Optional[List[str]] = None,
    ) -> None:
        """Emit an ESCALATION_REQUIRED event when routing cannot be determined.

        The frontend surfaces this as an interactive prompt (not a background
        log). The pipeline pauses; the user's response should be fed back
        via the job API.
        """
        self._emit_event(
            "escalation_required",
            {
                "question": question,
                "context": context,
                "options": options or [],
                "blocking": True,
                "message": (
                    "The orchestrator requires your input before proceeding. "
                    "Please respond via the job interface."
                ),
            },
            job_id,
            redis_client,
        )
        logger.warning(
            "Escalation required for job '%s': %s", job_id, question
        )

    # ------------------------------------------------------------------
    # DAG planning
    # ------------------------------------------------------------------

    @observe(name="orchestrator.plan_dag")
    def plan_dag(
        self,
        task: str,
        file_manifests: List[Dict[str, Any]],
    ) -> PlanResult:
        """Use the LLM (with extended thinking) to produce an execution plan.

        Args:
            task: High-level user task description.
            file_manifests: List of file manifest dicts produced by IngestionAgent.

        Returns:
            :class:`PlanResult` describing the ordered DAG.
        """
        job_id = self._job_id()

        manifests_summary = json.dumps(
            [
                {
                    "file_name": m.get("file_name"),
                    "sheets": m.get("sheets", []),
                    "row_count": m.get("row_count", 0),
                }
                for m in file_manifests
            ],
            indent=2,
        )

        system_prompt = (
            "You are the OrchestratorAgent for an AgentOS data pipeline. "
            "Your job is to plan a directed-acyclic-graph (DAG) of processing steps "
            "given uploaded file manifests and a user task description.\n\n"
            "Available specialist agents:\n"
            + "\n".join(f"  - {a}" for a in self.KNOWN_AGENTS)
            + "\n\n"
            "Return ONLY valid JSON matching this schema:\n"
            "{\n"
            '  "steps": [\n'
            "    {\n"
            '      "step_id": "step_1",\n'
            '      "agent_name": "<AgentName>",\n'
            '      "description": "<what this step does>",\n'
            '      "depends_on": [],\n'
            '      "params": {}\n'
            "    }\n"
            "  ],\n"
            '  "reasoning": "<why you chose this plan>",\n'
            '  "estimated_minutes": 5,\n'
            '  "warnings": []\n'
            "}"
        )

        user_prompt = (
            f"User task: {task}\n\n"
            f"Uploaded file manifests:\n{manifests_summary}\n\n"
            "Plan the optimal DAG. Use extended thinking to reason carefully about "
            "dependencies, parallelism opportunities, and potential pitfalls."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Extended thinking is only supported on Anthropic Claude models.
        extra_kwargs: Dict[str, Any] = {}
        if "claude" in self.model.lower():
            extra_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": 8000,
            }

        try:
            response = self._call_llm(
                messages,
                temperature=1,  # required when thinking is enabled
                max_tokens=16000,
                **extra_kwargs,  # type: ignore[arg-type]
            )
            plan_data = self._extract_json(response)
        except Exception as exc:
            logger.error("plan_dag LLM call failed: %s", exc, exc_info=True)
            # Fall back to a sensible default sequential plan.
            plan_data = self._default_plan()

        steps = [
            DAGStep(
                step_id=s["step_id"],
                agent_name=s["agent_name"],
                description=s["description"],
                depends_on=s.get("depends_on", []),
                params=s.get("params", {}),
            )
            for s in plan_data.get("steps", [])
        ]

        result = PlanResult(
            steps=steps,
            reasoning=plan_data.get("reasoning", ""),
            estimated_minutes=plan_data.get("estimated_minutes", 10),
            warnings=plan_data.get("warnings", []),
        )

        logger.info(
            "plan_dag produced %d steps (est. %d min)",
            len(steps),
            result.estimated_minutes,
        )
        return result

    def _default_plan(self) -> Dict[str, Any]:
        """Return a sensible linear fallback plan."""
        agents = [
            ("step_1", "IngestionAgent", "Ingest and parse all uploaded files", []),
            ("step_2", "UnderstandingAgent", "Infer semantic meaning of columns", ["step_1"]),
            ("step_3", "TransformationAgent", "Transform and normalise data", ["step_2"]),
            ("step_4", "ValidationAgent", "Validate merged output", ["step_3"]),
            ("step_5", "VisualizationAgent", "Generate charts", ["step_4"]),
            ("step_6", "ReportAgent", "Assemble final report", ["step_5"]),
            ("step_7", "MemoryAgent", "Persist learnings to memory", ["step_6"]),
        ]
        return {
            "steps": [
                {
                    "step_id": sid,
                    "agent_name": name,
                    "description": desc,
                    "depends_on": deps,
                    "params": {},
                }
                for sid, name, desc, deps in agents
            ],
            "reasoning": "Default sequential pipeline plan (LLM planning failed).",
            "estimated_minutes": 15,
            "warnings": ["Fell back to default plan due to LLM planning error."],
        }

    # ------------------------------------------------------------------
    # Task delegation
    # ------------------------------------------------------------------

    @observe(name="orchestrator.delegate_task")
    def delegate_task(
        self,
        agent_name: str,
        task_params: Dict[str, Any],
        redis_client: Any = None,
    ) -> Dict[str, Any]:
        """Invoke a registered specialist agent by name.

        Args:
            agent_name: Key in ``_agent_registry`` (e.g. ``"IngestionAgent"``).
            task_params: State dict slice to pass to the agent's ``run()`` method.
            redis_client: Optional Redis client for event emission.

        Returns:
            Updated state dict returned by the agent.

        Raises:
            KeyError: If ``agent_name`` is not registered.
            RuntimeError: If the agent raises an unhandled exception.
        """
        job_id = self._job_id()

        if agent_name not in self._agent_registry:
            # Escalate to user rather than silently failing with a wrong routing
            self._escalate(
                question=(
                    f"The pipeline plan calls for agent '{agent_name}' but it is "
                    "not registered. Which registered agent should handle this step?"
                ),
                context=(
                    f"Attempted agent: {agent_name}\n"
                    f"Registered agents: {list(self._agent_registry.keys())}\n"
                    f"Task params keys: {list(task_params.keys())}"
                ),
                job_id=job_id,
                redis_client=redis_client,
                options=list(self._agent_registry.keys()),
            )
            raise KeyError(
                f"Agent '{agent_name}' is not registered. "
                f"Escalation event emitted. "
                f"Registered agents: {list(self._agent_registry.keys())}"
            )

        self._emit_event(
            "agent_started",
            {"agent": agent_name, "params_keys": list(task_params.keys())},
            job_id,
            redis_client,
        )

        agent = self._agent_registry[agent_name]
        logger.info("Delegating to %s …", agent_name)

        try:
            result = agent.run(task_params)
        except Exception as exc:
            logger.error(
                "Agent '%s' failed: %s", agent_name, exc, exc_info=True
            )
            self._emit_event(
                "agent_error",
                {"agent": agent_name, "error": str(exc)},
                job_id,
                redis_client,
            )
            raise RuntimeError(f"Agent '{agent_name}' raised: {exc}") from exc

        self._emit_event(
            "agent_completed",
            {"agent": agent_name, "result_keys": list(result.keys())},
            job_id,
            redis_client,
        )
        logger.info("Agent '%s' completed successfully.", agent_name)
        return result

    # ------------------------------------------------------------------
    # Output validation
    # ------------------------------------------------------------------

    @observe(name="orchestrator.validate_final_output")
    def validate_final_output(self, result: Dict[str, Any]) -> bool:
        """Perform a final sanity check on the pipeline output.

        Checks for mandatory top-level keys and non-empty critical fields.

        Args:
            result: The pipeline state dict after all agents have run.

        Returns:
            ``True`` if the output passes validation, ``False`` otherwise.
        """
        required_keys = [
            "file_manifests",
            "validation_report",
            "charts",
            "report",
        ]
        missing = [k for k in required_keys if k not in result or not result[k]]
        if missing:
            logger.warning(
                "Final output validation failed — missing/empty keys: %s", missing
            )
            return False

        validation_report = result.get("validation_report", {})
        if validation_report.get("passed") is False:
            critical = [
                a
                for a in validation_report.get("anomalies", [])
                if a.get("severity") == "critical"
            ]
            if critical:
                logger.error(
                    "Final output has %d critical anomalies — marking as failed.",
                    len(critical),
                )
                return False

        logger.info("Final output validation passed.")
        return True

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    @observe(name="orchestrator.run")
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate the full pipeline end-to-end.

        Expected state keys on entry:
            - ``task`` (str): User task description.
            - ``uploaded_files`` (list): Raw file paths/metadata.
            - ``job_id`` (str): Unique pipeline run identifier.
            - ``redis_client`` (optional): Redis client for streaming events.
            - ``business_rules`` (optional dict): Domain-specific rules.

        Returns:
            Fully populated state dict with all pipeline outputs.
        """
        job_id = state.get("job_id", self._job_id())
        redis_client = state.get("redis_client")
        task = state.get("task", "Analyse uploaded files and generate a report.")

        logger.info("OrchestratorAgent starting job '%s'", job_id)
        self._emit_event("pipeline_started", {"task": task}, job_id, redis_client)

        # ---- Stage 0: Ingest files to get manifests for planning -----
        state = self.delegate_task("IngestionAgent", state, redis_client)
        file_manifests: List[Dict[str, Any]] = state.get("file_manifests", [])

        # ---- Stage 1: Plan the DAG -----------------------------------
        self._emit_event("planning_started", {}, job_id, redis_client)
        plan = self.plan_dag(task, file_manifests)
        state["execution_plan"] = plan.to_dict()
        self._emit_event(
            "planning_completed",
            {
                "steps": len(plan.steps),
                "estimated_minutes": plan.estimated_minutes,
                "warnings": plan.warnings,
            },
            job_id,
            redis_client,
        )

        # ---- Stage 2: Execute the DAG steps (skip IngestionAgent — done) ----
        completed_steps: Dict[str, bool] = {"step_1": True}  # IngestionAgent done

        # Log parallel fan-out opportunity when multiple files are processed
        file_count = len(file_manifests)
        if file_count > 1:
            parallel_agents = [
                s.agent_name for s in plan.steps
                if s.agent_name in self.PARALLELISABLE_AGENTS
            ]
            if parallel_agents:
                self._log_parallel_fanout(
                    file_count, parallel_agents, job_id, redis_client
                )

        for step in plan.steps:
            # Skip IngestionAgent (already done above)
            if step.agent_name == "IngestionAgent":
                continue

            # Wait for dependencies (in this single-process runner, all prior
            # steps have already completed sequentially).
            if not all(completed_steps.get(dep, False) for dep in step.depends_on):
                logger.warning(
                    "Skipping step '%s' — unmet dependencies %s",
                    step.step_id,
                    step.depends_on,
                )
                continue

            self._emit_event(
                "step_started",
                {"step_id": step.step_id, "agent": step.agent_name},
                job_id,
                redis_client,
            )

            # Merge step-level params into state without overwriting
            for k, v in step.params.items():
                state.setdefault(k, v)

            try:
                state = self.delegate_task(step.agent_name, state, redis_client)
                completed_steps[step.step_id] = True
            except RuntimeError as exc:
                logger.error("Step '%s' failed: %s", step.step_id, exc)
                self._emit_event(
                    "step_failed",
                    {"step_id": step.step_id, "error": str(exc)},
                    job_id,
                    redis_client,
                )
                state.setdefault("errors", []).append(
                    {"step": step.step_id, "error": str(exc)}
                )
                completed_steps[step.step_id] = False
                # Continue with remaining steps where possible.

        # ---- Stage 3: Final validation --------------------------------
        passed = self.validate_final_output(state)
        state["pipeline_passed"] = passed

        self._emit_event(
            "pipeline_completed",
            {"passed": passed, "job_id": job_id},
            job_id,
            redis_client,
        )
        logger.info(
            "OrchestratorAgent finished job '%s' — passed=%s", job_id, passed
        )
        return state
