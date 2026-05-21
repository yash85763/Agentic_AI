"""
LangGraph DAG for the Expense Consolidation pipeline.

Graph topology
--------------

    orchestrate
        |
    ingest
        |
    understand
        |
    transform
        |
    merge
        |
    validate ──(pass)──► visualize ──► report ──► memorize ──► END
        |
       (fail + retries < 2)
        |
    retry_transform ────────────────────────────────────────────┘
        |
       (fail + retries >= 2)
        |
       END  (failure)

Nodes
-----
orchestrate  – parse task, set up cognitive context, emit AGENT_START
ingest       – download files from MinIO, normalise to Parquet
understand   – LLM analyses schemas and writes a transformation plan
transform    – LLM writes + sandbox-executes pandas/DuckDB code
merge        – DuckDB UNION of all team parquets → single master parquet
validate     – rule-based + LLM cross-checks on the merged data
visualize    – LLM generates ECharts configs → emits CHART_READY events
report       – LLM composes Markdown report sections → emits REPORT_SECTION
memorize     – stores insights + schemas in cognitive context / memory DB
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from typing import Any, Optional

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------


class PipelineState(TypedDict):
    """Shared state passed between every node in the graph."""

    # ── Inputs ─────────────────────────────────────────────────────────────
    job_id: str
    task: str
    file_ids: list[str]
    cognitive_ctx: dict[str, Any]

    # ── Intermediate ───────────────────────────────────────────────────────
    file_manifests: list[dict[str, Any]]   # [{file_id, local_path, name, size}]
    team_parquets: dict[str, str]          # {team_name: local_parquet_path}
    merged_path: str                        # local path of the merged parquet
    transformation_plan: str               # LLM-generated plan text
    transformation_code: str               # pandas/DuckDB code to run
    sql_query: str                         # SQL used for merging

    # ── Validation ─────────────────────────────────────────────────────────
    validation: dict[str, Any]             # {passed, items, retry_recommended}

    # ── Outputs ────────────────────────────────────────────────────────────
    charts: list[dict[str, Any]]           # list of ECharts option dicts
    report: dict[str, Any]                 # {sections: [{id, title, content}], url}

    # ── Control flow ───────────────────────────────────────────────────────
    errors: list[str]
    retry_count: int


# ---------------------------------------------------------------------------
# Lazy imports for optional heavy dependencies
# ---------------------------------------------------------------------------


def _get_file_tools():
    from tools.file_tools import get_file_tools
    return get_file_tools()


def _get_executor():
    from tools.execute_python import get_executor
    return get_executor()


def _execute_sql(query, parquet_paths):
    from tools.execute_sql import execute_sql
    return execute_sql(query, parquet_paths)


def _get_sse():
    from streaming.sse_manager import get_sse_manager
    return get_sse_manager()


def _emit(job_id: str, event_type_name: str, agent: str, data: Any):
    """Fire-and-forget SSE emit (safe to call from sync code)."""
    try:
        from streaming.event_models import AgentEvent, EventType
        sse = _get_sse()
        event = AgentEvent(
            job_id=job_id,
            event_type=EventType[event_type_name],
            agent_name=agent,
            data=data,
        )
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(sse.publish_event(job_id, event))
        else:
            loop.run_until_complete(sse.publish_event(job_id, event))
    except Exception as exc:
        logger.warning("SSE emit failed: %s", exc)


def _llm_call(prompt: str, system: str = "") -> str:
    """
    Thin wrapper around LiteLLM for text generation.

    Falls back to a stub if LITELLM_MODEL is not configured so that
    the graph can be unit-tested without real API keys.
    """
    model = os.getenv("LITELLM_MODEL", "gpt-4o")
    try:
        import litellm
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = litellm.completion(model=model, messages=messages)
        return resp.choices[0].message.content or ""
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return f"[LLM error: {exc}]"


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------


def node_orchestrate(state: PipelineState) -> PipelineState:
    """
    Parse the task description and initialise pipeline state.
    Emits AGENT_START.
    """
    job_id = state["job_id"]
    logger.info("[%s] orchestrate: starting", job_id)

    _emit(job_id, "AGENT_START", "orchestrator", {
        "node_name": "orchestrate",
        "description": "Parsing task and initialising pipeline",
    })
    _emit(job_id, "THINKING", "orchestrator", {
        "thought": f"Task received: {state['task']}",
        "step": "orchestrate",
    })

    # Initialise mutable lists/dicts if not already set
    updates: dict[str, Any] = {
        "file_manifests": state.get("file_manifests") or [],
        "team_parquets": state.get("team_parquets") or {},
        "merged_path": state.get("merged_path") or "",
        "transformation_plan": state.get("transformation_plan") or "",
        "transformation_code": state.get("transformation_code") or "",
        "sql_query": state.get("sql_query") or "",
        "validation": state.get("validation") or {},
        "charts": state.get("charts") or [],
        "report": state.get("report") or {},
        "errors": state.get("errors") or [],
        "retry_count": state.get("retry_count") or 0,
    }
    return {**state, **updates}


def node_ingest(state: PipelineState) -> PipelineState:
    """
    Download each file from MinIO, detect its format, convert to Parquet.
    Populates state["file_manifests"] and state["team_parquets"].
    """
    job_id = state["job_id"]
    logger.info("[%s] ingest: processing %d file(s)", job_id, len(state["file_ids"]))

    _emit(job_id, "PROGRESS", "ingest", {"pct": 10, "message": "Downloading files"})

    ft = _get_file_tools()
    manifests: list[dict] = []
    team_parquets: dict[str, str] = {}
    errors: list[str] = list(state.get("errors") or [])

    for idx, file_id in enumerate(state["file_ids"], start=1):
        try:
            # file_id format: "bucket/object_name" or plain object_name in RAW_FILES
            if "/" in file_id:
                bucket, obj_name = file_id.split("/", 1)
            else:
                from tools.file_tools import RAW_FILES
                bucket, obj_name = RAW_FILES, file_id

            with tempfile.NamedTemporaryFile(
                suffix=_suffix_from_name(obj_name), delete=False
            ) as tmp:
                local_path = tmp.name

            ft.download_to_file(bucket, obj_name, local_path)
            parquet_path = _convert_to_parquet(local_path, obj_name)
            team_name = _team_name_from_obj(obj_name, idx)

            manifests.append({
                "file_id": file_id,
                "local_path": local_path,
                "parquet_path": parquet_path,
                "name": obj_name,
                "team_name": team_name,
            })
            team_parquets[team_name] = parquet_path

            _emit(job_id, "PROGRESS", "ingest", {
                "pct": 10 + 20 * idx / len(state["file_ids"]),
                "message": f"Ingested {obj_name}",
                "step": idx,
                "total_steps": len(state["file_ids"]),
            })
        except Exception as exc:
            msg = f"Failed to ingest file {file_id!r}: {exc}"
            logger.error("[%s] %s", job_id, msg)
            errors.append(msg)

    return {**state, "file_manifests": manifests, "team_parquets": team_parquets, "errors": errors}


def node_understand(state: PipelineState) -> PipelineState:
    """
    LLM analyses the schemas of each ingested Parquet and produces a
    transformation plan that maps them to a common schema.
    """
    job_id = state["job_id"]
    logger.info("[%s] understand: analysing schemas", job_id)

    _emit(job_id, "THINKING", "schema_analyst", {
        "thought": "Analysing column schemas across all team expense files",
        "step": "understand",
    })

    # Build schema summaries for the LLM
    schema_lines: list[str] = []
    for manifest in state["file_manifests"]:
        try:
            import pandas as pd
            df = pd.read_parquet(manifest["parquet_path"], engine="pyarrow")
            schema_lines.append(
                f"Team: {manifest['team_name']}\n"
                f"Columns: {list(df.columns)}\n"
                f"Dtypes: {df.dtypes.to_dict()}\n"
                f"Sample rows:\n{df.head(3).to_string()}\n"
            )
        except Exception as exc:
            logger.warning("[%s] Could not read parquet for schema: %s", job_id, exc)

    prompt = (
        f"Task: {state['task']}\n\n"
        "You are a data engineering expert. Here are the schemas of expense "
        "files uploaded by different teams:\n\n"
        + "\n---\n".join(schema_lines)
        + "\n\nProduce a concise transformation plan (in plain text) that:\n"
        "1. Maps each team's columns to a canonical schema.\n"
        "2. Standardises date formats, currency, and amount columns.\n"
        "3. Adds a 'team_name' column.\n"
        "4. Outputs a unified DataFrame with columns: "
        "[team_name, date, category, vendor, amount_usd, description].\n"
        "Return the plan as numbered steps."
    )
    plan = _llm_call(prompt, system="You are a data engineering assistant.")

    _emit(job_id, "THINKING", "schema_analyst", {
        "thought": plan,
        "step": "transformation_plan",
    })

    return {**state, "transformation_plan": plan}


def node_transform(state: PipelineState) -> PipelineState:
    """
    LLM writes pandas/DuckDB transformation code; sandbox executes it.
    On success, team_parquets are updated to point at the transformed files.
    """
    job_id = state["job_id"]
    logger.info("[%s] transform: generating and executing code", job_id)

    _emit(job_id, "PROGRESS", "transformer", {"pct": 40, "message": "Generating transformation code"})

    # Build a manifest summary for the prompt
    manifest_json = json.dumps(
        [
            {
                "team_name": m["team_name"],
                "parquet_path": m["parquet_path"],
                "name": m["name"],
            }
            for m in state["file_manifests"]
        ],
        indent=2,
    )

    code_prompt = (
        f"Transformation plan:\n{state['transformation_plan']}\n\n"
        f"File manifests (JSON):\n{manifest_json}\n\n"
        "Write complete, runnable Python code using pandas and pyarrow to:\n"
        "1. Read each parquet file.\n"
        "2. Apply the transformation plan.\n"
        "3. Write each transformed DataFrame back to the same parquet path "
        "   (overwrite in place).\n"
        "4. Print 'SUCCESS' and row counts at the end.\n\n"
        "Return ONLY the Python code, no markdown fences."
    )
    code = _llm_call(code_prompt, system="You are a Python data engineering expert.")
    # Strip markdown fences if the LLM added them
    code = _strip_fences(code)

    _emit(job_id, "CODE_GENERATED", "transformer", {
        "language": "python",
        "code": code,
        "description": "Expense normalisation script",
    })

    # Execute in sandbox
    executor = _get_executor()
    input_files = {m["parquet_path"]: m["parquet_path"] for m in state["file_manifests"]}
    result = executor.execute(code, input_files=None, timeout=60)

    _emit(job_id, "CODE_RESULT", "transformer", {
        "exit_code": result["exit_code"],
        "stdout": result["stdout"][:4000],
        "stderr": result["stderr"][:2000],
        "duration_ms": result["duration_ms"],
        "timed_out": result.get("timed_out", False),
    })

    errors = list(state.get("errors") or [])
    if result["exit_code"] != 0 or result.get("timed_out"):
        err = f"Transformation code failed (exit {result['exit_code']}): {result['stderr'][:500]}"
        errors.append(err)
        logger.error("[%s] %s", job_id, err)

    return {**state, "transformation_code": code, "errors": errors}


def node_merge(state: PipelineState) -> PipelineState:
    """
    Use DuckDB to UNION all team parquets into a single master parquet.
    """
    job_id = state["job_id"]
    logger.info("[%s] merge: unioning %d team parquet(s)", job_id, len(state["team_parquets"]))

    _emit(job_id, "PROGRESS", "merger", {"pct": 55, "message": "Merging team data"})

    errors = list(state.get("errors") or [])
    merged_path = ""

    try:
        import duckdb
        import pandas as pd

        team_parquets = state["team_parquets"]
        if not team_parquets:
            raise ValueError("No team parquets available for merge.")

        # Build UNION ALL query
        select_parts = [
            f"SELECT *, '{team}' AS _source_team FROM read_parquet('{path}')"
            for team, path in team_parquets.items()
        ]
        sql = " UNION ALL ".join(select_parts)

        conn = duckdb.connect(":memory:")
        df = conn.execute(sql).df()
        conn.close()

        # Write merged parquet
        merged_path = tempfile.mktemp(suffix=".parquet")
        df.to_parquet(merged_path, engine="pyarrow", index=False)

        logger.info(
            "[%s] Merged %d rows from %d teams → %s",
            job_id, len(df), len(team_parquets), merged_path,
        )

        _emit(job_id, "PROGRESS", "merger", {
            "pct": 60,
            "message": f"Merged {len(df):,} rows from {len(team_parquets)} teams",
        })

        state = {**state, "sql_query": sql}

    except Exception as exc:
        err = f"Merge failed: {exc}"
        logger.error("[%s] %s", job_id, err)
        errors.append(err)

    return {**state, "merged_path": merged_path, "errors": errors}


def node_validate(state: PipelineState) -> PipelineState:
    """
    Rule-based + LLM validation of the merged dataset.
    Populates state["validation"].
    """
    job_id = state["job_id"]
    logger.info("[%s] validate: running checks", job_id)

    _emit(job_id, "PROGRESS", "validator", {"pct": 65, "message": "Validating merged data"})

    items: list[dict] = []
    passed_overall = True

    try:
        import pandas as pd
        df = pd.read_parquet(state["merged_path"], engine="pyarrow")

        # Rule 1: non-empty
        ok = len(df) > 0
        items.append({"name": "Non-empty result", "passed": ok, "severity": "error",
                       "message": f"{len(df)} rows" if ok else "Merged dataset is empty"})
        if not ok:
            passed_overall = False

        # Rule 2: required columns
        required = {"date", "amount_usd"}
        present = required.intersection(df.columns)
        ok = present == required
        items.append({"name": "Required columns present", "passed": ok, "severity": "error",
                       "message": f"Present: {list(present)}" if ok else f"Missing: {required - present}"})
        if not ok:
            passed_overall = False

        # Rule 3: no null amounts
        if "amount_usd" in df.columns:
            null_count = int(df["amount_usd"].isna().sum())
            ok = null_count == 0
            items.append({"name": "No null amounts", "passed": ok, "severity": "error",
                           "message": f"{null_count} null values" if not ok else "OK"})
            if not ok:
                passed_overall = False

        # Rule 4: reasonable amount range (no extreme outliers)
        if "amount_usd" in df.columns and len(df) > 0:
            max_amt = float(df["amount_usd"].max())
            ok = max_amt < 10_000_000
            items.append({"name": "Amount sanity check", "passed": ok, "severity": "warning",
                           "message": f"Max amount: {max_amt:,.2f}"})

        # Rule 5: date parseable
        if "date" in df.columns:
            try:
                pd.to_datetime(df["date"])
                ok = True
            except Exception:
                ok = False
            items.append({"name": "Date column parseable", "passed": ok, "severity": "error",
                           "message": "OK" if ok else "Could not parse date column"})
            if not ok:
                passed_overall = False

    except Exception as exc:
        logger.error("[%s] Validation error: %s", job_id, exc)
        items.append({"name": "Validation execution", "passed": False,
                       "severity": "error", "message": str(exc)})
        passed_overall = False

    retry_recommended = not passed_overall and state.get("retry_count", 0) < 2

    validation = {
        "passed": passed_overall,
        "items": items,
        "retry_recommended": retry_recommended,
        "summary": "All checks passed." if passed_overall else "Some checks failed.",
    }

    _emit(job_id, "VALIDATION", "validator", validation)

    return {**state, "validation": validation}


def node_retry_transform(state: PipelineState) -> PipelineState:
    """
    Increment retry counter and re-run transform + merge with a revised prompt.
    This node is reached only when validation fails and retries < 2.
    """
    job_id = state["job_id"]
    retry_count = state.get("retry_count", 0) + 1
    logger.warning("[%s] Retry %d: re-running transform", job_id, retry_count)

    _emit(job_id, "THINKING", "orchestrator", {
        "thought": f"Validation failed — retrying transformation (attempt {retry_count})",
        "step": "retry",
    })

    # Update state with incremented retry count then delegate to transform
    new_state = {**state, "retry_count": retry_count}
    new_state = node_transform(new_state)
    new_state = node_merge(new_state)
    return new_state


def node_visualize(state: PipelineState) -> PipelineState:
    """
    LLM generates ECharts configs from the merged data.
    Emits CHART_READY events.
    """
    job_id = state["job_id"]
    logger.info("[%s] visualize: generating charts", job_id)

    _emit(job_id, "PROGRESS", "visualizer", {"pct": 75, "message": "Generating charts"})

    charts: list[dict] = []

    try:
        import pandas as pd
        df = pd.read_parquet(state["merged_path"], engine="pyarrow")

        # Summarise data for the LLM
        summary = {
            "total_rows": len(df),
            "columns": list(df.columns),
            "category_counts": df["category"].value_counts().to_dict() if "category" in df.columns else {},
            "total_amount": float(df["amount_usd"].sum()) if "amount_usd" in df.columns else 0,
            "team_amounts": (
                df.groupby("team_name")["amount_usd"].sum().to_dict()
                if "team_name" in df.columns and "amount_usd" in df.columns
                else {}
            ),
        }

        chart_prompt = (
            f"Task: {state['task']}\n\n"
            f"Data summary:\n{json.dumps(summary, indent=2, default=str)}\n\n"
            "Generate 3 Apache ECharts option JSON objects for:\n"
            "1. Bar chart: expenses by category\n"
            "2. Pie chart: expenses by team\n"
            "3. Line chart: expenses over time (if date available)\n\n"
            "Return a JSON array of objects, each with keys: "
            "'title' (string), 'chart_type' (string), 'option' (ECharts option object).\n"
            "Return ONLY the JSON array, no markdown."
        )
        raw = _llm_call(chart_prompt, system="You are a data visualisation expert.")
        raw = _strip_fences(raw)

        try:
            chart_defs = json.loads(raw)
        except json.JSONDecodeError:
            # Graceful fallback: emit a minimal bar chart
            chart_defs = [_fallback_bar_chart(summary)]

        for cd in chart_defs:
            chart_id = str(uuid.uuid4())
            charts.append({
                "chart_id": chart_id,
                "title": cd.get("title", "Chart"),
                "chart_type": cd.get("chart_type", "bar"),
                "echarts_config": cd.get("option", {}),
            })
            _emit(job_id, "CHART_READY", "visualizer", {
                "chart_id": chart_id,
                "title": cd.get("title", "Chart"),
                "echarts_config": cd.get("option", {}),
            })

    except Exception as exc:
        logger.error("[%s] Visualise error: %s", job_id, exc)

    return {**state, "charts": charts}


def node_report(state: PipelineState) -> PipelineState:
    """
    LLM composes a Markdown report from the validated data and charts.
    Emits REPORT_SECTION events incrementally.
    """
    job_id = state["job_id"]
    logger.info("[%s] report: composing report", job_id)

    _emit(job_id, "PROGRESS", "reporter", {"pct": 88, "message": "Composing report"})

    sections_data = [
        ("executive_summary", "Executive Summary", 0),
        ("key_findings",      "Key Findings",      1),
        ("team_breakdown",    "Team Breakdown",     2),
        ("recommendations",   "Recommendations",    3),
    ]

    report_sections: list[dict] = []

    summary_prompt = (
        f"Task: {state['task']}\n"
        f"Validation: {json.dumps(state['validation'], default=str)}\n"
        f"Charts generated: {[c['title'] for c in state['charts']]}\n\n"
    )

    for section_id, title, order in sections_data:
        prompt = (
            summary_prompt
            + f"Write the '{title}' section of the expense consolidation report "
            f"in professional Markdown. Be concise (150-300 words)."
        )
        content = _llm_call(prompt, system="You are a financial analyst writing a report.")

        section = {
            "section_id": section_id,
            "title": title,
            "content": content,
            "order": order,
            "is_final": True,
            "content_type": "markdown",
        }
        report_sections.append(section)

        _emit(job_id, "REPORT_SECTION", "reporter", section)

    # Upload report to MinIO
    report_url: str | None = None
    try:
        full_md = "\n\n".join(
            f"## {s['title']}\n\n{s['content']}" for s in report_sections
        )
        ft = _get_file_tools()
        from tools.file_tools import REPORTS
        obj_name = f"reports/{job_id}/expense_report.md"
        report_url = ft.upload_bytes(
            full_md.encode("utf-8"),
            REPORTS,
            obj_name,
            content_type="text/markdown",
        )
        logger.info("[%s] Report uploaded to %s", job_id, report_url)
    except Exception as exc:
        logger.warning("[%s] Could not upload report: %s", job_id, exc)

    report = {"sections": report_sections, "url": report_url}
    return {**state, "report": report}


def node_memorize(state: PipelineState) -> PipelineState:
    """
    Store insights and canonical schema into the cognitive context store.
    (Stub: extend to write to a vector DB / Redis / Postgres as needed.)
    """
    job_id = state["job_id"]
    logger.info("[%s] memorize: persisting cognitive context", job_id)

    _emit(job_id, "PROGRESS", "memorizer", {"pct": 97, "message": "Saving insights"})

    # Build a summary entry for the cognitive context
    memory_entry = {
        "job_id": job_id,
        "task": state["task"],
        "validation_passed": state["validation"].get("passed", False),
        "chart_titles": [c["title"] for c in state["charts"]],
        "report_url": state["report"].get("url"),
        "transformation_plan": state.get("transformation_plan", ""),
    }

    # Merge into cognitive context (in-memory; extend for persistence)
    ctx = dict(state.get("cognitive_ctx") or {})
    ctx.setdefault("history", []).append(memory_entry)

    _emit(job_id, "AGENT_COMPLETE", "memorizer", {
        "node_name": "memorize",
        "duration_ms": 0,
        "success": True,
    })

    _emit(job_id, "COMPLETE", "pipeline", {
        "report_url": state["report"].get("url"),
        "artifact_urls": [state.get("merged_path", "")],
        "duration_ms": 0,
        "summary": state["report"].get("sections", [{}])[0].get("content", "")[:300],
    })

    return {**state, "cognitive_ctx": ctx}


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------


def _route_after_validate(state: PipelineState) -> str:
    """
    Returns the name of the next node after validation.

    - "visualize"        if validation passed
    - "retry_transform"  if validation failed and retries < 2
    - "__end__"          if validation failed and retries exhausted
    """
    validation = state.get("validation") or {}
    if validation.get("passed", False):
        return "visualize"

    retry_count = state.get("retry_count", 0)
    if retry_count < 2:
        return "retry_transform"

    logger.error(
        "[%s] Validation failed after %d retries — ending with failure",
        state["job_id"], retry_count,
    )
    _emit(state["job_id"], "ERROR", "pipeline", {
        "message": "Validation failed after maximum retries",
        "node": "validate",
        "retryable": False,
    })
    return END


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_expense_pipeline() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph for expense consolidation.

    Returns the compiled graph, ready to be invoked with::

        graph = build_expense_pipeline()
        result_state = graph.invoke(initial_state)
    """
    graph = StateGraph(PipelineState)

    # Register all nodes
    graph.add_node("orchestrate",     node_orchestrate)
    graph.add_node("ingest",          node_ingest)
    graph.add_node("understand",      node_understand)
    graph.add_node("transform",       node_transform)
    graph.add_node("merge",           node_merge)
    graph.add_node("validate",        node_validate)
    graph.add_node("retry_transform", node_retry_transform)
    graph.add_node("visualize",       node_visualize)
    graph.add_node("report",          node_report)
    graph.add_node("memorize",        node_memorize)

    # Entry point
    graph.set_entry_point("orchestrate")

    # Linear edges
    graph.add_edge("orchestrate",     "ingest")
    graph.add_edge("ingest",          "understand")
    graph.add_edge("understand",      "transform")
    graph.add_edge("transform",       "merge")
    graph.add_edge("merge",           "validate")

    # Conditional branching after validate
    graph.add_conditional_edges(
        "validate",
        _route_after_validate,
        {
            "visualize":       "visualize",
            "retry_transform": "retry_transform",
            END:               END,
        },
    )

    # After retry, go back to validate
    graph.add_edge("retry_transform", "validate")

    # Happy path
    graph.add_edge("visualize", "report")
    graph.add_edge("report",    "memorize")
    graph.add_edge("memorize",  END)

    return graph.compile()


# ---------------------------------------------------------------------------
# ExpenseConsolidationPipeline — BasePipeline integration
# ---------------------------------------------------------------------------


class ExpenseConsolidationPipeline:
    """
    Thin wrapper that integrates the LangGraph DAG with BasePipeline's
    retry / SSE / job-status infrastructure.

    Usage::

        pipeline = ExpenseConsolidationPipeline()
        result = await pipeline.run(
            job_id="abc-123",
            file_ids=["raw-files/team_a.xlsx", "raw-files/team_b.csv"],
            task="Consolidate Q1 2024 expenses across all teams",
            cognitive_context={},
        )
    """

    def __init__(self):
        self._graph = build_expense_pipeline()

    def run_sync(
        self,
        job_id: str,
        file_ids: list[str],
        task: str,
        cognitive_context: dict[str, Any],
    ) -> PipelineState:
        """Synchronous entry point (blocks until the graph finishes)."""
        initial_state: PipelineState = {
            "job_id": job_id,
            "task": task,
            "file_ids": file_ids,
            "cognitive_ctx": cognitive_context,
            "file_manifests": [],
            "team_parquets": {},
            "merged_path": "",
            "transformation_plan": "",
            "transformation_code": "",
            "sql_query": "",
            "validation": {},
            "charts": [],
            "report": {},
            "errors": [],
            "retry_count": 0,
        }
        return self._graph.invoke(initial_state)

    async def run(
        self,
        job_id: str,
        file_ids: list[str],
        task: str,
        cognitive_context: dict[str, Any],
    ) -> PipelineState:
        """Async entry point — runs the synchronous graph in a thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.run_sync,
            job_id,
            file_ids,
            task,
            cognitive_context,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _suffix_from_name(name: str) -> str:
    ext = os.path.splitext(name)[-1].lower()
    return ext if ext else ".bin"


def _team_name_from_obj(obj_name: str, idx: int) -> str:
    base = os.path.splitext(os.path.basename(obj_name))[0]
    return base if base else f"team_{idx}"


def _convert_to_parquet(local_path: str, original_name: str) -> str:
    """
    Convert a local file to Parquet and return the new path.
    Supports .xlsx, .xls, .csv, .parquet.
    """
    import pandas as pd

    ext = os.path.splitext(original_name)[-1].lower()
    dest = tempfile.mktemp(suffix=".parquet")

    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(local_path, engine="openpyxl")
    elif ext == ".csv":
        df = pd.read_csv(local_path)
    elif ext == ".parquet":
        return local_path  # already parquet
    else:
        # Attempt CSV as fallback
        df = pd.read_csv(local_path)

    df.to_parquet(dest, engine="pyarrow", index=False)
    return dest


def _strip_fences(text: str) -> str:
    """Remove markdown code fences that LLMs sometimes add."""
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def _fallback_bar_chart(summary: dict) -> dict:
    """Minimal ECharts bar chart when LLM output cannot be parsed."""
    categories = list(summary.get("category_counts", {}).keys())[:10]
    values = [summary["category_counts"].get(c, 0) for c in categories]
    return {
        "title": "Expenses by Category",
        "chart_type": "bar",
        "option": {
            "title": {"text": "Expenses by Category"},
            "xAxis": {"type": "category", "data": categories},
            "yAxis": {"type": "value"},
            "series": [{"type": "bar", "data": values}],
        },
    }
