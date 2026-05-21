"""
AgentOS Chainlit UI — interactive chat interface for agent workflows.

Run: chainlit run chainlit_app.py --port 8001
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from typing import Any

import chainlit as cl
import httpx

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


@cl.on_chat_start
async def on_chat_start():
    """Initialize a new chat session."""
    cl.user_session.set("uploaded_files", [])
    cl.user_session.set("current_job_id", None)

    await cl.Message(
        content=(
            "## Welcome to AgentOS\n\n"
            "I orchestrate a team of specialist agents that ingest, transform, "
            "validate, and report on your data.\n\n"
            "**To start an analysis:**\n"
            "1. Upload one or more Excel/CSV files\n"
            "2. Describe what you want me to do\n"
            "3. I'll run the pipeline and stream agent activity here\n\n"
            "Drag-and-drop files or click the paperclip below."
        ),
        author="AgentOS",
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming chat messages."""
    # Check for file attachments
    uploaded = cl.user_session.get("uploaded_files", [])

    if message.elements:
        new_files = await _upload_attachments(message.elements)
        uploaded = uploaded + new_files
        cl.user_session.set("uploaded_files", uploaded)

        files_summary = "\n".join(f"  • `{f['filename']}` ({f['size']} bytes)" for f in new_files)
        await cl.Message(
            content=f"✅ Uploaded {len(new_files)} file(s):\n{files_summary}",
            author="AgentOS",
        ).send()

    task = (message.content or "").strip()
    if not task:
        if uploaded:
            await cl.Message(
                content="Files received. Now describe what you'd like me to do with them.",
                author="AgentOS",
            ).send()
        return

    if not uploaded:
        await cl.Message(
            content="⚠️ Please upload at least one data file before starting.",
            author="AgentOS",
        ).send()
        return

    # Launch pipeline
    await _run_pipeline(task, uploaded)


@cl.on_chat_end
async def on_chat_end():
    """Cleanup on chat end."""
    pass


# ─── Helpers ─────────────────────────────────────────────────────────────────

async def _upload_attachments(elements: list[Any]) -> list[dict]:
    """Upload file attachments to the backend, return file records."""
    file_elements = [e for e in elements if isinstance(e, cl.File)]
    if not file_elements:
        return []

    async with httpx.AsyncClient(timeout=60.0) as client:
        files = []
        for fe in file_elements:
            with open(fe.path, "rb") as f:
                content = f.read()
            files.append(("files", (fe.name, content, fe.mime or "application/octet-stream")))

        res = await client.post(f"{BACKEND_URL}/api/files/upload", files=files)
        res.raise_for_status()
        return res.json()


async def _run_pipeline(task: str, uploaded_files: list[dict]):
    """Submit a job and stream events as Chainlit Steps."""
    file_ids = [f["file_id"] for f in uploaded_files]

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Create job
        try:
            res = await client.post(
                f"{BACKEND_URL}/api/jobs",
                json={"task_description": task, "file_ids": file_ids},
            )
            res.raise_for_status()
            job = res.json()
        except Exception as exc:
            await cl.Message(
                content=f"❌ Failed to create job: {exc}",
                author="AgentOS",
            ).send()
            return

    job_id = job["id"]
    cl.user_session.set("current_job_id", job_id)

    await cl.Message(
        content=f"🚀 Job `{job_id[:8]}` started. Streaming agent activity...",
        author="AgentOS",
    ).send()

    # Stream events from the SSE endpoint
    await _consume_event_stream(job_id)

    # Clear uploaded files after submission
    cl.user_session.set("uploaded_files", [])


async def _consume_event_stream(job_id: str):
    """Connect to the backend SSE stream and render events as Chainlit Steps."""
    url = f"{BACKEND_URL}/api/jobs/{job_id}/stream"

    open_steps: dict[str, cl.Step] = {}

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", url) as response:
                buffer = ""
                async for chunk in response.aiter_text():
                    buffer += chunk
                    while "\n\n" in buffer:
                        raw_event, buffer = buffer.split("\n\n", 1)
                        event = _parse_sse_event(raw_event)
                        if not event:
                            continue

                        if await _render_event(event, open_steps):
                            return  # COMPLETE or ERROR received
    except Exception as exc:
        await cl.Message(
            content=f"⚠️ Event stream interrupted: {exc}",
            author="AgentOS",
        ).send()


def _parse_sse_event(raw: str) -> dict | None:
    """Parse a single SSE event block into a dict."""
    data_line = next((l for l in raw.split("\n") if l.startswith("data:")), None)
    if not data_line:
        return None
    try:
        return json.loads(data_line[5:].strip())
    except json.JSONDecodeError:
        return None


async def _render_event(event: dict, open_steps: dict[str, cl.Step]) -> bool:
    """Render a single agent event. Returns True if pipeline finished."""
    event_type = event.get("event_type", "")
    agent_name = event.get("agent_name", "")
    data = event.get("data", {})

    # Lifecycle events
    if event_type == "agent_start":
        step = cl.Step(name=agent_name, type="run")
        await step.send()
        open_steps[agent_name] = step
        return False

    if event_type == "agent_complete":
        step = open_steps.pop(agent_name, None)
        if step:
            step.output = json.dumps(data, indent=2)
            await step.update()
        return False

    # Content events
    if event_type == "thinking":
        step = open_steps.get(agent_name) or cl.Step(name=f"{agent_name} thinking", type="llm")
        if not step.id:
            await step.send()
        step.output = (step.output or "") + str(data.get("text", data))
        await step.update()
        return False

    if event_type == "code_generated":
        code = data.get("code", "")
        if isinstance(data, str):
            code = data
        step = cl.Step(name=f"{agent_name} — code", type="tool")
        step.language = "python"
        step.output = code
        await step.send()
        return False

    if event_type == "code_executing":
        await cl.Message(
            content=f"⚙️ Running code in Docker sandbox...",
            author=agent_name,
        ).send()
        return False

    if event_type == "code_result":
        step = cl.Step(name=f"{agent_name} — output", type="tool")
        step.output = json.dumps(data, indent=2, default=str)[:5000]
        await step.send()
        return False

    if event_type == "chart_ready":
        chart_data = data
        # Render ECharts via plotly placeholder (Chainlit's Plotly element can show static images;
        # for a richer view, link out to the React /report/:jobId page)
        msg = cl.Message(
            content=f"📊 Chart generated: **{chart_data.get('title', 'Untitled')}**\n"
                    f"_View interactive version in the report page._",
            author=agent_name,
        )
        await msg.send()
        return False

    if event_type == "validation":
        passed = data.get("passed", False)
        emoji = "✅" if passed else "❌"
        checks = data.get("checks", [])
        check_lines = "\n".join(
            f"  - {'✅' if c.get('passed') else '❌'} {c.get('name', '?')}: {c.get('message', '')}"
            for c in checks[:8]
        )
        await cl.Message(
            content=f"{emoji} **Validation {'passed' if passed else 'failed'}**\n{check_lines}",
            author=agent_name,
        ).send()
        return False

    if event_type == "report_section":
        section = data.get("section", "section")
        content = data.get("content", "")
        await cl.Message(
            content=f"### {section.replace('_', ' ').title()}\n\n{content}",
            author=agent_name,
        ).send()
        return False

    if event_type == "complete":
        job_id = event.get("job_id", "")
        await cl.Message(
            content=(
                f"🎉 **Pipeline complete!**\n\n"
                f"- Sections: {data.get('section_count', 0)}\n"
                f"- Charts: {data.get('chart_count', 0)}\n\n"
                f"[Open interactive report →](/report/{job_id})"
            ),
            author="AgentOS",
        ).send()
        return True

    if event_type == "error":
        await cl.Message(
            content=f"❌ **Error**: {data.get('message', 'Unknown error')}",
            author=agent_name,
        ).send()
        return True

    if event_type == "progress":
        # Silent progress — could update a progress indicator if Chainlit supported it
        return False

    return False
