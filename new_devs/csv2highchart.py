# main.py

# pip install fastapi python-dotenv langchain langchain-openai jinja2

import os
import re
import json
import uuid
import base64
from fastapi import FastAPI, Request, HTTPException, Form, File, UploadFile, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import asyncio
import logging
from datetime import datetime, timezone

# --- Imports for LangChain and OpenAI ---
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Load environment variables from a .env file
load_dotenv()

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Setup ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --- In-memory store for task statuses ---
task_store = {}

# --- Simple In-Memory Conversation History ---
# A dictionary to hold conversation history for each session
conversation_histories = {}

# --- System Prompt for the AI ---
SYSTEM_PROMPT = """
**Persona:**
You are an expert web developer specializing in data visualization with Highcharts. Your code is clean, well-documented, and you prioritize creating responsive and visually appealing user interfaces.

**Objective:**
Based on the user's request, the conversation history, and any provided error feedback, create a single, self-contained, and responsive HTML file that renders the appropriate Highcharts chart.

**Core Requirements:**
1.  **Analyze the Request and Error:** Look at the user's latest request. Also, review the conversation history for context. If the user provides data, use it. If the user does not provide data, you MUST generate a realistic set of dummy data that fits the user's request. If there is error feedback, you MUST fix the previous code based on the error message. The goal is to produce a chart that renders without any JavaScript errors.
2.  **Frameworks:** Use Highcharts for the charting library and Tailwind CSS for all styling.
3.  **Output:** Generate a complete, runnable HTML file.

**Implementation Steps:**
1.  **HTML Structure:**
    - Start with a standard HTML5 boilerplate.
    - Set the viewport for responsive behavior.
    - Include a `<script>` tag for the Tailwind CSS CDN.
    - The `<body>` should have a neutral background color (e.g., `bg-gray-100`).
    - Create a main container that centers the chart on the page with appropriate padding.
    - Inside the main container, add a chart container `<div>` (e.g., `<div id="container"></div>`) and give it a white background, rounded corners, and a subtle box shadow.

2.  **Highcharts Scripts (VERY IMPORTANT):**
    - **The core `highcharts.js` script MUST be included BEFORE any other Highcharts module scripts** (like `sankey.js`, `exporting.js`, etc.). The order is critical.
      **Always include the following Highcharts JavaScript libraries in this specific order**:
        <script src="https://code.highcharts.com/highcharts.js"></script>
        <script src="https://code.highcharts.com/highcharts-more.js"></script>
        <script src="https://code.highcharts.com/highcharts-3d.js"></script>
        <script src="https://code.highcharts.com/modules/stock.js"></script>
        <script src="https://code.highcharts.com/maps/modules/map.js"></script>
        <script src="https://code.highcharts.com/modules/gantt.js"></script>
        <script src="https://code.highcharts.com/dashboards/datagrid.js"></script>
        <script src="https://code.highcharts.com/dashboards/dashboards.js"></script>
        <script src="https://code.highcharts.com/dashboards/modules/layout.js"></script>
        <script src="https://code.highcharts.com/modules/exporting.js"></script>
        <script src="https://code.highcharts.com/modules/parallel-coordinates.js"></script>
        <script src="https://code.highcharts.com/modules/accessibility.js"></script>
        <script src="https://code.highcharts.com/modules/annotations-advanced.js"></script>
        <script src="https://code.highcharts.com/modules/data.js"></script>
        <script src="https://code.highcharts.com/modules/draggable-points.js"></script>
        <script src="https://code.highcharts.com/modules/static-scale.js"></script>
        <script src="https://code.highcharts.com/modules/broken-axis.js"></script>
        <script src="https://code.highcharts.com/modules/heatmap.js"></script>
        <script src="https://code.highcharts.com/modules/tilemap.js"></script>
        <script src="https://code.highcharts.com/modules/timeline.js"></script>
        <script src="https://code.highcharts.com/modules/treemap.js"></script>
        <script src="https://code.highcharts.com/modules/treegraph.js"></script>
        <script src="https://code.highcharts.com/modules/item-series.js"></script>
        <script src="https://code.highcharts.com/modules/drilldown.js"></script>
        <script src="https://code.highcharts.com/modules/histogram-bellcurve.js"></script>
        <script src="https://code.highcharts.com/modules/bullet.js"></script>
        <script src="https://code.highcharts.com/modules/funnel.js"></script>
        <script src="https://code.highcharts.com/modules/funnel3d.js"></script>
        <script src="https://code.highcharts.com/modules/pyramid3d.js"></script>
        <script src="https://code.highcharts.com/modules/networkgraph.js"></script>
        <script src="https://code.highcharts.com/modules/pareto.js"></script>
        <script src="https://code.highcharts.com/modules/pattern-fill.js"></script>
        <script src="https://code.highcharts.com/modules/price-indicator.js"></script>
        <script src="https://code.highcharts.com/modules/sankey.js"></script>
        <script src="https://code.highcharts.com/modules/arc-diagram.js"></script>
        <script src="https://code.highcharts.com/modules/dependency-wheel.js"></script>
        <script src="https://code.highcharts.com/modules/series-label.js"></script>
        <script src="https://code.highcharts.com/modules/solid-gauge.js"></script>
        <script src="https://code.highcharts.com/modules/sonification.js"></script>
        <script src="https://code.highcharts.com/modules/streamgraph.js"></script>
        <script src="https://code.highcharts.com/modules/sunburst.js"></script>
        <script src="https://code.highcharts.com/modules/variable-pie.js"></script>
        <script src="https://code.highcharts.com/modules/variwide.js"></script>
        <script src="https://code.highcharts.com/modules/vector.js"></script>
        <script src="https://code.highcharts.com/modules/venn.js"></script>
        <script src="https://code.highcharts.com/modules/windbarb.js"></script>
        <script src="https://code.highcharts.com/modules/wordcloud.js"></script>
        <script src="https://code.highcharts.com/modules/xrange.js"></script>
        <script src="https://code.highcharts.com/modules/no-data-to-display.js"></script>
        <script src="https://code.highcharts.com/modules/drag-panes.js"></script>
        <script src="https://code.highcharts.com/modules/debugger.js"></script>
        <script src="https://code.highcharts.com/modules/dumbbell.js"></script>
        <script src="https://code.highcharts.com/modules/lollipop.js"></script>
        <script src="https://code.highcharts.com/modules/cylinder.js"></script>
        <script src="https://code.highcharts.com/modules/organization.js"></script>
        <script src="https://code.highcharts.com/modules/dotplot.js"></script>
        <script src="https://code.highcharts.com/modules/marker-clusters.js"></script>
        <script src="https://code.highcharts.com/modules/hollowcandlestick.js"></script>
        <script src="https://code.highcharts.com/modules/heikinashi.js"></script>
        <script src="https://code.highcharts.com/modules/full-screen.js"></script>

    - Include the minimum necessary Highcharts modules from their official CDN to render the specified chart. You MUST determine which modules are needed based on the requested chart type. For example, a Sankey diagram requires the 'sankey.js' module. Always include 'exporting.js' and 'accessibility.js'.


3.  **Highcharts Configuration (in a `<script>` tag):**
- Wait for the DOM to be fully loaded before initializing the chart.
    - `title`: Set an appropriate title for the chart based on the user's request.
    - `series`:
        * Set the 'type' to the lowercase version of the chart type (e.g., 'sankey', 'pie').
        * Provide a name for the series.
        * Use the provided data (or the dummy data you generated), ensuring it's correctly parsed.
        * Set the `keys` property appropriately based on the data structure.
    - `tooltip`: Customize the tooltip to display data points in a a clear and meaningful way.
    - `credits`: Disable the Highcharts.com credits.

4.  **Error Correction Logic:**
    - If there is any `error_feedback`, **You must fix the script loading order.**

5.  **Code Quality:**
    - Add comments to both the HTML and JavaScript sections to explain the structure, styling, and Highcharts configuration.
    - Ensure the code is correctly formatted and indented.
    - Respond ONLY with the raw HTML code. Do NOT include backticks, "html" markers, or any commentary. The response must start with `<!DOCTYPE html>`.
"""


async def run_generation_in_background(task_id: str, session_id: str, prompt: str, error_feedback: str | None, image_bytes: bytes | None, mime_type: str | None):
    """
    Handles the core AI chart generation logic in a background task.

    This function constructs the full prompt for the LLM, including the system
    prompt, conversation history, the user's request, and any error feedback from
    a previous attempt. It then invokes the OpenAI model, processes the response,
    updates the conversation history, and stores the final result or error in the
    global `task_store`.

    Args:
        task_id: The unique identifier for this generation task.
        session_id: The identifier for the user's current session.
        prompt: The user's text prompt for the chart.
        error_feedback: Optional error message from a previous failed generation attempt.
        image_bytes: Optional bytes of an uploaded image (currently unused).
        mime_type: Optional MIME type of the uploaded image (currently unused).
    """
    try:
        # --- Get conversation history from memory ---
        conversation_history_messages = conversation_histories.get(session_id, [])
        
        # --- Construct the message list for the LLM ---
        messages_for_llm = [SystemMessage(content=SYSTEM_PROMPT)]
        messages_for_llm.extend(conversation_history_messages)
        
        # Combine the user's prompt and any error feedback into the final message
        current_user_prompt = prompt
        if error_feedback:
            current_user_prompt += f"\n\n--- PREVIOUS ERROR ---\nPlease fix the code based on the following error:\n{error_feedback}"
        
        messages_for_llm.append(HumanMessage(content=current_user_prompt))

        # --- Initialize LLM and make the call ---
        logger.info("Invoking OpenAI for content generation...")
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = ""
        model = "gpt-4.1-2025-04-14"

        os.environ["OPENAI_API_BASE"] = base_url
        os.environ["OPENAI_API_KEY"] = api_key
        
        llm = ChatOpenAI(temperature=0, model_name=model)
        
        response = await llm.ainvoke(messages_for_llm)
        generated_html = response.content

        # --- Update conversation memory ---
        conversation_histories[session_id] = conversation_history_messages + [
            HumanMessage(content=prompt),
            AIMessage(content=generated_html)
        ]

        # --- Process and store the result ---
        html_match = re.search(r'<!DOCTYPE html>', generated_html, re.IGNORECASE | re.DOTALL)
        if html_match:
            start_index = html_match.start()
            cleaned_html = generated_html[start_index:].strip()
        else:
            cleaned_html = generated_html.strip().removeprefix('```html').removesuffix('```').strip()
        
        task_store[task_id] = {
            "status": "SUCCESS",
            "result": {
                "html_code": cleaned_html,
                "original_prompt": prompt 
            }
        }
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}", exc_info=True)
        task_store[task_id] = {"status": "FAILED", "result": {"detail": str(e)}}


# --- API Endpoints ---
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serves the main single-page application `index.html`.

    Args:
        request: The incoming FastAPI request object.

    Returns:
        A TemplateResponse that renders the main HTML page.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/generate")
async def generate_chart_request(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    session_id: str = Form(...),
    error_feedback: str | None = Form(None),
    file: UploadFile | None = File(None)
):
    """
    Accepts a chart generation request and starts it as a background task.

    This endpoint immediately returns a task ID to the client, allowing the
    long-running AI generation process to run in the background without blocking
    the response. The client can then use the task ID to poll for the result.

    Args:
        background_tasks: FastAPI dependency to schedule background tasks.
        prompt: The user's text prompt for the chart.
        session_id: The identifier for the user's session.
        error_feedback: Optional error string from a previous attempt.
        file: Optional uploaded file (currently unused).

    Returns:
        A JSONResponse with the unique task_id and a 202 Accepted status code.
    """
    task_id = str(uuid.uuid4())
    image_bytes = await file.read() if file else None
    mime_type = file.content_type if file else None

    task_store[task_id] = {"status": "PENDING", "result": None}
    background_tasks.add_task(
        run_generation_in_background,
        task_id=task_id,
        session_id=session_id,
        prompt=prompt,
        error_feedback=error_feedback,
        image_bytes=image_bytes,
        mime_type=mime_type
    )

    return JSONResponse(content={"task_id": task_id}, status_code=202)

@app.post("/api/reset")
async def reset_memory(session_id: str = Form(...)):
    """
    Resets and clears the conversation memory for a given session ID.

    This allows a user to start a new conversation from scratch without
    the AI being influenced by previous requests in the same session.

    Args:
        session_id: The session identifier to be cleared.

    Returns:
        A JSONResponse confirming success or indicating that the session was not found.
    """
    if session_id in conversation_histories:
        del conversation_histories[session_id]
        logger.info(f"Memory for session {session_id} has been cleared.")
        return JSONResponse(content={"message": "Memory cleared successfully."})
    else:
        logger.warning(f"No memory found for session {session_id} to clear.")
        return JSONResponse(content={"message": "No active session to clear."}, status_code=404)



@app.get("/api/status/{task_id}")
async def get_task_status(task_id: str):
    """
    Allows the client to poll for the status of a background generation task.

    The frontend uses this endpoint to check if the chart generation is
    'PENDING', 'SUCCESS', or 'FAILED'.

    Args:
        task_id: The unique identifier of the task being polled.

    Returns:
        A JSONResponse containing the current status and result of the task.
        Raises an HTTPException with a 404 status if the task is not found.
    """
    task = task_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return JSONResponse(content=task)
    
    