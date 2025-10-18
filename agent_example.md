Awesome—here’s a compact, end-to-end set of **drop-in code pieces** showing how to implement each core concept (autonomy, memory, reasoning, planning, execution, and a simple RL loop) using **LangGraph + LangChain**. I’ve avoided `langchain_community` so you won’t hit the arXiv wrapper issue; the examples use `langgraph`, `langchain_core`, `langchain_openai`, and `chromadb` directly.

> Minimal deps you’ll likely need:
> `pip install langgraph langchain-core langchain-openai chromadb pydantic typing-extensions`

---

# 0) Common setup (LLM, state, tools)

```python
from __future__ import annotations
from typing import TypedDict, List, Literal, Optional
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# --- LLM ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# --- Agent State ---
class AgentState(TypedDict, total=False):
    messages: List[AnyMessage]
    plan: Optional[str]
    result: Optional[str]
    iterations: int
    budget_tokens: int
    reasoning_style: Literal["cot", "reflect", "tot"]
    score: Optional[float]        # for RL reward tracking
    stop: bool

# --- Tools (examples) ---
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web (stub). Replace with your real search function or API."""
    return f"[search results for: {query}]"

@tool
def run_calc(expr: str) -> str:
    """A tiny calculator (eval guarded in real life)."""
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"error: {e}"

tools = [search_web, run_calc]
tool_node = ToolNode(tools=tools)
```

---

# 1) Autonomy (controller + guardrails + budgets)

**Idea:** A controller node decides whether to continue, replan, or finish, enforcing limits (iterations, token budget, allowed tools).

```python
MAX_ITERS = 6
TOKEN_BUDGET = 8_000

def controller(state: AgentState) -> AgentState:
    iters = state.get("iterations", 0)
    if iters >= MAX_ITERS:
        return {**state, "stop": True,
                "messages": state["messages"] + [AIMessage("Stopping: max iterations reached.")]}
    if state.get("budget_tokens", TOKEN_BUDGET) <= 0:
        return {**state, "stop": True,
                "messages": state["messages"] + [AIMessage("Stopping: token budget exhausted.")]}
    return state  # continue
```

**Routing based on autonomy policy:**

```python
def autonomy_condition(state: AgentState):
    if state.get("stop"):
        return "stop"
    # If the last AI message used a tool or produced a plan, go execute
    last = state["messages"][-1] if state["messages"] else None
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    if state.get("plan"):
        return "act"
    return "plan"
```

---

# 2) Planning (structure the work)

**Idea:** Ask the LLM to produce a compact, actionable plan.

```python
plan_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a planning module. Produce a minimal step-by-step plan to
                solve the user goal. Include which tools to call if needed."),
    ("human", "{question}")
])

def plan_node(state: AgentState) -> AgentState:
    # Extract the user question from the earliest HumanMessage
    q = next((m.content for m in state["messages"] if isinstance(m, HumanMessage)), "")
    msg = plan_prompt.format_messages(question=q)
    resp = llm.invoke(msg)
    return {**state, "plan": resp.content, "messages": state["messages"] + [resp]}
```

---

# 3) Reasoning (CoT / Reflection / ToT switches)

**Idea:** Switch reasoning style per task. The controller (or RL, below) sets `reasoning_style`.

```python
cot_prompt = ChatPromptTemplate.from_messages([
    ("system", "Think step by step. Only include necessary steps."),
    ("human", "{task}")
])

reflect_prompt = ChatPromptTemplate.from_messages([
    ("system", "First produce a concise answer, then reflect with 2-3 bullet critiques,
    then refine the answer."),
    ("human", "{task}")
])

tot_prompt = ChatPromptTemplate.from_messages([
    ("system", "Propose 2-3 distinct solution branches (bulleted). For each,
    outline steps and a quick feasibility score (0-1). Conclude with the best branch and final answer."),
    ("human", "{task}")
])

def reason_node(state: AgentState) -> AgentState:
    style = state.get("reasoning_style", "cot")
    task = state.get("plan") or next((m.content for m in state["messages"] if isinstance(m, HumanMessage)), "")
    prompt = {"cot": cot_prompt, "reflect": reflect_prompt, "tot": tot_prompt}[style]
    resp = llm.invoke(prompt.format_messages(task=task))
    return {**state, "messages": state["messages"] + [resp]}
```

---

# 4) Execution (tool calling with LangGraph)

**Idea:** Let the model decide to call a tool; `ToolNode` executes; results return as messages.

```python
from langgraph.prebuilt import tools_condition

# The 'act' node encourages the LLM to call tools based on the plan
act_prompt = ChatPromptTemplate.from_messages([
    ("system", "You can call tools if needed. If the plan requires external info or math,
    use the proper tool. Otherwise answer directly."),
    ("human", "Plan:\n{plan}\n\nContinue.")
])

def act_node(state: AgentState) -> AgentState:
    resp = llm.bind_tools(tools).invoke(
        act_prompt.format_messages(plan=state.get("plan", ""))
    )
    # Note: if tool calls exist, the autonomy_condition will route to tool_node
    return {**state, "messages": state["messages"] + [resp]}
```

---

# 5) Reflection & termination (quality gate)

**Idea:** A simple self-critique that either finalizes or asks to replan.

```python
judge_prompt = ChatPromptTemplate.from_messages([
    ("system", "Judge correctness, completeness, and safety.
    Output one of: FINISH, REPLAN, CONTINUE. Provide a 0-1 score."),
    ("human", "Conversation so far:\n{history}\n\nYour decision:")
])

def reflect_node(state: AgentState) -> AgentState:
    history = "\n".join(m.content for m in state["messages"]
    if isinstance(m, (HumanMessage, AIMessage)))
    resp = llm.invoke(judge_prompt.format_messages(history=history))
    text = resp.content.upper()
    score = 1.0 if "0.9" in text or "1.0" in text or "1" in text else 0.6  # toy scoring
    if "FINISH" in text:
        return {**state, "result": history, "score": score,
    "messages": state["messages"] + [resp], "stop": True}
    if "REPLAN" in text:
        return {**state, "plan": None, "messages": state["messages"] + [resp]}
    return {**state, "messages": state["messages"] + [resp]}  # CONTINUE
```

---

# 6) Short-term memory (message history per session)

**Idea:** Use LangChain’s `RunnableWithMessageHistory` (conversation memory) for **ephemeral context**, or just keep it inside the graph state (`messages`) as shown. If you prefer LangChain wrappers:

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

store = {}  # session_id -> history
def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Wrap LLM with history:
conversational_llm = RunnableWithMessageHistory(
    llm,
    get_history,
    input_messages_key="messages",
    history_messages_key="messages",
)
```

> In the LangGraph above, we already persist `messages` in `AgentState`, which functions as short-term memory for the loop.

---

# 7) Long-term memory (vector DB via **Chroma** directly, no `langchain_community`)

**Idea:** Store “episodes” (problem → plan → answer) and retrieve later.

```python
import chromadb
from chromadb.config import Settings
from uuid import uuid4

# --- Create / connect ---
client = chromadb.PersistentClient(path="./agent_memory")
mem = client.get_or_create_collection(
    name="episodes",
    metadata={"hnsw:space": "cosine"}  # optional
)

def write_episode(query: str, plan: str, answer: str, tags: dict | None = None):
    doc_id = str(uuid4())
    doc = f"QUERY:\n{query}\n\nPLAN:\n{plan}\n\nANSWER:\n{answer}"
    mem.add(documents=[doc], ids=[doc_id], metadatas=[tags or {}])
    return doc_id

def retrieve_memory(query: str, k: int = 3) -> list[str]:
    res = mem.query(query_texts=[query], n_results=k)
    return res.get("documents", [[]])[0]
```

**Integrate as a “memory tool”:**

```python
@tool
def recall_episodes(query: str) -> str:
    """Retrieve similar past episodes from long-term memory."""
    docs = retrieve_memory(query, k=3)
    return "\n\n---\n\n".join(docs)

tools_with_memory = tools + [recall_episodes]
tool_node_with_memory = ToolNode(tools=tools_with_memory)
```

On successful completion, persist:

```python
def persist_on_finish(state: AgentState):
    user_q = next((m.content for m in state["messages"] if isinstance(m, HumanMessage)), "")
    plan = state.get("plan") or ""
    last_ai = next((m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage)), "")
    write_episode(user_q, plan, last_ai, tags={"domain": "general"})
    return state
```

---

# 8) Lightweight Reinforcement Learning (strategy bandit)

**Idea:** Use a **contextual multi-armed bandit** to choose the reasoning style (`cot`, `reflect`, `tot`) and **update** it from an internal reward (`score` from the judge or your own evaluator). This is simple but effective and doesn’t require training the LLM.

```python
# Very small in-memory bandit (swap for SQLite if you want persistence)
import random
from collections import defaultdict

arms = ["cot", "reflect", "tot"]
counts = defaultdict(int)
values = defaultdict(float)
EPS = 0.2  # exploration

def choose_arm() -> str:
    if random.random() < EPS:
        return random.choice(arms)
    if not counts:  # cold start
        return "cot"
    # pick highest average value
    return max(arms, key=lambda a: values[a] / (counts[a] or 1e-9))

def update_arm(arm: str, reward: float):
    counts[arm] += 1
    # incremental mean
    values[arm] += (reward - values[arm]) / counts[arm]
```

**Use it in the graph’s START hook and after reflection:**

```python
def assign_strategy(state: AgentState) -> AgentState:
    arm = choose_arm()
    return {**state, "reasoning_style": arm}

def learn_from_outcome(state: AgentState) -> AgentState:
    arm = state.get("reasoning_style", "cot")
    reward = state.get("score", 0.0)
    update_arm(arm, reward)
    return state
```

> You can plug in a **task-specific reward**: pass\@k on a unit test, BLEU/ROUGE for summarization, a human rating, a cost/latency penalty, etc.

---

# 9) Building the LangGraph

```python
workflow = StateGraph(AgentState)

# Nodes
workflow.add_node("controller", controller)
workflow.add_node("assign_strategy", assign_strategy)
workflow.add_node("plan", plan_node)
workflow.add_node("reason", reason_node)
workflow.add_node("act", act_node)
workflow.add_node("reflect", reflect_node)
workflow.add_node("learn", learn_from_outcome)
workflow.add_node("persist", persist_on_finish)
workflow.add_node("tools", tool_node_with_memory)  # includes memory tool

# Edges
workflow.add_edge(START, "assign_strategy")
workflow.add_edge("assign_strategy", "controller")
workflow.add_conditional_edges("controller", autonomy_condition, {
    "stop": "learn",      # when stop=True, record reward & exit
    "plan": "plan",
    "act": "act",
    "tools": "tools",
})
workflow.add_edge("plan", "reason")
workflow.add_conditional_edges("reason", tools_condition)  # if tool call, go to tools
workflow.add_edge("reason", "reflect")
workflow.add_conditional_edges("act", tools_condition)     # if tool call, go to tools
workflow.add_edge("tools", "reflect")
workflow.add_edge("reflect", "controller")

# When learning is done, persist memory and end
workflow.add_edge("learn", "persist")
workflow.add_edge("persist", END)

agent = workflow.compile()
```

**Run it:**

```python
init_state: AgentState = {
    "messages": [HumanMessage("Find present value of $10,000 received in 3 years at 7% and justify with sources.")],
    "iterations": 0,
    "budget_tokens": TOKEN_BUDGET
}
final = agent.invoke(init_state)
print("Reasoning style used:", final.get("reasoning_style"))
print("Score (reward):", final.get("score"))
print("Stopped:", final.get("stop"))
```

---

# 10) Safety, observability, and best practices (quick add-ons)

**Guardrails (domain/cost constraints):**

```python
ALLOWED_TOOLS = {"search_web", "run_calc", "recall_episodes"}

def enforce_guardrails(state: AgentState) -> AgentState:
    # Strip any attempted disallowed tool calls from the last AIMessage
    last = state["messages"][-1] if state["messages"] else None
    if isinstance(last, AIMessage) and last.tool_calls:
        last.tool_calls = [tc for tc in last.tool_calls if tc["name"] in ALLOWED_TOOLS]
    return state
```

**Cost/latency budget accounting** (stub):

```python
def decrement_budget(state: AgentState, used_tokens: int = 200) -> AgentState:
    budget = state.get("budget_tokens", TOKEN_BUDGET) - used_tokens
    return {**state, "budget_tokens": budget}
```

You can insert these as mini-nodes before/after LLM calls to make the agent **cost-aware** and **safe by construction**.

---

## How these pieces map to your slide concepts

* **Autonomy** → `controller`, `autonomy_condition`, budgets, guardrails, and the iterative loop in LangGraph.
* **Planning** → `plan_node` generates a concrete, tool-aware plan.
* **Reasoning** → `reason_node` switches among CoT/Reflection/ToT.
* **Execution** → `act_node` + `ToolNode` + `tools_condition` route tool calls and return observations.
* **Memory (short-term)** → `messages` inside the graph state (or `RunnableWithMessageHistory`).
* **Memory (long-term)** → Chroma collection (`write_episode`, `retrieve_memory`) exposed via a `recall_episodes` tool; `persist_on_finish` stores outcomes.
* **Reinforcement Learning** → ε-greedy bandit controlling `reasoning_style` and updating from the `reflect_node`’s score.

If you want, I can tailor this skeleton to a **specific domain** (e.g., **RegTech**, **FinTech analytics**, **PDF extraction agents**) and swap in your real tools (databases, retrieval, calculators, web searchers) while keeping the same architecture.









# How Search Works on a JSON File 🔍

Great question! The search doesn’t query the JSON directly like a database. Instead, it uses a **two-phase approach**:

## 📋 Phase 1: Crawling (Pre-computation)

**During crawling**, embeddings are generated and **stored inside the JSON**:

```python
# In KnowledgeGraphCrawler.crawl()
for url in queue:
    # Extract page content
    text = extract_page_body_content(soup)
    
    # Generate embedding (calls OpenAI API)
    embedding = self.embedding_generator.generate_embedding(text)
    # embedding = [0.023, -0.145, 0.087, ..., 0.012]  # 1536 floats
    
    # Store in graph node
    self.graph.nodes[url]['embeddings'] = embedding
    self.graph.nodes[url]['text'] = text
    self.graph.nodes[url]['label'] = "Page Title"

# Save entire graph to JSON
graph.json = {
    "nodes": [
        {
            "id": "https://example.com/401k",
            "label": "401k Overview",
            "text": "A 401k is a retirement account...",
            "embeddings": [0.023, -0.145, ..., 0.012]  # ← Pre-computed!
        },
        {
            "id": "https://example.com/roth",
            "label": "Roth vs Traditional",
            "text": "Roth contributions are after-tax...",
            "embeddings": [0.031, -0.152, ..., 0.019]  # ← Pre-computed!
        }
        // ... 50 more pages
    ],
    "edges": [...],
    "metadata": {...}
}
```

**Key point:** The JSON file already contains ALL the embeddings. No database needed!

## 🔎 Phase 2: Searching (In-Memory)

**When searching**, the entire JSON is loaded into RAM:

```python
class GraphSimilaritySearch:
    def load_graph(self):
        # 1. Load entire JSON file into memory
        with open('graph.json', 'r') as f:
            self.graph_data = json.load(f)  # ← Full JSON in RAM
        
        # 2. Extract nodes with embeddings into a Python list
        self.nodes_with_embeddings = []
        for node in self.graph_data['nodes']:
            if node.get('embeddings'):
                self.nodes_with_embeddings.append(node)
        
        # Now we have a list like:
        # [
        #   {'url': '...', 'text': '...', 'embeddings': [...]},
        #   {'url': '...', 'text': '...', 'embeddings': [...]},
        #   ...
        # ]
```

**When user queries:**

```python
def search_similar_nodes(self, query, top_k=5):
    # 1. Generate embedding for the query (NEW API call)
    query_embedding = self.embedding_generator.generate_embedding(query)
    # query_embedding = [0.028, -0.141, ..., 0.015]
    
    # 2. Compare query embedding with ALL stored embeddings
    results = []
    
    for node in self.nodes_with_embeddings:  # ← Loop through Python list
        node_embedding = node['embeddings']  # ← Already in memory!
        
        # 3. Calculate cosine similarity (pure math, no API)
        similarity = self.cosine_similarity(query_embedding, node_embedding)
        # similarity = 0.87 (for example)
        
        results.append({
            'url': node['url'],
            'label': node['label'],
            'similarity': similarity
        })
    
    # 4. Sort by similarity and return top k
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_k]


def cosine_similarity(self, vec1, vec2):
    # Pure numpy math - no API calls, no database queries
    vec1 = np.array(vec1)  # Convert list to numpy array
    vec2 = np.array(vec2)
    
    # Cosine similarity formula
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    return dot_product / (norm1 * norm2)
```

## 📊 Complete Example

Let’s trace a full search:

### JSON File (graph.json)

```json
{
  "nodes": [
    {
      "id": "https://myucretirement.com/401k",
      "label": "401k Contribution Limits",
      "text": "The 2024 401k contribution limit is $23,000...",
      "embeddings": [0.023, -0.145, 0.087, ..., 0.012]  // 1536 numbers
    },
    {
      "id": "https://myucretirement.com/roth",
      "label": "Roth vs Traditional",
      "text": "Roth contributions are made with after-tax dollars...",
      "embeddings": [0.031, -0.152, 0.091, ..., 0.019]
    },
    {
      "id": "https://myucretirement.com/catch-up",
      "label": "Catch-up Contributions",
      "text": "If you're 50 or older, you can contribute an extra $7,500...",
      "embeddings": [0.027, -0.148, 0.089, ..., 0.014]
    }
  ]
}
```

### Search Process

```python
# User query: "How much can I contribute to my 401k?"

# Step 1: Load JSON (happens once at startup)
graph_data = json.load(open('graph.json'))
# graph_data is now a Python dictionary in RAM

# Step 2: Generate query embedding (1 API call)
query_emb = openai.embeddings.create(
    model="text-embedding-3-small",
    input="How much can I contribute to my 401k?"
)
# query_emb = [0.025, -0.147, 0.088, ..., 0.013]

# Step 3: Compare with each node (NO API calls, pure math)
similarities = []

# Node 1: 401k Contribution Limits
sim1 = cosine_similarity(
    [0.025, -0.147, 0.088, ...],  # query
    [0.023, -0.145, 0.087, ...]   # node 1
)
# sim1 = 0.92  ← Very similar!

# Node 2: Roth vs Traditional
sim2 = cosine_similarity(
    [0.025, -0.147, 0.088, ...],  # query
    [0.031, -0.152, 0.091, ...]   # node 2
)
# sim2 = 0.65  ← Somewhat related

# Node 3: Catch-up Contributions
sim3 = cosine_similarity(
    [0.025, -0.147, 0.088, ...],  # query
    [0.027, -0.148, 0.089, ...]   # node 3
)
# sim3 = 0.89  ← Also very similar!

# Step 4: Sort and return
results = [
    {'label': '401k Contribution Limits', 'similarity': 0.92},
    {'label': 'Catch-up Contributions', 'similarity': 0.89},
    {'label': 'Roth vs Traditional', 'similarity': 0.65}
]
```

## ⚡ Performance Characteristics

### Why This Works Well

**For small-to-medium knowledge graphs (< 10,000 pages):**

```python
# Example: 100 pages
nodes_with_embeddings = 100

# Search operation:
for node in nodes_with_embeddings:  # 100 iterations
    similarity = cosine_similarity(...)  # ~0.001ms per comparison
    
# Total time: 100 × 0.001ms = 0.1ms (instant!)
```

**Memory usage:**

```python
# Each embedding: 1536 floats × 4 bytes = 6KB
# 100 pages: 100 × 6KB = 600KB
# Plus text content: ~100KB per page = 10MB
# Total: ~11MB in RAM (tiny!)
```

### When This Approach Breaks Down

**Large scale (> 10,000 pages):**

```python
# 100,000 pages
# Memory: 100,000 × 6KB = 600MB embeddings (still manageable)
# Search time: 100,000 × 0.001ms = 100ms (starting to feel slow)
```

**At that scale, you’d use:**

- **Vector databases** (Pinecone, Weaviate, Qdrant, ChromaDB)
- **Approximate Nearest Neighbors** (FAISS, HNSW)
- **Index structures** for O(log n) search instead of O(n)

## 🔄 Why JSON Instead of a Database?

### Advantages of JSON approach:

✅ **Simple deployment** - No database setup  
✅ **Portable** - Just copy one file  
✅ **Version control** - Can commit to git  
✅ **Fast for small graphs** - Everything in RAM  
✅ **Easy to inspect** - Human-readable

### When to switch to a database:

❌ > 10,000 pages (slow search)  
❌ Frequent updates (would need to regenerate entire JSON)  
❌ Multiple users (concurrent access issues)  
❌ Need complex queries (filtering, aggregation)

## 💡 Alternative: Vector Database

If you wanted to scale, you’d change the architecture:

```python
# Instead of JSON
from pinecone import Pinecone

# Store embeddings in vector DB
pc = Pinecone(api_key="...")
index = pc.Index("retirement-knowledge")

# Upload during crawl
index.upsert(vectors=[
    {
        "id": "page-1",
        "values": [0.023, -0.145, ...],  # embedding
        "metadata": {"url": "...", "title": "..."}
    }
])

# Search (fast even with millions of vectors)
results = index.query(
    vector=query_embedding,
    top_k=5
)
```

## 🎯 Summary

**The JSON approach works by:**

1. **Pre-computing** embeddings during crawl (expensive)
1. **Storing** them in JSON (one-time cost)
1. **Loading** entire JSON into RAM (fast)
1. **Comparing** query embedding with all stored embeddings (in-memory math)
1. **No database needed** for small-to-medium graphs!

**It’s fast because:**

- Embeddings are pre-computed (not generated on each search)
- Similarity is pure math (numpy vector operations)
- Everything is in RAM (no disk I/O)

**For this retirement planning use case (probably < 1000 pages), JSON is perfect!** 🎯​​​​​​​​​​​​​​​​
