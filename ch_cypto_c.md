When Agentic RAG actually helps (vs. vanilla RAG)

Great fit
	•	Multi-hop questions: “How does ETH fee burn relate to 2024 L2 fees? cite slides.”
	•	Ambiguous or sparse queries: agent can reformulate and expand.
	•	Persona tailoring: agent plans teaching steps (analogy vs. technical).
	•	Verification/citations: agent runs post-hoc checks before answering.

Avoid / constrain
	•	Ultra-simple lookups (“What is halving?”) → agent detours add latency/cost.
	•	Noisy corpora without slide/page structure → tool use won’t fix bad data.
	•	Hard realtime news (outside your corpus) → keep agent from webbrowsing unless you intend to.

Agentic RAG design (graph of skills)

Nodes (skills)
	1.	Persona Router → set style/depth.
	2.	Planner → break query into sub-goals (e.g., define → compare → risks → cite).
	3.	Retriever (hybrid+rerank) → k=50 union → cross-encoder → top8.
	4.	Query Refiner (optional loop) → rewrite if coverage is weak.
	5.	Composer → draft answer using persona style + citations.
	6.	Verifier → claim-to-evidence check; hallucination guard.
	7.	Citations Builder → normalize Deck • Slide and add “as-of” date.
	8.	Safety/Compliance → disclaimers, remove advicey phrasing.
	9.	Finalizer → TL;DR, Key Takeaways, Sources.

State
```python
class State(TypedDict):
    persona: Literal["p1","p2","p3"]
    query: str
    subgoals: list[str]
    retrieved: list[dict]     # {text, doc_id, slide_no, date, score}
    draft: str
    citations: list[dict]     # {doc_id, slide_no, date}
    safety_flags: list[str]
    needs_refine: bool
```

Prompts (core snippets)

Planner (few-shot)

Task: Turn the user question and persona into 2–4 subgoals. Prefer “Define → Explain → Risks → Cite”. Return JSON list only. Keep persona depth/jargon in mind.

Composer (system)

You are a crypto explainer that must only use the provided context. Tailor tone and depth using persona:\n{{persona_json}}. Respond with: TL;DR (2–4 lines) → Main Answer (persona style) → Key Takeaways (3 bullets) → Sources (Deck • Slide, with dates). Add “Not financial advice.” If evidence is thin, say so.

Verifier (judge)

Check if each key claim is supported by cited chunks. Flag any unsupported sentence, stale date (>18 months), or numeric inconsistency. Return JSON: {ok: bool, issues: [..], missing_citations: [..]}.

Skeleton code (LangGraph)

```python
# pyproject: langgraph, langchain-openai, qdrant-client (or pgvector), fastapi

from typing import TypedDict, Literal, List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient

### ---- STATE ----
class State(TypedDict):
    persona: Literal["p1","p2","p3"]
    query: str
    subgoals: List[str]
    retrieved: List[Dict[str, Any]]
    draft: str
    citations: List[Dict[str, Any]]
    safety_flags: List[str]
    needs_refine: bool

### ---- MODELS & STORES ----
llm = ChatOpenAI(model="gpt-4o-mini")
judge = ChatOpenAI(model="gpt-4o-mini")   # lean judge; can swap in larger
emb = OpenAIEmbeddings(model="text-embedding-3-large")
qdrant = QdrantClient(url="http://localhost:6333")

### ---- HELPERS ----
def hybrid_retrieve(query: str, k: int = 8) -> List[Dict[str, Any]]:
    # 1) vector search
    vec = emb.embed_query(query)
    vhits = qdrant.search(collection_name="crypto_docs", query_vector=vec, limit=40)
    # 2) keyword/BM25 (optional: via OpenSearch) → khits
    # 3) union + cross-encoder reranker (pseudo)
    # return top-8 dicts with {text, doc_id, slide_no, date, score}
    return postprocess_hits(vhits)[:k]

def build_context(chunks: List[Dict[str, Any]]) -> str:
    # Concatenate with minimal extra tokens; include slide/page headers
    parts = []
    for c in chunks:
        parts.append(f"[{c['doc_id']} • Slide {c['slide_no']} • {c['date']}]\n{c['text']}")
    return "\n\n".join(parts)

def citation_scan(answer: str) -> List[Dict[str, Any]]:
    # naive: parse (Doc • Slide N) patterns → normalize via store metadata
    return extract_citations(answer)

### ---- NODES ----
def persona_router(state: State) -> State:
    # assume persona already set in config; default to p2
    state.setdefault("persona", "p2")
    return state

def planner(state: State) -> State:
    prompt = f"""User question: {state['query']}
Persona: {state['persona']}
Return 2-4 subgoals as a JSON list."""
    goals = llm.invoke([{"role": "user", "content": prompt}]).content
    state["subgoals"] = parse_json_list(goals)
    return state

def retriever(state: State) -> State:
    # join original query + subgoals keywords
    expanded = state["query"] + " | " + " | ".join(state.get("subgoals", []))
    hits = hybrid_retrieve(expanded, k=8)
    state["retrieved"] = hits
    # if few or low score, trigger refine
    state["needs_refine"] = len(hits) < 3
    return state

def refiner(state: State) -> State:
    if not state["needs_refine"]:
        return state
    q = state["query"]
    hint = "Rewrite to be unambiguous and include key entities; 12-18 tokens."
    newq = llm.invoke([{"role":"user","content": f"{hint}\n\nOriginal: {q}"}]).content
    hits = hybrid_retrieve(newq, k=8)
    state["retrieved"] = hits or state["retrieved"]
    state["needs_refine"] = False
    return state

def composer(state: State) -> State:
    persona_json = get_persona_json(state["persona"])
    context = build_context(state["retrieved"])
    prompt = f"""System: You must only use the context. Tailor to persona.
Persona: {persona_json}

Context:
{context}

Write: TL;DR → Main Answer → 3 Key Takeaways → Sources (Deck • Slide). Add 'Not financial advice.'"""
    out = llm.invoke([{"role":"user","content": prompt}]).content
    state["draft"] = out
    state["citations"] = citation_scan(out)
    return state

def verifier(state: State) -> State:
    context = build_context(state["retrieved"])
    ask = f"""Given the CONTEXT and the DRAFT, check support for each claim.
Return JSON: {{ "ok": bool, "issues": [..], "missing_citations": [..], "stale": [..] }}.
CONTEXT:\n{context}\n\nDRAFT:\n{state['draft']}"""
    res = judge.invoke([{"role":"user","content": ask}]).content
    verdict = parse_json(res)
    if not verdict.get("ok"):
        # soft fix: append "What we know" section + inject missing citations where possible
        state["draft"] = patch_answer(state["draft"], verdict, state["retrieved"])
        state["safety_flags"] = verdict.get("issues", [])
    return state

def finalizer(state: State) -> State:
    # ensure as-of newest date & disclaimer
    newest = max([c["date"] for c in state["retrieved"] if c.get("date")], default=None)
    footer = f"\n\n— Not financial advice. Sources as of {newest}."
    state["draft"] = state["draft"].rstrip() + footer
    return state

### ---- GRAPH ----
graph = StateGraph(State)
graph.add_node("persona", persona_router)
graph.add_node("plan", planner)
graph.add_node("retrieve", retriever)
graph.add_node("refine", refiner)
graph.add_node("compose", composer)
graph.add_node("verify", verifier)
graph.add_node("final", finalizer)

graph.set_entry_point("persona")
graph.add_edge("persona", "plan")
graph.add_edge("plan", "retrieve")
graph.add_edge("retrieve", "refine")
graph.add_edge("refine", "compose")
graph.add_edge("compose", "verify")
graph.add_edge("verify", "final")
graph.add_edge("final", END)

app = graph.compile(checkpointer=MemorySaver())

# ---- RUNTIME EXAMPLE ----
def answer(query: str, persona: str = "p2"):
    state: State = {"query": query, "persona": persona,
                    "subgoals": [], "retrieved": [], "draft": "",
                    "citations": [], "safety_flags": [], "needs_refine": False}
    result = app.invoke(state, config={"configurable": {"thread_id": "demo"}})
    return result["draft"]
```

Minimal alternative (no LangGraph, plain Python)

```python
def agentic_rag(query, persona):
    state = {"query": query, "persona": persona}
    state["subgoals"] = plan(query, persona)
    hits = hybrid_retrieve(query + " | " + " | ".join(state["subgoals"]))
    if len(hits) < 3:
        query2 = refine_query(query)
        hits2 = hybrid_retrieve(query2)
        if hits2: hits = hits2
    draft = compose(hits, persona, query, state["subgoals"])
    verdict = verify(draft, hits)
    if not verdict["ok"]:
        draft = patch_answer(draft, verdict, hits)
    return add_footer(draft, hits)
```

Measuring “applicability” (is Agentic worth it?)

Design an A/B test
	•	A (baseline RAG): retrieve → compose → return.
	•	B (Agentic): plan → retrieve → refine loop (≤1) → compose → verify → return.

Metrics
	•	Faithfulness (LLM-judge rubric 0–5)
	•	Citation accuracy (slide exact-match, page overlap)
	•	Persona satisfaction (thumbs-up rate per persona)
	•	Coverage rate (% answers with ≥2 independent sources)
	•	Latency & cost deltas (p50/p95, tokens)
	•	Escapes to “insufficient evidence” (should go up slightly but correlate with fewer hallucinations)

Acceptance
	•	+0.5 ↑ faithfulness & +10–15% ↑ citation accuracy at ≤35% cost ↑ and ≤1.0s latency ↑ (p50).
	•	If gains <5% or p95 latency hurts UX, restrict Agentic path to only multi-hop intents.

Practical knobs (keep it efficient)
	•	Refine loop cap: 1 iteration max.
	•	Top-k: 50 union → rerank to 8.
	•	Verifier: smaller model; only full check if answer > 180 tokens or multiple numeric claims.
	•	Caching: memoize subgoal plans for common queries; cache rerank results by n-grams.
	•	Safety: block wallet/private-key operational steps; always “Not financial advice.”

⸻

