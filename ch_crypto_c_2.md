```python

"""
Agentic RAG over website-parsed TXT files (H1/H2/Paragraph format)
------------------------------------------------------------------

What this gives you
- Ingestion from ./data/*.txt with simple H1/H2/paragraph parsing
- Token-aware chunking with configurable CHUNK_SIZE / CHUNK_OVERLAP
- Heading-aware context carried into each chunk's text + metadata
- Pluggable embeddings (OpenAI or Sentence-Transformers)
- FAISS vector index persisted to ./index
- Simple Agentic RAG query() with one-step refinement + verification
- Persona-aware answer style hooks

Quick start
-----------
1) Put your .txt files into ./data/
2) Set env vars if needed:
   - EMBEDDINGS_BACKEND=openai|hf
   - OPENAI_API_KEY=...
   - HF_MODEL=sentence-transformers/all-MiniLM-L6-v2  (if EMBEDDINGS_BACKEND=hf)
3) Adjust CHUNK_SIZE/CHUNK_OVERLAP in Config below (examples provided)
4) Run:  python agentic_rag_textfiles_pipeline.py --rebuild-index
5) Query: python agentic_rag_textfiles_pipeline.py --ask "What is staking?" --persona p2

Recommended chunk sizes (starting points)
- Short, dense text: CHUNK_SIZE=400 tokens, CHUNK_OVERLAP=60
- Mixed prose/slides: CHUNK_SIZE=600-800, CHUNK_OVERLAP=80-120
- Very long paragraphs: CHUNK_SIZE=1000, CHUNK_OVERLAP=120

Note: token counting uses tiktoken if available; otherwise approximates 4 chars ≈ 1 token.
"""
from __future__ import annotations
import os
import re
import json
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Optional deps imports guarded
try:
    import tiktoken  # for token counting (OpenAI-style)
except Exception:
    tiktoken = None

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception:
    SentenceTransformer = None
    np = None

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

# -------------------------
# Config
# -------------------------
@dataclass
class Config:
    data_dir: str = "./data"
    index_dir: str = "./index"
    collection: str = "web_txt_collection"
    chunk_size: int = 800            # tokens
    chunk_overlap: int = 100         # tokens
    embeddings_backend: str = os.getenv("EMBEDDINGS_BACKEND", "hf")  # "openai" or "hf"
    openai_model: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
    hf_model: str = os.getenv("HF_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    top_k: int = 8
    min_coverage_hits: int = 3       # below this -> trigger refine

CFG = Config()

# -------------------------
# Utilities: tokenization and chunking
# -------------------------

def _approx_tokens(s: str) -> int:
    if tiktoken:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(s))
    # Fallback rough heuristic: 4 chars ≈ 1 token
    return max(1, len(s) // 4)


def split_into_token_windows(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into token windows, preserving sentence-ish boundaries when possible."""
    # naive sentence-ish split
    parts = re.split(r"(?<=[.!?])\s+|\n\n+", text.strip())
    windows: List[str] = []
    cur: List[str] = []
    cur_tokens = 0
    for p in parts:
        t = _approx_tokens(p)
        if cur_tokens + t > chunk_size and cur:
            windows.append(" ".join(cur).strip())
            # build overlap from tail
            if overlap > 0 and windows[-1]:
                tail_tokens_needed = overlap
                tail = []
                # take sentences from the end until reach overlap
                for sent in reversed(re.split(r"\s+", windows[-1])):
                    tail.append(sent)
                    if _approx_tokens(" ".join(reversed(tail))) >= tail_tokens_needed:
                        break
                cur = [" ".join(reversed(tail))]
                cur_tokens = _approx_tokens(cur[0])
            else:
                cur, cur_tokens = [], 0
        cur.append(p)
        cur_tokens += t
    if cur:
        windows.append(" ".join(cur).strip())
    return windows

# -------------------------
# Parser for TXT format (H1/H2/Paragraph blocks)
# -------------------------
H1_RE = re.compile(r"^H1\s+(.*)$", re.IGNORECASE)
H2_RE = re.compile(r"^H2\s+(.*)$", re.IGNORECASE)
PARA_RE = re.compile(r"^PARAGRAPH\s+(.*)$", re.IGNORECASE)

@dataclass
class RawBlock:
    h1: Optional[str]
    h2: Optional[str]
    text: str
    order: int
    doc_id: str


def parse_txt_document(path: Path) -> List[RawBlock]:
    """Parses a TXT with lines like 'H1 Title', 'H2 Sub-title', 'PARAGRAPH ...'"""
    h1 = None
    h2 = None
    blocks: List[RawBlock] = []
    buf: List[str] = []
    order = 0

    def flush_buf():
        nonlocal buf, order
        if buf:
            text = "\n".join(buf).strip()
            if text:
                blocks.append(RawBlock(h1=h1, h2=h2, text=text, order=order, doc_id=path.stem))
                order += 1
            buf = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                # blank line = paragraph separator -> keep collecting; flush later
                buf.append("")
                continue
            m1 = H1_RE.match(line)
            m2 = H2_RE.match(line)
            mp = PARA_RE.match(line)
            if m1:
                flush_buf()
                h1 = m1.group(1).strip()
                h2 = None  # reset h2 when new h1
            elif m2:
                flush_buf()
                h2 = m2.group(1).strip()
            elif mp:
                buf.append(mp.group(1).strip())
            else:
                # If the line doesn't start with H1/H2/PARAGRAPH, treat as continuation
                buf.append(line.strip())
    flush_buf()
    return blocks

# -------------------------
# Heading-aware chunk construction
# -------------------------
@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, Any]


def build_chunks(blocks: List[RawBlock], cfg: Config = CFG) -> List[Chunk]:
    chunks: List[Chunk] = []
    for b in blocks:
        header = []
        if b.h1:
            header.append(f"H1: {b.h1}")
        if b.h2:
            header.append(f"H2: {b.h2}")
        header_text = " | ".join(header)
        full = (header_text + "\n\n" if header_text else "") + b.text
        windows = split_into_token_windows(full, cfg.chunk_size, cfg.chunk_overlap)
        for i, w in enumerate(windows):
            cid = f"{b.doc_id}_{b.order}_{i}"
            meta = {
                "doc_id": b.doc_id,
                "order": b.order,
                "window_index": i,
                "h1": b.h1,
                "h2": b.h2,
                "chunk_tokens": _approx_tokens(w),
            }
            chunks.append(Chunk(id=cid, text=w, metadata=meta))
    return chunks

# -------------------------
# Embeddings backends
# -------------------------
class Embeddings:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.backend = cfg.embeddings_backend
        self.model = None
        if self.backend == "hf":
            if SentenceTransformer is None:
                raise ImportError("Install sentence-transformers for hf backend")
            self.model = SentenceTransformer(cfg.hf_model)
        elif self.backend == "openai":
            import openai  # lazy import
            self.openai = openai
            self.openai.api_key = os.getenv("OPENAI_API_KEY")
            if not self.openai.api_key:
                raise RuntimeError("OPENAI_API_KEY not set")
        else:
            raise ValueError("Unknown EMBEDDINGS_BACKEND; use 'hf' or 'openai'")

    def encode(self, texts: List[str]) -> Any:
        if self.backend == "hf":
            return self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        else:
            # batched OpenAI embeddings
            from openai import OpenAI
            client = OpenAI()
            vectors = []
            for t in texts:
                resp = client.embeddings.create(model=self.cfg.openai_model, input=t)
                vectors.append(resp.data[0].embedding)
            return np.array(vectors, dtype="float32")

# -------------------------
# FAISS indexer
# -------------------------
class FaissStore:
    def __init__(self, dim: int, index_dir: str):
        if faiss is None:
            raise ImportError("Install faiss-cpu for vector indexing")
        self.index_path = Path(index_dir) / "faiss.index"
        self.meta_path = Path(index_dir) / "meta.jsonl"
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # use normalized vectors
        self.metadata: List[Dict[str, Any]] = []

    def add(self, vecs, metas: List[Dict[str, Any]]):
        # ensure numpy float32
        arr = np.asarray(vecs, dtype="float32")
        # L2-normalize to make IP ~ cosine
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        arr = arr / norms
        self.index.add(arr)
        self.metadata.extend(metas)

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with self.meta_path.open("w", encoding="utf-8") as f:
            for m in self.metadata:
                f.write(json.dumps(m) + "\n")

    def load(self):
        if not self.index_path.exists():
            raise FileNotFoundError("FAISS index not found; build it first")
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = []
        with self.meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))

    def search(self, vec, top_k: int = 8) -> List[Dict[str, Any]]:
        v = np.asarray(vec, dtype="float32").reshape(1, -1)
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        D, I = self.index.search(v, top_k)
        out = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx == -1:
                continue
            m = self.metadata[idx].copy()
            m["score"] = float(score)
            out.append(m)
        return out

# -------------------------
# Ingestion Pipeline
# -------------------------

def ingest(cfg: Config = CFG) -> Tuple[FaissStore, Embeddings]:
    data_dir = Path(cfg.data_dir)
    files = sorted(list(data_dir.glob("*.txt")))
    assert files, f"No .txt files found in {data_dir}"

    # parse + chunk
    all_chunks: List[Chunk] = []
    for p in files:
        blocks = parse_txt_document(p)
        chunks = build_chunks(blocks, cfg)
        all_chunks.extend(chunks)

    print(f"Parsed {len(files)} files → {len(all_chunks)} chunks")

    # embeddings
    emb = Embeddings(cfg)
    texts = [c.text for c in all_chunks]
    vecs = emb.encode(texts)

    # store
    store = FaissStore(dim=vecs.shape[1], index_dir=cfg.index_dir)
    metas = [
        {
            **c.metadata,
            "id": c.id,
            "text": c.text,
        }
        for c in all_chunks
    ]
    store.add(vecs, metas)
    store.save()
    print(f"Index saved to {cfg.index_dir}")
    return store, emb

# -------------------------
# Retrieval + (simple) rerank
# -------------------------

def retrieve(query: str, store: FaissStore, emb: Embeddings, cfg: Config = CFG) -> List[Dict[str, Any]]:
    qv = emb.encode([query])[0]
    hits = store.search(qv, top_k=max(cfg.top_k * 3, 24))  # widen, then rerank
    # naive rerank by score (already cosine IP); could add keyword BM25 here
    hits = sorted(hits, key=lambda x: x["score"], reverse=True)[: cfg.top_k]
    return hits

# -------------------------
# LLM Compose + simple Agentic refine/verify hooks
# -------------------------
PERSONAS = {
    "p1": {
        "style": "concise, slightly technical; examples welcome",
        "jargon": "moderate",
        "depth": "medium+",
    },
    "p2": {
        "style": "practical, connect to traditional finance",
        "jargon": "low",
        "depth": "medium",
    },
    "p3": {
        "style": "plain language, patient, short sections",
        "jargon": "very low",
        "depth": "shallow→medium",
    },
}


def _context_from_hits(hits: List[Dict[str, Any]]) -> str:
    parts = []
    for h in hits:
        tag = f"[{h.get('doc_id','?')} • H1:{h.get('h1')} • H2:{h.get('h2')} • order:{h.get('order')}]"
        parts.append(f"{tag}\n{h['text']}")
    return "\n\n".join(parts)


def compose_answer(query: str, persona: str, hits: List[Dict[str, Any]]) -> str:
    persona_desc = PERSONAS.get(persona, PERSONAS["p2"])  # default p2
    context = _context_from_hits(hits)
    system = (
        "You are a crypto explainer that must only use the provided context. "
        "Respond with: TL;DR (2–4 lines) → Main Answer (persona style) → Key Takeaways (3 bullets) → Sources. "
        "Add 'Not financial advice.'"
    )
    instruction = (
        f"Persona style: {json.dumps(persona_desc)}\n"
        f"User question: {query}\n\n"
        f"CONTEXT:\n{context}\n\nAnswer:"
    )
    # Here we leave a placeholder LLM call; integrate your LLM of choice.
    # For demonstration, we'll just return a stub with sources list.
    sources = []
    for h in hits:
        src = f"{h.get('doc_id','?')} • H1:{h.get('h1')} • H2:{h.get('h2')} (chunk {h.get('id')})"
        sources.append(src)

    stub = (
        "TL;DR: (stub) This is where the LLM's concise summary goes.\n\n"
        "Main Answer:\n(stub) Replace with LLM output conditioned on persona and the context above.\n\n"
        "Key Takeaways:\n- (stub) point 1\n- (stub) point 2\n- (stub) point 3\n\n"
        f"Sources:\n- " + "\n- ".join(sources) + "\n\n— Not financial advice."
    )
    return stub


def refine_query_if_needed(query: str, hits: List[Dict[str, Any]], cfg: Config = CFG) -> Optional[str]:
    if len(hits) >= cfg.min_coverage_hits:
        return None
    # simple heuristic refinement: add top H1/H2 terms as expansions
    heads = []
    for h in hits:
        if h.get("h1"): heads.append(h["h1"]) 
        if h.get("h2"): heads.append(h["h2"]) 
    heads = list({w for w in heads if w})[:3]
    if heads:
        return query + " " + " ".join(f"\"{w}\"" for w in heads)
    return None


def verify_answer_stub(answer: str, hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Placeholder verification: ensure every section exists; in prod use an LLM judge."""
    ok = all(section in answer for section in ["TL;DR", "Main Answer", "Key Takeaways", "Sources"])
    return {"ok": ok, "issues": [] if ok else ["missing_sections"]}

# -------------------------
# Public API
# -------------------------
class RAGService:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        # lazy load
        self.emb: Optional[Embeddings] = None
        self.store: Optional[FaissStore] = None

    def ensure_loaded(self):
        if self.emb is None:
            self.emb = Embeddings(self.cfg)
        if self.store is None:
            self.store = FaissStore(dim=self._dim_hint(), index_dir=self.cfg.index_dir)
            self.store.load()

    def _dim_hint(self) -> int:
        # quick heuristic: 384 for MiniLM, 1024+ for larger models; used only before real load
        if self.cfg.embeddings_backend == "hf":
            if "MiniLM" in self.cfg.hf_model:
                return 384
            return 768
        else:
            return 3072  # text-embedding-3-large

    def query(self, q: str, persona: str = "p2") -> str:
        self.ensure_loaded()
        hits = retrieve(q, self.store, self.emb, self.cfg)
        rq = refine_query_if_needed(q, hits, self.cfg)
        if rq:
            hits2 = retrieve(rq, self.store, self.emb, self.cfg)
            if len(hits2) > len(hits):
                hits = hits2
        ans = compose_answer(q, persona, hits)
        verdict = verify_answer_stub(ans, hits)
        if not verdict["ok"]:
            ans += "\n\n[Note] Automated verification flagged format issues; consider retrying."
        return ans

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-index", action="store_true", help="Parse data and rebuild FAISS index")
    ap.add_argument("--ask", type=str, default=None, help="Query the RAG system")
    ap.add_argument("--persona", type=str, default="p2", choices=["p1","p2","p3"]) 
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--chunk-overlap", type=int, default=None)
    args = ap.parse_args()

    if args.chunk_size is not None:
        CFG.chunk_size = args.chunk_size
    if args.chunk_overlap is not None:
        CFG.chunk_overlap = args.chunk_overlap

    if args.rebuild_index:
        ingest(CFG)

    if args.ask:
        svc = RAGService(CFG)
        print(svc.query(args.ask, persona=args.persona))


# --- fastapi_app.py ---
"""
FastAPI wrapper for the Agentic RAG TXT pipeline
- Endpoints:
  - GET  /health
  - POST /index                -> (re)build FAISS index from ./data/*.txt
  - POST /query                -> run Agentic RAG with persona + OpenAI LLM compose

Run locally:
  uvicorn fastapi_app:app --reload --port 8000

Env vars you may set:
  EMBEDDINGS_BACKEND=hf|openai
  OPENAI_API_KEY=sk-...
  HF_MODEL=sentence-transformers/all-MiniLM-L6-v2
"""
from __future__ import annotations
import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import pipeline pieces from the sibling module
import agentic_rag_textfiles_pipeline as rag

# ---------- App & CORS ----------
app = FastAPI(title="Agentic RAG over TXT (FastAPI)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class IndexResponse(BaseModel):
    ok: bool
    chunks_indexed: int

class QueryRequest(BaseModel):
    query: str = Field(..., description="User query")
    persona: str = Field("p2", description="p1|p2|p3")
    top_k: Optional[int] = Field(None, description="Override k (default from config)")
    chunk_size: Optional[int] = Field(None, description="Override chunk size (tokens)")
    chunk_overlap: Optional[int] = Field(None, description="Override chunk overlap (tokens)")

class QueryResponse(BaseModel):
    answer: str
    hits: List[Dict[str, Any]]

# ---------- OpenAI LLM (dummy compose) ----------
USE_OPENAI = True
try:
    from openai import OpenAI
    _client = OpenAI()
    # Will raise if key not set
    _ = os.environ.get("OPENAI_API_KEY") or _client.api_key
except Exception:
    USE_OPENAI = False
    _client = None


def _compose_with_openai(query: str, persona: str, hits: List[Dict[str, Any]]) -> str:
    """Compose final answer via OpenAI chat if key exists; else fall back to stub composer."""
    if not USE_OPENAI:
        return rag.compose_answer(query, persona, hits)

    persona_desc = rag.PERSONAS.get(persona, rag.PERSONAS["p2"])
    context = rag._context_from_hits(hits)

    system = (
        "You are a crypto explainer that must only use the provided context. "
        "Respond with: TL;DR (2–4 lines) → Main Answer (persona style) → Key Takeaways (3 bullets) → Sources. "
        "Add 'Not financial advice.'"
    )
    user = (
        f"Persona style: {persona_desc}\n"
        f"User question: {query}\n\n"
        f"CONTEXT:\n{context}\n\nAnswer:"
    )

    try:
        resp = _client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
            max_tokens=700,
        )
        return resp.choices[0].message.content
    except Exception as e:
        # Fallback to stub if OpenAI fails
        return rag.compose_answer(query, persona, hits) + f"\n\n[LLM fallback due to error: {e}]"

# ---------- Lifecycle ----------
_svc: Optional[rag.RAGService] = None

@app.on_event("startup")
def _startup():
    global _svc
    _svc = rag.RAGService(rag.CFG)
    # Attempt to load existing index; ignore if missing
    try:
        _svc.ensure_loaded()
    except Exception:
        pass

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"ok": True, "embeddings_backend": rag.CFG.embeddings_backend}

@app.post("/index", response_model=IndexResponse)
def build_index(chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
    if chunk_size is not None:
        rag.CFG.chunk_size = chunk_size
    if chunk_overlap is not None:
        rag.CFG.chunk_overlap = chunk_overlap

    store, emb = rag.ingest(rag.CFG)
    # After rebuild, refresh the in-memory service
    global _svc
    _svc = rag.RAGService(rag.CFG)
    _svc.ensure_loaded()

    # Count from stored metadata
    try:
        count = len(_svc.store.metadata)  # type: ignore
    except Exception:
        count = 0
    return IndexResponse(ok=True, chunks_indexed=count)

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if _svc is None:
        raise HTTPException(500, "Service not initialized")

    # Allow per-request overrides
    prev = (rag.CFG.chunk_size, rag.CFG.chunk_overlap, rag.CFG.top_k)
    if req.chunk_size is not None:
        rag.CFG.chunk_size = req.chunk_size
    if req.chunk_overlap is not None:
        rag.CFG.chunk_overlap = req.chunk_overlap
    if req.top_k is not None:
        rag.CFG.top_k = req.top_k

    # Retrieve (with one-step refine) using the pipeline's helpers
    _svc.ensure_loaded()
    hits = rag.retrieve(req.query, _svc.store, _svc.emb, rag.CFG)  # type: ignore
    rq = rag.refine_query_if_needed(req.query, hits, rag.CFG)
    if rq:
        hits2 = rag.retrieve(rq, _svc.store, _svc.emb, rag.CFG)  # type: ignore
        if len(hits2) > len(hits):
            hits = hits2

    # Compose with OpenAI (or stub fallback)
    answer = _compose_with_openai(req.query, req.persona, hits)

    # Restore config knobs after request
    rag.CFG.chunk_size, rag.CFG.chunk_overlap, rag.CFG.top_k = prev

    # Trim hit text in response for payload size (keep metadata + first 400 chars)
    trimmed_hits: List[Dict[str, Any]] = []
    for h in hits:
        hh = dict(h)
        text = hh.pop("text", "")
        hh["preview"] = text[:400] + ("…" if len(text) > 400 else "")
        trimmed_hits.append(hh)

    return QueryResponse(answer=answer, hits=trimmed_hits)

# ---------- Notes ----------
# curl examples:
# 1) Rebuild index with custom chunking
#    curl -X POST "http://localhost:8000/index?chunk_size=800&chunk_overlap=100"
# 2) Ask a question (persona p3)
#    curl -X POST http://localhost:8000/query \
#         -H 'Content-Type: application/json' \
#         -d '{"query":"Explain staking risks","persona":"p3"}'


```




All set! I added a FastAPI server file to the canvas called fastapi_app.py that:
	•	exposes /index to (re)build your FAISS index from ./data/*.txt (with optional chunk_size and chunk_overlap overrides),
	•	exposes /query to run the Agentic-RAG flow,
	•	uses a dummy OpenAI chat call for composing the final answer (falls back to the stub if OPENAI_API_KEY isn’t set),
	•	returns trimmed hits (metadata + first 400 chars) for easy UI display.

How to run:
	1.	install deps (example):

pip install fastapi uvicorn pydantic faiss-cpu sentence-transformers numpy openai tiktoken

	2.	start the API:

uvicorn fastapi_app:app --reload --port 8000

	3.	build the index:

curl -X POST "http://localhost:8000/index?chunk_size=800&chunk_overlap=100"

	4.	ask a question (persona p3):

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Explain staking risks","persona":"p3"}'

Want me to add a simple BM25 keyword boost (e.g., with Whoosh or Elasticsearch/OpenSearch) and a cross-encoder reranker next?