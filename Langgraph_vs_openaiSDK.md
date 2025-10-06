awesome — here’s a clean, production-ready Part-1 ingestion pack with four files:
	•	load_files.py → walks your data/ tree, reads .txt/.md (and optionally .pdf if you want), emits RawDoc
	•	normalize.py → Unicode/whitespace normalization, markdown → plain-text, heading extraction, emits NormalizedDoc
	•	chunk.py → heading-aware, token-aware chunking (tiktoken fallback included), emits Chunk
	•	embed.py → OpenAI embeddings via LangChain (async + sync), emits (chunk, embedding) pairs

All files use strict type hints and small, well-documented APIs so you can drop them in.

⸻
```
load_files.py

# src/ingestion/load_files.py
from __future__ import annotations
from pathlib import Path
from typing import Iterator, Literal, TypedDict, Dict, Any, Optional
import hashlib
import datetime as dt

SourceType = Literal["txt", "md", "pdf", "transcript", "webinar", "audio", "other"]

class RawDoc(TypedDict):
    doc_id: str                 # stable id (hash-based)
    source_type: SourceType
    source_path: str            # absolute or relative path
    created_at: str             # ISO8601
    updated_at: str             # ISO8601
    text: str                   # raw text payload
    metadata: Dict[str, Any]    # free-form: url, heading_path, page_or_time, etc.

def _iso(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).isoformat(timespec="seconds")

def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "ignore")).hexdigest()[:16]

def _guess_source_type(p: Path) -> SourceType:
    ext = p.suffix.lower()
    if ext in {".txt"}: return "txt"
    if ext in {".md", ".markdown"}: return "md"
    if ext in {".pdf"}: return "pdf"
    # folder hints (optional)
    hint = p.as_posix().lower()
    if "transcript" in hint: return "transcript"
    if "webinar" in hint: return "webinar"
    if "audio" in hint: return "audio"
    return "other"

def _read_text(p: Path) -> Optional[str]:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        # last-resort binary decode
        return p.read_bytes().decode("utf-8", "ignore")

def walk_data_dir(root: str | Path, include_pdf: bool = False) -> Iterator[RawDoc]:
    """
    Recursively yields RawDoc from a content directory.
    - Supports .txt/.md by default; .pdf if include_pdf=True (you'll convert to text later).
    - You may pre-convert PDFs to .txt for homogeneity; keep original path in metadata.
    """
    root = Path(root)
    assert root.exists(), f"Path not found: {root}"

    patterns = ["**/*.txt", "**/*.md"]
    if include_pdf:
        patterns.append("**/*.pdf")

    for pat in patterns:
        for p in root.glob(pat):
            if p.is_dir():  # skip dirs
                continue
            st = p.stat()
            src_type = _guess_source_type(p)
            if p.suffix.lower() == ".pdf" and not include_pdf:
                continue

            if p.suffix.lower() in {".txt", ".md"}:
                text = _read_text(p) or ""
            else:
                # Minimal PDF placeholder; prefer pre-converted .txt
                # (You can plug in PyPDF here if desired.)
                text = ""

            doc_id = f"{p.stem}-{_hash(str(p.resolve()))}"
            yield RawDoc(
                doc_id=doc_id,
                source_type=src_type,
                source_path=str(p),
                created_at=_iso(st.st_ctime),
                updated_at=_iso(st.st_mtime),
                text=text,
                metadata={
                    "heading_path": [],
                    "page_or_time": None,
                    "original_ext": p.suffix.lower(),
                },
            )

```
⸻
```
normalize.py

# src/ingestion/normalize.py
from __future__ import annotations
import re
import unicodedata
from typing import List, Dict, Any, TypedDict
from .load_files import RawDoc

class Heading(TypedDict):
    level: int
    title: str
    start_char: int

class NormalizedDoc(TypedDict):
    doc_id: str
    text: str                 # clean, plain text
    headings: List[Heading]   # detected markdown-ish headings (optional)
    metadata: Dict[str, Any]  # carried forward + normalization notes

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

def _strip_markdown(md: str) -> tuple[str, List[Heading]]:
    """
    Converts markdown to plain-ish text while capturing headings.
    - Strips code fences, inline code, images; keeps link text.
    """
    s = md

    # Remove fenced code blocks
    s = re.sub(r"```.*?```", "", s, flags=re.DOTALL)

    # Inline code → text
    s = re.sub(r"`([^`]+)`", r"\1", s)

    # Images ![alt](url) → alt
    s = re.sub(r"!$begin:math:display$([^$end:math:display$]*)\]$begin:math:text$[^)]+$end:math:text$", r"\1", s)

    # Links [text](url) → text
    s = re.sub(r"$begin:math:display$([^$end:math:display$]+)\]$begin:math:text$[^)]+$end:math:text$", r"\1", s)

    # Extract headings
    headings: List[Heading] = []
    for m in _HEADING_RE.finditer(s):
        level = len(m.group(1))
        title = m.group(2).strip()
        headings.append({"level": level, "title": title, "start_char": m.start()})

    # Remove leading hashes from headings in body
    s = re.sub(r"^#{1,6}\s+", "", s, flags=re.MULTILINE)

    return s, headings

def _normalize_text(t: str) -> str:
    # Unicode NFC
    t = unicodedata.normalize("NFC", t)
    # Replace weird NBSPs / zero-width / smart quotes minimal
    t = t.replace("\u00A0", " ").replace("\u200B", "")
    # Normalize line endings
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse trailing spaces
    t = re.sub(r"[ \t]+$", "", t, flags=re.MULTILINE)
    # Collapse 3+ newlines → 2
    t = re.sub(r"\n{3,}", "\n\n", t)
    # Trim
    return t.strip()

def normalize_rawdoc(doc: RawDoc) -> NormalizedDoc:
    """
    Converts RawDoc into NormalizedDoc:
      - Unicode + whitespace normalization
      - Markdown → text (if needed)
      - Captures headings (level, title, start_char)
    """
    text = doc["text"] or ""
    headings: List[Heading] = []

    if doc["source_type"] == "md":
        text, headings = _strip_markdown(text)

    # (Optional) transcripts timestamp keeper — keep as-is but normalized
    # Example timestamps like [00:01:23] will be preserved.

    norm = _normalize_text(text)

    metadata = dict(doc["metadata"])
    metadata.update({
        "normalized": True,
        "source_type": doc["source_type"],
        "source_path": doc["source_path"],
        "created_at": doc["created_at"],
        "updated_at": doc["updated_at"],
    })

    return NormalizedDoc(
        doc_id=doc["doc_id"],
        text=norm,
        headings=headings,
        metadata=metadata,
    )
```

⸻
```
chunk.py

# src/ingestion/chunk.py
from __future__ import annotations
from typing import Iterator, List, Dict, Any, TypedDict, Optional
import re

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENC = None

from .normalize import NormalizedDoc, Heading

class Chunk(TypedDict):
    chunk_id: str
    doc_id: str
    ordinal: int
    text: str
    heading_path: List[str]
    start_char: int
    end_char: int
    page_or_time: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]

def _token_len(s: str) -> int:
    if _ENC:
        try:
            return len(_ENC.encode(s))
        except Exception:
            pass
    # Fallback heuristic ~ 4 chars/token
    return max(1, len(s) // 4)

def _pack_by_tokens(paras: List[str], max_tokens: int, overlap_tokens: int) -> List[str]:
    """Greedy packer over paragraphs with token-based length target + overlap."""
    out: List[str] = []
    cur: List[str] = []
    cur_tokens = 0
    for p in paras:
        ptok = _token_len(p)
        if cur_tokens + ptok <= max_tokens or not cur:
            cur.append(p)
            cur_tokens += ptok
        else:
            out.append("\n\n".join(cur))
            # overlap: take tail paragraphs until overlap_tokens reached
            tail: List[str] = []
            tail_tokens = 0
            for para in reversed(cur):
                t = _token_len(para)
                if tail_tokens + t > overlap_tokens: break
                tail_tokens += t
                tail.insert(0, para)
            cur = tail + [p]
            cur_tokens = sum(_token_len(x) for x in cur)
    if cur:
        out.append("\n\n".join(cur))
    return out

def _split_into_sections(text: str, headings: List[Heading]) -> List[Dict[str, Any]]:
    """
    Splits by detected headings (level 1-3) to keep local context; falls back to full text.
    Returns list of {'title': str, 'start': int, 'end': int, 'body': str, 'path': [..]}
    """
    if not headings:
        return [{"title": "", "start": 0, "end": len(text), "body": text, "path": []}]

    # Use only top-ish headings to segment
    anchors = [{"idx": h["start_char"], "title": h["title"], "level": h["level"]} for h in headings]
    anchors.sort(key=lambda x: x["idx"])
    sections = []
    for i, a in enumerate(anchors):
        start = a["idx"]
        end = anchors[i + 1]["idx"] if i + 1 < len(anchors) else len(text)
        body = text[start:end].strip()
        # Build simple path: all previous titles with lower level
        path = [a["title"]]
        sections.append({"title": a["title"], "start": start, "end": end, "body": body, "path": path})
    return sections

def chunk_document(
    doc: NormalizedDoc,
    max_tokens: int = 350,
    overlap_tokens: int = 60,
) -> Iterator[Chunk]:
    """
    Inputs:
      - doc: NormalizedDoc
      - max_tokens: desired target per chunk (token-aware)
      - overlap_tokens: tail overlap between consecutive chunks
    Yields:
      - Chunk dict with stable 'chunk_id' = f"{doc_id}-{ordinal}"
    """
    text = doc["text"]
    # Paragraph split: preserve double newlines as boundaries
    # Before packing, create sections around headings for better locality
    sections = _split_into_sections(text, doc.get("headings", []))

    ordinal = 0
    for sec in sections:
        body = sec["body"]
        # naïve paragraph split
        paras = [p.strip() for p in body.split("\n\n") if p.strip()]
        if not paras: continue

        packed = _pack_by_tokens(paras, max_tokens=max_tokens, overlap_tokens=overlap_tokens)

        # Compute char spans approximately by searching substrings sequentially
        cursor = sec["start"]
        for piece in packed:
            # find piece in original slice
            rel = text.find(piece, cursor, sec["end"])
            start = rel if rel != -1 else cursor
            end = start + len(piece)
            heading_path = [h for h in sec.get("path", [])]
            chunk: Chunk = {
                "chunk_id": f"{doc['doc_id']}-{ordinal}",
                "doc_id": doc["doc_id"],
                "ordinal": ordinal,
                "text": piece,
                "heading_path": heading_path,
                "start_char": start,
                "end_char": end,
                "page_or_time": doc["metadata"].get("page_or_time"),
                "metadata": {
                    **doc["metadata"],
                    "heading_path": heading_path,
                },
            }
            yield chunk
            ordinal += 1
            cursor = end

```
⸻
```
embed.py

# src/ingestion/embed.py
from __future__ import annotations
from typing import Sequence, List, Dict, TypedDict, Optional
from langchain_openai import OpenAIEmbeddings

from .chunk import Chunk

class EmbeddingRecord(TypedDict):
    chunk_id: str
    doc_id: str
    embedding: List[float]
    text: str
    metadata: Dict

class Embedder:
    """
    Thin wrapper over LangChain's OpenAIEmbeddings with async + sync helpers.
    """
    def __init__(self, model: str = "text-embedding-3-large"):
        self.model = model
        self._emb = OpenAIEmbeddings(model=model)

    # ---------- batch APIs ----------
    async def aembed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Input: list[str]
        Output: list[list[float]] in same order
        """
        return await self._emb.aembed_documents(list(texts))

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Synchronous fallback (uses .embed_documents under the hood).
        """
        return self._emb.embed_documents(list(texts))

    async def aembed_query(self, text: str) -> List[float]:
        return (await self.aembed_texts([text]))[0]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

# --------- high-level helper ----------
async def aembed_chunks(chunks: Sequence[Chunk], embedder: Optional[Embedder] = None) -> List[EmbeddingRecord]:
    """
    Asynchronously embeds chunks and returns records ready for upsert.
    Input:
      - chunks: Sequence[Chunk]
      - embedder: optional Embedder (created if None)
    Output:
      - List[EmbeddingRecord]: {'chunk_id','doc_id','embedding','text','metadata'}
    """
    emb = embedder or Embedder()
    texts = [c["text"] for c in chunks]
    vecs = await emb.aembed_texts(texts)
    out: List[EmbeddingRecord] = []
    for c, v in zip(chunks, vecs):
        out.append({
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "embedding": v,
            "text": c["text"],
            "metadata": c["metadata"],
        })
    return out

def embed_chunks(chunks: Sequence[Chunk], embedder: Optional[Embedder] = None) -> List[EmbeddingRecord]:
    """
    Synchronous variant of aembed_chunks().
    """
    emb = embedder or Embedder()
    texts = [c["text"] for c in chunks]
    vecs = emb.embed_texts(texts)
    return [{
        "chunk_id": c["chunk_id"],
        "doc_id": c["doc_id"],
        "embedding": v,
        "text": c["text"],
        "metadata": c["metadata"],
    } for c, v in zip(chunks, vecs)]
```

⸻

How they fit together (mini driver)
```
# scripts/example_ingest_driver.py (optional helper to see flow)
import asyncio
from pathlib import Path
from src.ingestion.load_files import walk_data_dir
from src.ingestion.normalize import normalize_rawdoc
from src.ingestion.chunk import chunk_document
from src.ingestion.embed import aembed_chunks

async def run():
    docs = [normalize_rawdoc(d) for d in walk_data_dir("data")]
    for nd in docs:
        chunks = list(chunk_document(nd, max_tokens=350, overlap_tokens=60))
        records = await aembed_chunks(chunks)   # -> ready to upsert to pgvector
        print(nd["doc_id"], len(records), "chunks embedded")

if __name__ == "__main__":
    asyncio.run(run())

```
⸻

if you want, I can add the pgvector “upsert” function next (with SQLAlchemy), or slot these into your build_index.py pipeline you already have.


```python
# pg_vector_store.py

# psycopg + pgvector upsert / schema helpers
from __future__ import annotations
from typing import Iterable, List, Dict, Any, Optional, Sequence, Tuple
import os
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json
from psycopg import sql
from .embed import EmbeddingRecord

DEFAULT_DIM = int(os.getenv("EMBED_DIM", "3072"))

def dsn_from_env() -> str:
    host = os.getenv("PG_HOST", "localhost")
    port = os.getenv("PG_PORT", "5432")
    db   = os.getenv("PG_DB",   "invest_chat")
    usr  = os.getenv("PG_USER", "postgres")
    pwd  = os.getenv("PG_PASSWORD", "postgres")
    return f"host={host} port={port} dbname={db} user={usr} password={pwd}"

def vector_literal(vec: Sequence[float]) -> str:
    # pgvector accepts string form: '[0.1,0.2,...]'
    return "[" + ",".join(f"{x:.7f}" for x in vec) + "]"

class PGVectorStore:
    """
    Minimal pgvector store using psycopg3
    Table schema (recommended fresh setup):

    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE IF NOT EXISTS chunks(
      chunk_id TEXT PRIMARY KEY,
      doc_id   TEXT,
      text     TEXT,
      source_type TEXT,
      source_path TEXT,
      heading_path JSONB,
      page_or_time JSONB,
      metadata JSONB,
      embedding vector(<DIM>)
    );
    CREATE INDEX IF NOT EXISTS chunks_embedding_ivfflat
      ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists=100);
    """
    def __init__(self, dsn: Optional[str] = None, embed_dim: int = DEFAULT_DIM):
        self.dsn = dsn or dsn_from_env()
        self.embed_dim = embed_dim

    def connect(self):
        return psycopg.connect(self.dsn, row_factory=dict_row)

    def ensure_schema(self):
        ddl = f"""
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS chunks(
          chunk_id TEXT PRIMARY KEY,
          doc_id   TEXT,
          text     TEXT,
          source_type TEXT,
          source_path TEXT,
          heading_path JSONB,
          page_or_time JSONB,
          metadata JSONB,
          embedding vector({self.embed_dim})
        );
        """
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(ddl)
            # Build ANN index (ivfflat) if missing
            cur.execute("""
                DO $$
                BEGIN
                  IF NOT EXISTS (
                    SELECT 1 FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE c.relname = 'chunks_embedding_ivfflat'
                  ) THEN
                    EXECUTE 'CREATE INDEX chunks_embedding_ivfflat
                             ON chunks USING ivfflat (embedding vector_cosine_ops)
                             WITH (lists=100)';
                  END IF;
                END $$;
            """)
            # Optional text search index (uncomment if you want sparse search baseline)
            # cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
            # cur.execute("CREATE INDEX IF NOT EXISTS chunks_text_gin ON chunks USING gin (to_tsvector('english', text));")
            conn.commit()

    def upsert_batch(self, records: Sequence[EmbeddingRecord], batch_size: int = 500):
        """
        INSERT ... ON CONFLICT (chunk_id) DO UPDATE.
        The vector is passed as a string literal cast to ::vector.
        """
        if not records:
            return 0
        inserted = 0
        template = "(%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)"
        sql_stmt = """
        INSERT INTO chunks
        (chunk_id, doc_id, text, source_type, source_path, heading_path, page_or_time, metadata, embedding)
        VALUES %s
        ON CONFLICT (chunk_id) DO UPDATE SET
          doc_id = EXCLUDED.doc_id,
          text   = EXCLUDED.text,
          source_type = EXCLUDED.source_type,
          source_path = EXCLUDED.source_path,
          heading_path = EXCLUDED.heading_path,
          page_or_time = EXCLUDED.page_or_time,
          metadata = EXCLUDED.metadata,
          embedding = EXCLUDED.embedding;
        """
        from psycopg.extras import execute_values

        with self.connect() as conn, conn.cursor() as cur:
            for i in range(0, len(records), batch_size):
                batch = records[i:i+batch_size]
                values = []
                for r in batch:
                    md = r.get("metadata", {}) or {}
                    values.append((
                        r["chunk_id"],
                        r["doc_id"],
                        r["text"],
                        md.get("source_type"),
                        md.get("source_path"),
                        Json(md.get("heading_path", [])),
                        Json(md.get("page_or_time")),
                        Json(md),
                        vector_literal(r["embedding"]),   # casted in template
                    ))
                execute_values(cur, sql_stmt, values, template=template, page_size=batch_size)
                inserted += len(batch)
            conn.commit()
        return inserted

    # (Optional) quick cosine ANN search to sanity-check index
    def search_by_vector(self, qvec: Sequence[float], k: int = 12) -> List[Dict[str, Any]]:
        qlit = vector_literal(qvec)
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, doc_id, text, source_path, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM chunks
                ORDER BY embedding <-> %s::vector
                LIMIT %s;
                """, (qlit, qlit, k)
            )
            return cur.fetchall()

```
## scripts/build_index.py

```python
# Build the pgvector index from data/ using psycopg, batching embeddings.
from __future__ import annotations
import os, asyncio, math
from pathlib import Path
from typing import List
from src.ingestion.load_files import walk_data_dir
from src.ingestion.normalize import normalize_rawdoc
from src.ingestion.chunk import chunk_document, Chunk
from src.ingestion.embed import aembed_chunks, Embedder, EmbeddingRecord
from src.ingestion.pgvector_store import PGVectorStore, DEFAULT_DIM

DATA_DIR = os.getenv("DATA_DIR", "data")
MAX_TOKENS   = int(os.getenv("CHUNK_MAX_TOKENS", "350"))
OVERLAP_TOKS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "60"))
EMBED_BATCH  = int(os.getenv("EMBED_BATCH", "256"))
UPSERT_BATCH = int(os.getenv("UPSERT_BATCH", "500"))

async def embed_and_upsert(
    store: PGVectorStore,
    embedder: Embedder,
    chunks: List[Chunk]
) -> int:
    # Embed in manageable batches to respect rate limits
    total = 0
    for i in range(0, len(chunks), EMBED_BATCH):
        sub = chunks[i:i+EMBED_BATCH]
        recs: List[EmbeddingRecord] = await aembed_chunks(sub, embedder)
        total += store.upsert_batch(recs, batch_size=UPSERT_BATCH)
    return total

async def main():
    print(f"[build_index] Using DATA_DIR={DATA_DIR}")
    store = PGVectorStore(embed_dim=DEFAULT_DIM)
    store.ensure_schema()

    embedder = Embedder(model=os.getenv("EMBED_MODEL", "text-embedding-3-large"))

    # Stream docs → normalize → chunk → buffer → embed+upsert
    buffer: List[Chunk] = []
    indexed = 0
    doc_count = 0

    for raw in walk_data_dir(DATA_DIR, include_pdf=False):
        doc_count += 1
        norm = normalize_rawdoc(raw)
        pieces = list(chunk_document(norm, max_tokens=MAX_TOKENS, overlap_tokens=OVERLAP_TOKS))
        buffer.extend(pieces)

        # Flush in ~2000-chunk groups for memory safety
        if len(buffer) >= 2000:
            print(f"[build_index] Embedding {len(buffer)} chunks …")
            indexed += await embed_and_upsert(store, embedder, buffer)
            buffer.clear()
            print(f"[build_index] Total indexed so far: {indexed}")

    if buffer:
        print(f"[build_index] Embedding final {len(buffer)} chunks …")
        indexed += await embed_and_upsert(store, embedder, buffer)
        buffer.clear()

    print(f"[build_index] DONE. Docs: {doc_count}, Chunks indexed: {indexed}")

if __name__ == "__main__":
    asyncio.run(main())

```






# Selenium + All Links

```python

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import json
import os
from collections import deque
import time
import networkx as nx
import matplotlib.pyplot as plt

class InteractiveWebsiteKnowledgeGraph:
    def __init__(self, start_url, max_depth=4, output_dir="interactive_kg_output"):
        self.start_url = start_url
        self.max_depth = max_depth
        self.output_dir = output_dir
        self.visited = set()
        self.graph = nx.DiGraph()
        self.page_contents = {}
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/pages", exist_ok=True)
        
        # Setup Selenium
        self.driver = self.setup_driver()
        
    def setup_driver(self):
        """Setup Selenium WebDriver with options"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # Run in background
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        # Disable images and CSS for faster loading (optional)
        prefs = {
            'profile.managed_default_content_settings.images': 2,
            'profile.managed_default_content_settings.stylesheets': 2
        }
        chrome_options.add_experimental_option('prefs', prefs)
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(30)
            return driver
        except Exception as e:
            print(f"Error setting up Chrome driver: {e}")
            print("Make sure you have Chrome and ChromeDriver installed.")
            print("Install: pip install selenium")
            print("Download ChromeDriver: https://chromedriver.chromium.org/")
            raise
    
    def click_interactive_elements(self):
        """Click on dropdowns, buttons, and expandable elements to reveal hidden links"""
        elements_to_click = []
        
        # Find common interactive elements
        selectors = [
            "button[aria-expanded='false']",
            ".dropdown-toggle",
            ".menu-toggle",
            "[role='button']",
            ".accordion-header",
            ".expand-button",
            ".show-more",
            "button:not([type='submit'])",
            "[onclick]"
        ]
        
        for selector in selectors:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                elements_to_click.extend(elements)
            except:
                continue
        
        # Click elements to reveal content
        clicked_count = 0
        for element in elements_to_click[:20]:  # Limit to prevent infinite loops
            try:
                # Scroll element into view
                self.driver.execute_script("arguments[0].scrollIntoView(true);", element)
                time.sleep(0.3)
                
                # Check if element is visible and clickable
                if element.is_displayed() and element.is_enabled():
                    element.click()
                    clicked_count += 1
                    time.sleep(0.5)  # Wait for content to load
            except Exception as e:
                continue
        
        return clicked_count
    
    def get_page_content(self, url):
        """Fetch webpage content using Selenium"""
        try:
            print(f"  Loading page with Selenium...")
            self.driver.get(url)
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Additional wait for dynamic content
            time.sleep(2)
            
            # Click interactive elements to reveal hidden links
            clicked = self.click_interactive_elements()
            if clicked > 0:
                print(f"  Clicked {clicked} interactive elements to reveal content")
                time.sleep(1)  # Wait for revealed content
            
            # Get the rendered page source
            page_source = self.driver.page_source
            
            return page_source
            
        except TimeoutException:
            print(f"  Timeout loading page: {url}")
            return None
        except Exception as e:
            print(f"  Error fetching {url}: {e}")
            return None
    
    def extract_text_content(self, soup, url):
        """Extract clean text content from page"""
        # Check if it's a PDF
        if url.lower().endswith('.pdf'):
            return f"PDF Document: {url}\n\nThis is a PDF file. The URL is stored as content."
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "iframe"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def extract_all_links_with_context(self, soup, base_url):
        """Extract ALL links with their surrounding context"""
        links_with_context = []
        seen_urls = set()
        
        # Find all links
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            
            # Skip non-http links, fragments, and mailto
            if not href or href.startswith('#') or href.startswith('mailto:') or href.startswith('javascript:'):
                continue
            
            # Convert to absolute URL
            absolute_url = urljoin(base_url, href)
            
            # Skip if already processed
            if absolute_url in seen_urls:
                continue
            
            # Only process URLs from the same domain
            if not self.is_same_domain(absolute_url, self.start_url):
                continue
            
            seen_urls.add(absolute_url)
            
            # Get link text
            link_text = link.get_text().strip()
            if not link_text:
                link_text = "Link"
            
            # Get surrounding context
            context = self.get_surrounding_text(link)
            
            # Extract 4 words before and after
            relationship = self.extract_relationship(context, link_text)
            
            links_with_context.append({
                'url': absolute_url,
                'link_text': link_text,
                'relationship': relationship,
                'full_context': context
            })
        
        return links_with_context
    
    def is_same_domain(self, url1, url2):
        """Check if two URLs are from the same domain"""
        domain1 = urlparse(url1).netloc
        domain2 = urlparse(url2).netloc
        
        return domain1 == domain2
    
    def get_surrounding_text(self, link_element):
        """Get text surrounding a link element"""
        # Try to get parent paragraph or containing element
        parent = link_element.find_parent(['p', 'li', 'div', 'td', 'span', 'section'])
        
        if parent:
            text = parent.get_text()
            return ' '.join(text.split())
        
        return link_element.get_text()
    
    def extract_relationship(self, context, link_text):
        """Extract 4 words before and 4 words after the link text"""
        # Clean the context
        context = ' '.join(context.split())
        
        # Find the link text in context
        pattern = re.escape(link_text)
        match = re.search(pattern, context, re.IGNORECASE)
        
        if not match:
            return link_text
        
        # Get position of link text
        start_pos = match.start()
        end_pos = match.end()
        
        # Extract words before
        before_text = context[:start_pos].strip()
        before_words = before_text.split()[-4:] if before_text else []
        
        # Extract words after
        after_text = context[end_pos:].strip()
        after_words = after_text.split()[:4] if after_text else []
        
        # Construct relationship
        relationship_parts = before_words + [link_text] + after_words
        relationship = ' '.join(relationship_parts)
        
        return relationship if relationship else link_text
    
    def is_pdf(self, url):
        """Check if URL points to a PDF"""
        return url.lower().endswith('.pdf')
    
    def save_page_content(self, url, content):
        """Save page content to a text file"""
        # Create a safe filename from URL
        parsed = urlparse(url)
        filename = parsed.path.replace('/', '_').strip('_')
        if not filename:
            filename = 'index'
        
        # Limit filename length
        filename = filename[:200] + '.txt'
        
        filepath = os.path.join(self.output_dir, 'pages', filename)
        
        # Avoid overwriting by adding numbers
        counter = 1
        original_filepath = filepath
        while os.path.exists(filepath):
            name, ext = os.path.splitext(original_filepath)
            filepath = f"{name}_{counter}{ext}"
            counter += 1
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"URL: {url}\n")
            f.write("=" * 80 + "\n\n")
            f.write(content)
        
        return filepath
    
    def crawl(self):
        """Crawl website using BFS with depth tracking - ALL links per page"""
        queue = deque([(self.start_url, 0)])  # (url, depth)
        
        try:
            while queue:
                current_url, depth = queue.popleft()
                
                # Skip if already visited or max depth reached
                if current_url in self.visited or depth > self.max_depth:
                    continue
                
                print(f"\n{'='*80}")
                print(f"Crawling (depth {depth}): {current_url}")
                print(f"{'='*80}")
                self.visited.add(current_url)
                
                # Add node to graph
                self.graph.add_node(current_url, depth=depth)
                
                # Handle PDF files
                if self.is_pdf(current_url):
                    print(f"  PDF detected - storing URL as content")
                    content = f"PDF Document: {current_url}"
                    filepath = self.save_page_content(current_url, content)
                    self.page_contents[current_url] = filepath
                    continue
                
                # Fetch page content with Selenium
                html_content = self.get_page_content(current_url)
                if not html_content:
                    continue
                
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Extract and save text content
                text_content = self.extract_text_content(soup, current_url)
                filepath = self.save_page_content(current_url, text_content)
                self.page_contents[current_url] = filepath
                
                print(f"  ✓ Saved content to: {filepath}")
                
                # Extract ALL links with context
                links_with_context = self.extract_all_links_with_context(soup, current_url)
                print(f"  ✓ Found {len(links_with_context)} unique links")
                print(f"  ✓ Processing ALL links (no limit)")
                
                # Process ALL links
                for i, link_info in enumerate(links_with_context, 1):
                    target_url = link_info['url']
                    relationship = link_info['relationship']
                    
                    # Add edge to graph with relationship as attribute
                    self.graph.add_edge(
                        current_url, 
                        target_url,
                        relationship=relationship,
                        link_text=link_info['link_text']
                    )
                    
                    # Add to queue if not visited and within depth limit
                    if target_url not in self.visited and depth < self.max_depth:
                        queue.append((target_url, depth + 1))
                    
                    # Progress indicator
                    if i % 50 == 0:
                        print(f"  ... processed {i}/{len(links_with_context)} links")
                
                print(f"  ✓ Completed processing all {len(links_with_context)} links")
                print(f"  Queue size: {len(queue)} | Visited: {len(self.visited)}")
                
                # Be polite to the server
                time.sleep(1)
            
        finally:
            # Close the browser
            self.driver.quit()
        
        print(f"\n\n{'='*80}")
        print("CRAWLING COMPLETE!")
        print(f"{'='*80}")
        print(f"Total pages crawled: {len(self.visited)}")
        print(f"Total relationships: {self.graph.number_of_edges()}")
    
    def save_graph_data(self):
        """Save graph data to JSON"""
        graph_data = {
            'nodes': [],
            'edges': []
        }
        
        # Add nodes
        for node in self.graph.nodes():
            graph_data['nodes'].append({
                'url': node,
                'depth': self.graph.nodes[node].get('depth', 0),
                'content_file': self.page_contents.get(node, ''),
                'is_pdf': self.is_pdf(node)
            })
        
        # Add edges
        for source, target, data in self.graph.edges(data=True):
            graph_data['edges'].append({
                'source': source,
                'target': target,
                'relationship': data.get('relationship', ''),
                'link_text': data.get('link_text', '')
            })
        
        # Store for later use
        self.graph_data = graph_data
        
        # Save to JSON
        output_file = os.path.join(self.output_dir, 'knowledge_graph.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Graph data saved to: {output_file}")
        return output_file
    
    def visualize_graph(self):
        """Create a visualization of the knowledge graph with relationship labels"""
        plt.figure(figsize=(24, 20))
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(self.graph, k=3, iterations=100, seed=42)
        
        # Color nodes by depth
        depths = [self.graph.nodes[node].get('depth', 0) for node in self.graph.nodes()]
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_color=depths,
            cmap=plt.cm.viridis,
            node_size=700,
            alpha=0.8
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            self.graph, pos,
            edge_color='gray',
            arrows=True,
            arrowsize=15,
            alpha=0.4,
            width=1.5,
            connectionstyle='arc3,rad=0.1'
        )
        
        # Draw node labels
        node_labels = {}
        for node in self.graph.nodes():
            path = urlparse(node).path
            label = path.split('/')[-1][:20] if path else 'root'
            if self.is_pdf(node):
                label = "📄 " + label
            node_labels[node] = label
        
        nx.draw_networkx_labels(
            self.graph, pos,
            node_labels,
            font_size=7,
            font_weight='bold'
        )
        
        plt.title("Interactive Website Knowledge Graph\n(Colors represent depth level)", 
                  fontsize=18, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        # Save visualization
        output_file = os.path.join(self.output_dir, 'knowledge_graph_visualization.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Graph visualization saved to: {output_file}")
        
        plt.close()
    
    def create_interactive_html(self):
        """Create an interactive HTML visualization"""
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Interactive Website Knowledge Graph</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }
        h1 {
            color: #333;
            border-bottom: 4px solid #667eea;
            padding-bottom: 15px;
            margin-bottom: 20px;
        }
        .badge {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 12px;
            margin-left: 10px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .stat-card h3 {
            margin: 0;
            font-size: 32px;
            font-weight: bold;
        }
        .stat-card p {
            margin: 5px 0 0 0;
            font-size: 14px;
            opacity: 0.9;
        }
        .search-box {
            margin: 20px 0;
            padding: 15px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 10px;
        }
        .search-box input {
            width: 100%;
            padding: 12px;
            font-size: 16px;
            border: 2px solid #667eea;
            border-radius: 8px;
            box-sizing: border-box;
        }
        .node-list {
            margin-top: 20px;
        }
        .node-item {
            background-color: #fff;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            transition: all 0.3s;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .node-item:hover {
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
            border-color: #667eea;
            transform: translateY(-2px);
        }
        .node-title {
            font-weight: bold;
            color: #667eea;
            font-size: 18px;
            margin-bottom: 10px;
        }
        .node-url {
            color: #666;
            font-size: 13px;
            word-break: break-all;
            margin-bottom: 12px;
        }
        .depth-badge {
            display: inline-block;
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 12px;
            margin-right: 10px;
            font-weight: bold;
        }
        .pdf-badge {
            display: inline-block;
            background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
            color: white;
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 12px;
            margin-right: 10px;
            font-weight: bold;
        }
        .relationships {
            margin-top: 15px;
            padding: 15px;
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            border-left: 4px solid #ff6a00;
            border-radius: 8px;
        }
        .relationship-title {
            font-weight: bold;
            color: #d35400;
            margin-bottom: 12px;
            font-size: 14px;
        }
        .relationship-item {
            margin: 10px 0;
            padding: 12px;
            background-color: white;
            border-radius: 8px;
            font-size: 13px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .relationship-arrow {
            color: #ff6a00;
            font-weight: bold;
            margin: 0 8px;
        }
        .relationship-context {
            color: #555;
            font-style: italic;
            margin-top: 5px;
            font-size: 12px;
            padding: 8px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }
        .target-link {
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
        }
        .target-link:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🕸️ Interactive Website Knowledge Graph<span class="badge">All Links Crawled</span></h1>
        
        <div class="stats">
            <div class="stat-card">
                <h3 id="totalPages">0</h3>
                <p>Total Pages</p>
            </div>
            <div class="stat-card">
                <h3 id="totalEdges">0</h3>
                <p>Total Links</p>
            </div>
            <div class="stat-card">
                <h3 id="maxDepth">0</h3>
                <p>Maximum Depth</p>
            </div>
            <div class="stat-card">
                <h3 id="pdfCount">0</h3>
                <p>PDF Documents</p>
            </div>
        </div>
        
        <div class="search-box">
            <input type="text" id="searchInput" placeholder="🔍 Search for pages by URL or title..." onkeyup="filterNodes()">
        </div>
        
        <div class="node-list" id="nodeList"></div>
    </div>
    
    <script>
        const graphData = GRAPH_DATA_PLACEHOLDER;
        
        // Display stats
        document.getElementById('totalPages').textContent = graphData.nodes.length;
        document.getElementById('totalEdges').textContent = graphData.edges.length;
        
        const maxDepth = Math.max(...graphData.nodes.map(n => n.depth));
        document.getElementById('maxDepth').textContent = maxDepth;
        
        const pdfCount = graphData.nodes.filter(n => n.is_pdf).length;
        document.getElementById('pdfCount').textContent = pdfCount;
        
        // Create a map of node URLs to their outgoing edges
        const nodeEdges = {};
        graphData.edges.forEach(edge => {
            if (!nodeEdges[edge.source]) {
                nodeEdges[edge.source] = [];
            }
            nodeEdges[edge.source].push(edge);
        });
        
        // Function to get page title from URL
        function getPageTitle(url) {
            const path = new URL(url).pathname;
            const parts = path.split('/').filter(p => p);
            return parts[parts.length - 1] || 'Home';
        }
        
        // Function to display all nodes
        function displayNodes(nodes = graphData.nodes) {
            const nodeList = document.getElementById('nodeList');
            nodeList.innerHTML = '';
            
            nodes.forEach(node => {
                const nodeDiv = document.createElement('div');
                nodeDiv.className = 'node-item';
                
                const title = getPageTitle(node.url);
                const edges = nodeEdges[node.url] || [];
                
                const pdfBadge = node.is_pdf ? '<span class="pdf-badge">📄 PDF</span>' : '';
                
                let relationshipsHtml = '';
                if (edges.length > 0) {
                    relationshipsHtml = '<div class="relationships"><div class="relationship-title">🔗 Outgoing Links (' + edges.length + '):</div>';
                    edges.forEach(edge => {
                        const targetTitle = getPageTitle(edge.target);
                        relationshipsHtml += `
                            <div class="relationship-item">
                                <a href="${edge.target}" class="target-link" target="_blank">${targetTitle}</a>
                                <span class="relationship-arrow">→</span>
                                <div class="relationship-context">"${edge.relationship}"</div>
                            </div>
                        `;
                    });
                    relationshipsHtml += '</div>';
                }
                
                nodeDiv.innerHTML = `
                    <div class="node-title">
                        <span class="depth-badge">Depth ${node.depth}</span>
                        ${pdfBadge}
                        ${title}
                    </div>
                    <div class="node-url">🌐 <a href="${node.url}" target="_blank">${node.url}</a></div>
                    ${relationshipsHtml}
                `;
                
                nodeList.appendChild(nodeDiv);
            });
        }
        
        // Function to filter nodes based on search
        function filterNodes() {
            const searchTerm = document.getElementById('searchInput').value.toLowerCase();
            
            if (!searchTerm) {
                displayNodes();
                return;
            }
            
            const filteredNodes = graphData.nodes.filter(node => {
                const title = getPageTitle(node.url).toLowerCase();
                const url = node.url.toLowerCase();
                return title.includes(searchTerm) || url.includes(searchTerm);
            });
            
            displayNodes(filteredNodes);
        }
        
        // Initial display
        displayNodes();
    </script>
</body>
</html>
        """
        
        # Replace placeholder with actual data
        json_data = json.dumps({
            'nodes': [{'url': n['url'], 'depth': n['depth'], 'is_pdf': n.get('is_pdf', False)} 
                     for n in self.graph_data['nodes']],
            'edges': self.graph_data['edges']
        }, ensure_ascii=False)
        
        html_content = html_content.replace('GRAPH_DATA_PLACEHOLDER', json_data)
        
        # Save HTML file
        output_file = os.path.join(self.output_dir, 'knowledge_graph_interactive.html')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ Interactive HTML visualization saved to: {output_file}")
        return output_file
    
    def generate_report(self):
        """Generate a summary report"""
        report_file = os.path.join(self.output_dir, 'report.txt')
        
        pdf_pages = [n for n in self.graph.nodes() if self.is_pdf(n)]
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("INTERACTIVE WEBSITE KNOWLEDGE GRAPH REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Start URL: {self.start_url}\n")
            f.write(f"Max Depth: {self.max_depth}\n")
            f.write(f"Crawl Method: Interactive (Selenium + All Links)\n")
            f.write(f"Total Pages Crawled: {len(self.visited)}\n")
            f.write(f"Total Relationships: {self.graph.number_of_edges()}\n")
            f.write(f"PDF Documents Found: {len(pdf_pages)}\n\n")
            
            # Pages by depth
            f.write("Pages by Depth:\n")
            for depth in range(self.max_depth + 1):
                pages_at_depth = [n for n in self.graph.nodes() 
                                 if self.graph.nodes[n].get('depth') == depth]
                f.write(f"  Depth {depth}: {len(pages_at_depth)} pages\n")
            
            # PDF list
            if pdf_pages:
                f.write("\n" + "=" * 80 + "\n")
                f.write("PDF DOCUMENTS:\n\n")
                for pdf in pdf_pages:
                    f.write(f"  - {pdf}\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
            f.write("SAMPLE RELATIONSHIPS:\n\n")
            
            # Show sample relationships
            for i, (source, target, data) in enumerate(list(self.graph.edges(data=True))[:30]):
                source_name = urlparse(source).path.split('/')[-1] or 'root'
                target_name = urlparse(target).path.split('/')[-1] or 'root'
                relationship = data.get('relationship', '')
                
                f.write(f"{i+1}. {source_name}\n")
                f.write(f"   -> {target_name}\n")
                f.write(f"   Relationship: {relationship}\n\n")
        
        print(f"✓ Report saved to: {report_file}")


def main():
    print("=" * 80)
    print("INTERACTIVE WEBSITE KNOWLEDGE GRAPH CRAWLER")
    print("=" * 80)
    print("This crawler uses Selenium to handle JavaScript, dropdowns, and buttons")
    print("It will crawl ALL links found on each page (no limit)")
    print("=" * 80 + "\n")
    
    start_url = input("Enter the starting URL: ").strip()
    
    if not start_url:
        print("Error: URL is required")
        return
    
    max_depth = int(input("Enter maximum depth (default 4): ").strip() or "4")
    
    print("\n" + "=" * 80)
    print("STARTING KNOWLEDGE GRAPH CREATION")
    print("=" * 80)
    print(f"Start URL: {start_url}")
    print(f"Max Depth: {max_depth}")
    print(f"Mode: Interactive (Selenium + All Links)")
    print("=" * 80 + "\n")
    
    print("Setting up Selenium WebDriver...")
    
    # Create knowledge graph
    kg = InteractiveWebsiteKnowledgeGraph(start_url, max_depth=max_depth)
    
    # Crawl the website
    kg.crawl()
    
    # Save graph data
    kg.save_graph_data()
    
    # Create interactive HTML visualization
    kg.create_interactive_html()
    
    # Visualize the graph (PNG)
    kg.visualize_graph()
    
    # Generate report
    kg.generate_report()
    
    print("\n" + "=" * 80)
    print("✓ KNOWLEDGE GRAPH CREATION COMPLETE!")
    print("=" * 80)
    print(f"\nCheck the 'interactive_kg_output' directory for:")
    print("  ✓ knowledge_graph_interactive.html - Interactive visualization (OPEN THIS!)")
    print("  ✓ knowledge_graph.json - Complete graph structure")
    print("  ✓ knowledge_graph_visualization.png - Visual representation")
    print("  ✓ pages/*.txt - Individual page contents")
    print("  ✓ report.txt - Detailed summary report")
    print("=" * 80)


if __name__ == "__main__":
    main()

```
