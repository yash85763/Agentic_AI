Here’s a tight, production-minded recipe for Graph Retrieval and the context-engineering steps you’ll actually ship.

Graph retrieval — end-to-end in 8 steps
	1.	Normalize the graph (during ingestion)
	•	Nodes: Page, Section, Chunk, Tool, PdfDoc, PdfPage, PdfRegion.
	•	Edges:
contains(Page→Section, Section→Chunk, PdfDoc→PdfPage→PdfRegion),
parent_of(Section→Section),
links_to(Page→Page) (with anchor_text, position),
explained_by(Tool→Section|Chunk),
materialized_from(Chunk→Section|PdfRegion),
(optional) similar_to (kNN over embeddings for offline shortcuts).
	•	Stable IDs and rich metadata on chunks: url|pdf_id, heading_path, breadcrumbs, page_no, bbox, content_hash, last_modified.
	2.	Build indices (multi-index)
	•	Dense vectors: for chunk.text, heading_path, captions/anchor_texts (store as multi-vector per chunk).
	•	BM25/keyword: exact terms, codes, SKUs, formulas.
	•	Graph store: Neo4j / PG + pgvector (or NetworkX if local) with typed edges and edge attributes.
	•	Fielded filters: template_type, pdf_page, tool_name, version, last_modified.
	3.	Generate seed candidates (hybrid)
	•	Query → BM25 + dense over chunk.text (+ separate vectors for headings/anchors).
	•	Keep top 50–100 seeds (balanced across pages/files).
	•	If query looks like navigation, upweight heading/anchor fields; if PDF-ish (“page”, “figure”), bias Pdf* nodes.
	4.	Expand a small subgraph around seeds (rule-based)
	•	Up: add parents (Chunk→Section→Page / Region→Page→Doc) for orientation & breadcrumbs.
	•	Side: add siblings under the same parent when headings share terms (e.g., “Limits”, “Exceptions”).
	•	Down: if question asks for details/examples, pull children (tables, code, figures) of the winning Sections/Pages.
	•	Cross: follow high-signal links_to edges (anchors matching the query, nav/header links), and explained_by (tools → docs).
	•	Cap per hop (e.g., 2 hops, fanout ≤ 3–5 per node) so the subgraph stays small & relevant.
	5.	Score everything (blend graph + text)
\text{score}(n)=
0.45\,\text{text\_rel}+0.25\,\text{graph\_prox}+0.15\,\text{anchor\_hit}+0.10\,\text{authority}+0.05\,\text{freshness}
	•	text_rel: dense + BM25; re-score top 50 with a cross-encoder.
	•	graph_prox: 1/(1+min hop distance) from any seed; bonus if same Section/Page.
	•	anchor_hit: match in anchor_text along incoming/outgoing links_to.
	•	authority: PageRank / menu>body>footer weighting / click logs.
	•	freshness: last_modified decay.
	6.	Select & diversify
	•	Keep 6–12 chunks total; cap at 2 per Section and 3 per Page/File to avoid redundancy.
	•	Always include the parent Page (or PdfDoc→PdfPage summary) node for context + breadcrumbs.
	•	If a Tool is in scope, include the doc chunk that defines its logic and (if executed) the tool_result.
	7.	Context engineering (what you feed the LLM)
	•	Order: high-score → parent summary → supporting siblings → examples/figures/tables.
	•	Budget: ~1,500–2,500 tokens of evidence (after truncating long chunks by boundaries).
	•	Structure: pass a ContextBundle (not raw text) to your answerer:

{
  "query": "How do pro-tier fees work?",
  "persona": {"role":"Support Pro","tone":"concise"},
  "summaries": [
    {"node":"page:/pricing","breadcrumbs":"Home>Pricing","gist":"Explains fee tiers and formula"}
  ],
  "evidence": [
    {"S":"S1","chunk_id":"web:...","url":"https://site/pricing#fee-formula","heading_path":["Pricing","Fee formula"],"text":"Fee = base + rate × amount ...","offsets":[1024,1188]},
    {"S":"S2","chunk_id":"pdf:...","pdf":{"file_id":"whitepaper_v3.pdf","page":3,"bbox":[92,140,510,280]},"text":"Pro rate is 1.8% and base $2.00"}
  ],
  "tool_plan": null
}


	•	Prompt contract: require the model to attach [S#] after each factual sentence and emit a citations[] map (chunk_id→url/pdf.page+bbox). If a sentence can’t be grounded, it must say “not found” or ask to run a tool.

	8.	Verification & learning
	•	Verifier (LLM-as-judge) aligns each sentence → one S#; ungrounded sentences are dropped or revised.
	•	Log which chunks won, user clicks on citations, and missed queries → feed back into hard negative mining and weight tuning.

⸻

Minimal retriever pseudocode (graph-aware)

def graph_retrieve(query, k=10):
    # 1) seeds
    seeds = hybrid_search(query, fields=["text","headings","anchors"], topk=80)
    seeds = cross_encode_rerank(query, seeds, topk=50)

    # 2) expand
    subgraph = set(seeds)
    for s in seeds[:30]:
        subgraph |= parents(s, up_to=2)                 # chunk->section->page/doc
        subgraph |= siblings(s, limit=3, match=query)   # same parent
        subgraph |= children(s, limit=3)                # tables/figures/code
        for p in pages_or_docs(s):
            subgraph |= links_to_high_signal(p, query, limit=3)
            subgraph |= explained_by_tools_near(p, limit=2)

    # 3) score
    scored = []
    for n in subgraph:
        tr = text_rel(query, n)             # dense+bm25 or cross-enc for top N
        gp = graph_proximity(n, seeds)
        ah = anchor_signal(n, query)
        au = authority(n)
        fr = freshness(n)
        score = 0.45*tr + 0.25*gp + 0.15*ah + 0.10*au + 0.05*fr
        scored.append((score, n))

    # 4) diversify & pack
    top_nodes = diversify(scored, by=["section","page","pdf_file"], limit= k + 4)
    evidence = materialize_chunks(top_nodes, cap_per_section=2, cap_per_page=3, k=k)
    bundle = build_context_bundle(query, evidence)  # includes summaries/breadcrumbs
    return bundle


⸻

Context-engineering checklist (graph case)
	•	Chunking discipline: don’t cross H2 for HTML; use page/region for PDFs; store heading_path, breadcrumbs, page_no, bbox, ocr_conf.
	•	Metadata for citations: url with #fragment or pdf_id+page+bbox; also keep offsets to render snippet tooltips.
	•	Packing rules: include at least one overview chunk (parent Page/Doc) + 1–2 definition chunks + a few example/edge case chunks.
	•	Persona hooks: prepend a 1–2 line persona card (tone/format), not the whole history; keep it separate from evidence.
	•	Safety hooks: ingress filter (policy/risk); tool input validation (if tools are executed).
	•	Fallbacks: if graph expansion is too sparse, fall back to plain hybrid; if too dense, restrict to same parent Section and top 1 hop.

⸻

What makes graph retrieval “feel smart”
	•	It keeps structure: parents/siblings provide definitions + examples naturally.
	•	It navigates: links_to with anchor text brings the canonical explainer page.
	•	It grounds PDFs: region-level chunks let you cite page+bbox.
	•	It stays fresh: last_modified + recrawl hashes bias new/changed nodes.

If you want, I can wire this into your current “Supervisor → Retriever Agent → Response Agent” stack with a tiny FastAPI endpoint that returns the ContextBundle above and a cross-encoder reranker stub you can swap for your favorite model.