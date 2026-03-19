Research-Grounded Architecture for Table Reasoning, Stakeholder Analytics, and Chart Generation

A design document for the POC and long-term platform roadmap

> **What this document does**  
> Synthesizes the table-reasoning papers, extracts the exact techniques worth adopting, and turns the latest design decisions into a practical system plan. It also incorporates agentic-system lessons from the uploaded Auton framework paper: declarative specifications, runtime separation, constrained execution, memory, and DAG-aware parallelism.

Prepared for architecture planning and stakeholder-facing POC design

# Executive Summary

The recommended system is a hybrid architecture: one user-facing orchestrator handles intent and workflow selection, while deterministic analytics and chart runtimes perform exact execution. The analytics side uses semantic table understanding, multi-granular chunk retrieval, DAG-guided planning for common stakeholder questions, and deterministic table tools to produce a final result table. The charting side consumes only that result table and turns it into a validated Highcharts or Highstock configuration.

The core design principle is simple: use the LLM for interpretation and plan selection, but not for exact table computation or fragile config serialization. This improves accuracy, keeps latency manageable, supports follow-up edits and history, and gives the system a clean path to future dashboarding and A2A packaging.

> **Bottom-line recommendation**  
> For the POC, use one orchestrator agent over structured subsystems. Do not start with a fully free-form single-agent system, and do not overbuild a heavy multi-agent mesh. Use a DAG library for recurring stakeholder questions, a semantic table retrieval stack to localize candidate subtables, and exact tools to materialize the final analytical table before visualization.

# 1. Problem Framing and Design Goal

The project is not just a chart bot. It needs to understand uploaded spreadsheets or CSVs, answer higher-level business questions, handle hierarchical and pivot-style data, preserve state and versions, and ultimately return either a direct result table, a chart, or both.

Three failure modes must be avoided from the start:

- Full-table prompting: large tables overwhelm context and encourage hallucination or lossy reasoning.
- One-shot free-form planning: the model may choose the wrong operations or skip necessary intermediate steps.
- String-based config generation: final chart objects become inconsistent, hard to validate, and hard to edit.
The architecture therefore needs to do four things well:

1. Understand data semantically before answering questions.
2. Choose the correct reasoning methodology for the data shape and user intent.
3. Execute exact operations with deterministic tools.
4. Materialize a reusable result table that can be shown directly or passed to visualization.
# 2. Research Scan: What Each Paper Solves and What We Should Reuse

The papers below are not all solving the same problem. Some mainly improve retrieval, some improve reasoning fidelity, and some are most useful as architecture patterns for agentic execution. The right answer is not to copy one paper wholesale; it is to compose the strongest ideas into a system that fits the product.

| Paper | Main problem attacked | Core methodology | Technique we should adopt | Primary caution |
| --- | --- | --- | --- | --- |
| TableRAG (NeurIPS 2024) | Large-table scale / context limits | Query expansion + schema retrieval + cell retrieval before reasoning | Schema-first and value-aware retrieval for coarse localization | Retrieval only is not enough; still needs exact execution |
| H-STAR (NAACL 2025) | Mixed semantic and quantitative table reasoning | Multi-view column retrieval, then row extraction, then adaptive reasoning mode | Column-first then row extraction; question-type aware planning | Do not adopt full Text-to-SQL dependency as the main path |
| FRTR (2026 preprint) | Real spreadsheet complexity and workbook-scale reasoning | Row / column / block embeddings + hybrid lexical/dense retrieval + RRF | Spreadsheet-aware chunking and hybrid retrieval | Preprint; promising but less battle-tested |
| RoT / Row-of-Thought | Hallucination during hard table reasoning | Iterative row-wise traversal with reflection/refinement | Hard-question mode and verification mode | Can add latency if used indiscriminately |
| TableZoomer | Target-data localization for large-table QA | Query-aware table zooming + structured schema + executable reasoning / ReAct | Query-aware subtable creation and tool-based execution | Free-form code generation should stay constrained |
| Flow-of-Table-Reasoning (SemEval 2025) | Large tables, poor schema semantics, entity ambiguity | Schema integrating structure and semantics + multi-step schema linking + iterative thinking | Rich schema verbalization and focused sub-schema derivation | Competition system; convert ideas into production-friendly modules |
| QGpT (TRL 2025) | Weak table-side retrieval representations | Generate synthetic questions from partial tables and embed jointly | Later-stage retriever upgrade for chunk representations | Improves retrieval, not full reasoning |
| HELIOS (ACL 2025) | Multi-granular table-text retrieval | Edge-based bipartite subgraph retrieval + node expansion + star-level refinement | Future extension for mixed table + text evidence | Not a first-wave implementation item for table-only POC |

## TableRAG

- Why it matters: it is the clearest argument against whole-table prompting for large data.
- Method in plain language: expand the query, retrieve the relevant schema, retrieve the relevant cells, then reason over the reduced evidence instead of the full table.
- What we will use: coarse retrieval over schema and value signals before any exact analytics execution.
- What we will not rely on: retrieval alone. We still need deterministic tools to compute the final analytical answer.
## H-STAR

- Why it matters: it formalizes a very practical sequenceâidentify the right columns first, then narrow to the right rows, then choose the right reasoning style.
- Method in plain language: multi-view column retrieval followed by row extraction, plus adaptive reasoning for lookup-style vs quantitative/logical questions.
- What we will use: column-first localization and question-type aware planning.
- What we will avoid: making SQL generation the backbone of the product.
## FRTR

- Why it matters: it treats spreadsheets the way users actually encounter themârows, columns, blocks, multiple sheets, and workbook structure.
- Method in plain language: decompose the workbook into row, column, and block units; retrieve with hybrid lexical + dense methods and reciprocal rank fusion.
- What we will use: semantic chunking at the subtable/block level and hybrid retrieval over those chunks.
- What we will monitor: because it is a 2026 preprint, its claims are promising but need practical validation during implementation.
## RoT / Row-of-Thought

- Why it matters: it improves fidelity for hard reasoning by forcing evidence accumulation instead of one-shot free-form explanation.
- Method in plain language: inspect rows or small evidence slices iteratively, reflect, refine, and continue only when needed.
- What we will use: a hard-question verification mode after coarse retrieval and subtable selection.
- What we will avoid: using iterative reflection for every simple query, because that would waste latency.
## TableZoomer

- Why it matters: it is the closest paper to the agentic behavior we wantâlocalize a subtable, then execute over it.
- Method in plain language: represent the table with a focused schema, zoom into the relevant subtable, and use a ReAct / Program-of-Thought style to execute the required operations.
- What we will use: query-aware subtable creation plus tool-based analytics execution.
- What we will avoid: unrestricted code generation as the primary computation mechanism.
## Flow-of-Table-Reasoning

- Why it matters: it shows how schema linking and focused schemas reduce ambiguity on real-world table QA tasks.
- Method in plain language: build a schema that verbalizes both structure and semantics, derive a focused sub-schema through multi-step linking, then reason iteratively.
- What we will use: semantic schema modeling and focused schema derivation before planning.
- What we will avoid: overcomplicating the first POC with too much competition-style pipeline machinery.
## QGpT and HELIOS

- Why they matter: they are not first-wave core components, but they point to useful future upgrades.
- QGpT contributes a better chunk/table representation for retrieval by generating synthetic questions from partial tables.
- HELIOS contributes a strong idea for multi-granular retrieval when future versions need mixed table-plus-text evidence.
- What we will do now: keep both as backlog items, not day-one dependencies.
# 3. Final Recommended Architecture

The recommended architecture is a hybrid orchestrated system. It is not a fully free-form single-agent system, and it is not a heavy multi-agent architecture from day one. Instead, it uses one user-facing orchestrator over deterministic domain runtimes.

> **Chosen operating model**  
> One orchestrator agent handles intent, state loading, DAG selection, and workflow coordination. Under it sit two deterministic domains: (1) analytics, which returns a result table, and (2) charting, which consumes that result table to generate a validated Highcharts or Highstock config.

| Layer | What it does | Why it exists |
| --- | --- | --- |
| Orchestrator | Interprets user request, loads state, chooses question pattern / DAG, routes work | Keeps the user experience unified while avoiding ad hoc coupling between analytics and visualization |
| Analytics subsystem | Builds semantic understanding, retrieves candidate subtables, localizes exact evidence, executes deterministic tools, materializes ViewState | Owns business reasoning and exact table computation |
| Chart subsystem | Chooses chart family/spec, binds actual data from ViewState, validates output, stores ChartState | Owns visualization semantics and config correctness |
| State layer | Stores DatasetState, ViewState, ChartState, preferences, and version pointers | Supports history, follow-up edits, and future API / dashboard workflows |

# 4. End-to-End Pipeline: Step-by-Step Build Plan

## 4.1 Discover the stakeholder operating model

Collect representative datasets, business glossary, hierarchy definitions, and the 20â50 recurring questions that matter most. The POC should optimize for these patterns first.

## 4.2 Ingest and normalize data

Parse CSV / Excel, detect sheets and logical blocks, infer types, sample values, null rates, cardinalities, and likely semantic roles. This produces the initial DatasetState.

## 4.3 Build the semantic data dictionary

For each column and logical table block, store semantic role, type, sample values, likely aliases, and aggregation compatibility. If the data is hierarchical or pivot-oriented, also store hierarchy membership and level order.

## 4.4 Create semantic chunks / subtables

Chunk the dataset into meaningful units rather than arbitrary token windows: row blocks, logical subtables, hierarchy blocks, pivot-like summaries, and sheet regions. Each chunk gets labels, summaries, and metadata.

## 4.5 Index for hybrid retrieval

Index chunk summaries, column/schema documents, chunk metadata, and optionally value-level neighborhoods. Use hybrid lexical + dense retrieval with a rank-fusion step.

## 4.6 Build the stakeholder question-pattern library

Cluster recurring stakeholder questions into patterns such as trend vs baseline, top-N within hierarchy, contribution analysis, drill-down variance, or current-vs-prior-period comparison.

## 4.7 Define DAG templates for those patterns

Each question pattern gets a DAG describing the required analytical substeps and dependencies. These DAGs are planning guides, not free-form code scripts.

## 4.8 Implement analytics planning

The orchestrator or analytics planner classifies the user question, chooses a DAG, binds parameters such as metric, time field, hierarchy levels, filters, and ranking criteria, and decides whether more retrieval is needed.

## 4.9 Run coarse retrieval and shortlist candidates

Use the question plus semantic metadata to retrieve the best candidate subtable or top-k subtables. This is coarse localization.

## 4.10 Run fine localization inside candidates

Within the shortlisted chunk(s), resolve the exact columns, row filters, hierarchy levels, or sub-slices needed for the question. This is where H-STAR-like column-first narrowing and TableZoomer-like subtable focusing matter.

## 4.11 Execute deterministic table tools

Apply exact operations such as filter, group-and-aggregate, rolling-window metric, contribution computation, rank-within-group, roll-up, drill-down, or join. This creates the final result table.

## 4.12 Persist the result as ViewState

The result table becomes ViewState. It can be shown directly to the user, reused in follow-up turns, or passed to charting. This is the most important contract in the architecture.

## 4.13 Plan and build the chart

If visualization is requested, the chart planner selects the chart type and field roles based on ViewState, and the chart runtime binds real data deterministically into the final config.

## 4.14 Save ChartState and update session state

Store the new chart spec/config and update active dataset/view/chart pointers so later edits and reverts are deterministic.

## 4.15 Return response

Return textual answer, result table if requested, chart config if requested, and stable identifiers such as dataset_id, view_id, and chart_id.

# 5. How the DAGs Fit Into Planning

In this design, the DAG is a planning asset, not the final execution engine by itself. It tells the system how to decompose a recurring question class into required analytical steps. The LLM does not invent the workflow from scratch each time; it chooses and parameterizes the right DAG.

- Good fit for the POC: recurring stakeholder questions are often variations of a limited set of business-logic patterns.
- Higher accuracy: the model is guided by an existing reasoning scaffold instead of improvising the whole chain.
- Lower hallucination risk: deterministic tools execute the DAG nodes.
- Better explainability: each result can be tied back to a known workflow pattern.
| DAG pattern | Representative nodes | Typical stakeholder questions |
| --- | --- | --- |
| Trend vs baseline | Resolve time field â aggregate by period â compute moving average / baseline â compare latest to baseline â rank | Which regions are underperforming versus the last 3 months? |
| Top-N within hierarchy | Resolve parent level â resolve child level â aggregate parent-child â rank child within parent â filter top N | Which sub-categories drive sales in each region? |
| Drill-down variance | Aggregate parent level â detect negative variance â drill into child level â compute contribution â rank | Which territories are below plan, and what is causing it? |
| Current vs prior period | Resolve current and prior windows â aggregate both â compute delta and percent delta â sort | How did Q1 compare with Q4 by business unit? |

# 6. Output Contract: What the Analytics Pipeline Returns

The analytics pipeline should not default to returning only a single scalar. Its standard output should be a result package centered on a result table. Scalars are a special case, not the default.

- Scalar output: only when the question is inherently scalar (for example, total revenue or average margin).
- Result table: the default output for analytical questions; this can be shown directly to the user.
- Result package: textual insight + result table + metadata + optional chart recommendation.
> **Most important contract**  
> Analytics should usually return a result table. Charting should consume that result table. This keeps business reasoning and visualization cleanly separated.

# 7. What We Take from the Uploaded Auton Agentic AI Framework Paper

The uploaded Auton paper is not a table-reasoning paper, but it is highly relevant for overall agent-system design. It argues for a declarative separation between the Cognitive Blueprint and the Runtime Engine, deterministic governance through constraints, persistent memory, and runtime efficiency via DAG-style dependency analysis and speculative execution. Those ideas are directly useful for this projectâs architecture.

- Blueprint/runtime separation: the agent specification, contracts, and allowed tools should be declarative and separate from execution logic.
- Deterministic governance: safety and permission scope should be enforced at runtime by design, not just by prompts.
- Hierarchical memory: short-term context plus consolidated long-term lessons are a better fit than raw log replay.
- Graph/DAG execution: independent retrieval and analytics branches can be parallelized, reducing latency to the critical path.
- Speculative execution and dynamic context pruning: useful future optimizations once the core pipeline is stable.
For this project, the strongest Auton-derived design lesson is architectural: treat the orchestrator, analytics runtime, chart runtime, memory, and constraints as first-class components with explicit interfaces, rather than embedding all logic inside a prompt script.

# 8. Efficiency, Parallelism, and Expected Timing

The retrieval-and-analytics pipeline contains both serial and parallel segments. The best performance strategy is to parallelize understanding and retrieval aggressively while keeping exact final execution deterministic and mostly serial.

| Stage | Can it be parallelized? | Notes |
| --- | --- | --- |
| Upload-time preprocessing | Yes | Column profiling, hierarchy detection, chunk generation, summaries, embeddings, and lexical indexing can all run in parallel. |
| Query-time retrieval | Yes | Summary retrieval, schema retrieval, metadata filtering, and reranking can run concurrently. |
| Candidate chunk scoring | Yes | Score top-k candidates for metric fit, time fit, hierarchy fit, and pattern fit in parallel. |
| Methodology / DAG selection | Mostly no | Needs the retrieved evidence and semantic model to make a final planning decision. |
| Fine localization | Partly | Some branches can run in parallel, but the narrowing logic often depends on earlier localization decisions. |
| Deterministic result-table execution | Mostly no | Final tool chain (aggregate â compare â rank â select output) is usually serial or DAG-dependent. |
| Chart planning/build | Partly | Theme resolution and some response formatting can overlap, but final binding waits for ViewState. |

Practical timing estimate for a preprocessed dataset: a normal analytics-only request is likely to land around 2â8.5 seconds; a full analytics-plus-chart request often lands around 4.5â11 seconds. Worst-case synchronous scenariosâvague query, multiple candidate subtables, hierarchy-aware reasoning, heavy deterministic computationsâcan reach roughly 9â23 seconds end to end. This is exactly why the architecture should preserve both a fast synchronous path and an async heavy-analysis path.

> **Latency rule of thumb**  
> Keep the common path synchronous and optimized; move expensive, multi-branch, high-ambiguity or workbook-scale analysis to async execution. The user should not pay worst-case latency on every normal question.

# 9. Rollout Plan

1. Phase 0 â Discovery: collect stakeholder data, glossary, hierarchies, recurring questions, and expected outputs.
2. Phase 1 â Semantic ingestion: implement parsing, data dictionary, hierarchy detection, chunking, and indexing.
3. Phase 2 â DAG-guided analytics: implement question-pattern catalog, DAG templates, analytics planner, and deterministic table tools.
4. Phase 3 â Visualization: implement chart planner, deterministic chart runtime, and config validation.
5. Phase 4 â State and history: add DatasetState / ViewState / ChartState, active pointers, preferences, and versioning.
6. Phase 5 â Hardening: add logging, evaluation harnesses, regression suites on known stakeholder questions, and performance tuning.
7. Phase 6 â Future upgrades: dashboard composition, mixed table-plus-text retrieval, A2A/MCP packaging, and learning loops.
# 10. Final Decisions and Non-Decisions

| Decision | Status | Reasoning |
| --- | --- | --- |
| Use one orchestrator for the POC | Adopt | Lower latency and lower coordination overhead; still allows strong structured domains underneath. |
| Use DAG templates for common stakeholder questions | Adopt | Improves reliability and interpretability for the POC question set. |
| Use semantic chunking plus top-k retrieval | Adopt | Provides scalable coarse localization before fine reasoning. |
| Return result tables as the analytics contract | Adopt | Makes the output reusable for direct display, charting, and future dashboards. |
| Make the charting side re-analyze raw data | Reject | Analysis should be owned by analytics; charting should only visualize ViewState. |
| Make Text-to-SQL the backbone of the product | Reject | Useful ideas can be borrowed, but the full product should not depend on SQL generation as the primary reasoning method. |
| Start with reinforcement learning | Reject for now | Collect traces and feedback first; use supervised and retrieval improvements before RL. |
| Start with a heavy multi-agent fabric | Reject for now | The POC needs reliability, clarity, and speed more than a large agent mesh. |

# References and Source Material

Chen, S.-A. et al. (2024). TableRAG: Million-Token Table Understanding with Language Models. NeurIPS 2024. DOI: 10.52202/079017-2382.

Abhyankar, N., Gupta, V., Roth, D., and Reddy, C. K. (2025). H-STAR: LLM-driven Hybrid SQL-Text Adaptive Reasoning on Tables. NAACL 2025. DOI: 10.18653/v1/2025.naacl-long.445.

Gulati, A. et al. (2026). From Rows to Reasoning: A Retrieval-Augmented Multimodal Framework for Spreadsheet Understanding. 2026 preprint / FRTR-Bench release.

RoT / Row-of-Thought paper discussing iterative row-wise traversal and reflection for table reasoning (2026 search-verified preprint references).

TableZoomer: collaborative/agentic framework for large-scale table question answering (2025 search-verified references).

TeleAI at SemEval-2025 Task 8: Advancing Table Reasoning Framework with Large Language Models / Flow-of-Table-Reasoning. SemEval 2025.

Liang, Y. et al. (2025). Improving Table Retrieval with Question Generation from Partial Tables (QGpT). TRL 2025. DOI: 10.18653/v1/2025.trl-1.19.

Park, S. et al. (2025). HELIOS: Harmonizing Early Fusion, Late Fusion, and LLM Reasoning for Multi-Granular Table-Text Retrieval. ACL 2025. DOI: 10.18653/v1/2025.acl-long.1559.

Zhao, S. C. et al. (2026). The Auton Agentic AI Framework: A Declarative Architecture for Specification, Governance, and Runtime Execution of Autonomous Agent Systems. Uploaded by user for this document.
