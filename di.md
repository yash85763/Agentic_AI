Yes — here is the updated, detailed solution to both questions.

I am going to answer them in the form I would actually use as the current architecture decision.

1) What if the user query cannot be solved with the DAGs we already created?

That is a real problem, and a static DAG library by itself does not scale.

So the solution is:

Do not make the DAG library the only planner.

Make it the first planning tier, then add a typed plan compiler for unseen queries.

The right planning model

Use three planning tiers.

Tier A: DAG macros for common patterns

These are your hand-curated, high-confidence analytical workflows:
	•	top-N by metric
	•	current vs prior period
	•	trend vs baseline
	•	contribution analysis
	•	drill-down variance
	•	rank-within-group
	•	hierarchy roll-up / drill-down

This tier is best for:
	•	stakeholder POCs
	•	common enterprise questions
	•	high precision and low latency

Tier B: Dynamic DAG synthesis from a typed operator algebra

When no existing DAG fits well enough, do not fall back to arbitrary code generation.

Instead:
	1.	classify the query into a supported query family,
	2.	resolve typed slots:
	•	metrics
	•	dimensions
	•	hierarchy levels
	•	time grain
	•	filters
	3.	compile the request into a DAG using a finite operator set.

That finite operator set is your real scalable foundation:
	•	ResolveColumns
	•	ResolveHierarchy
	•	RetrieveChunks
	•	MaterializeSubtable
	•	FilterRows
	•	GroupAggregate
	•	JoinTables
	•	WindowMetric
	•	ComparePeriods
	•	RankWithinGroup
	•	DrillDown
	•	RollUp
	•	SelectOutput
	•	ValidateOutput
	•	ExplainLineage

This means the system scales by composing validated operators, not by pre-authoring every DAG.

Tier C: Clarify or abstain

If the query is underdetermined or confidence is too low, the system should:
	•	ask a targeted clarification question,
	•	or return likely interpretations,
	•	or abstain.

This is essential if you want rigor and trust.

Why this is the right answer

This design borrows the good parts of several lines of work:

ReWOO shows that separating planning from repeated observation can reduce redundant prompting and improve efficiency; its planner/worker/solver split is a good template for producing a full plan first rather than repeatedly replanning after each tool call.  ￼

LLMCompiler shows that once a plan exists, function calls can be represented as a task graph and independent branches can be executed in parallel, improving latency, cost, and sometimes accuracy over purely sequential ReAct-style loops.  ￼

H-STAR shows that step-wise extraction — first relevant columns, then relevant rows — plus question-type-aware reasoning is more reliable than flat one-shot table prompting.  ￼

TableZoomer and BRTR both reinforce the same idea from another angle: use query-aware narrowing / iterative retrieval over large or messy spreadsheet-like data rather than trying to reason over the whole source at once.  ￼

Practical conclusion

So the correct answer to the “DAG coverage” problem is:

DAGs are macros, not the whole language.

The real scalable system is a typed operator algebra + plan compiler, with DAG templates as high-confidence shortcuts.

That removes the scaling bottleneck.

⸻

2) A well-detailed pipeline architecture

Here is the pipeline I recommend now.

The simplest way to think about it is:

Heuristic shell for understanding and localization
+
deterministic core for execution and verification

⸻

A. High-level architecture

There are six layers:
	1.	State layer
	2.	Ingestion + semantic modeling layer
	3.	Retrieval layer
	4.	Planning layer
	5.	Execution + validation layer
	6.	Output layer

And the user-facing control is still:

one orchestrator agent over these layers.

So the application is still:
	•	not many chatty agents,
	•	not one monolithic free-form agent,
	•	but one orchestrator using strong tools and memory.

⸻

B. Detailed pipeline

Stage 0: State bootstrap

When a request comes in, load:
	•	active DatasetState
	•	active ViewState
	•	active ChartState
	•	session preferences
	•	pending approvals
	•	version pointers

Stored where
	•	PostgreSQL / Aurora PostgreSQL: source of truth for session/version metadata
	•	Redis: hot current-session cache
	•	S3: heavy blobs like raw uploads, materialized views, chart JSONs

Classes
	•	SessionStateStore
	•	DatasetStateStore
	•	ViewStateStore
	•	ChartStateStore
	•	PreferenceStore

Why

This ensures follow-ups, revert, approval continuation, and chart edits are deterministic.

⸻

Stage 1: File ingestion and canonicalization

When a user uploads CSV / Excel:
	1.	store raw file in S3
	2.	parse sheets / tables / blocks
	3.	normalize into canonical columnar storage
	4.	write canonical tables as Parquet to S3

Why Parquet

Because it is compact, columnar, and works well with DuckDB.

Stored where
	•	raw file: s3://.../uploads/...
	•	canonical normalized parquet: s3://.../datasets/{dataset_id}/canonical/*.parquet

Classes
	•	UploadService
	•	WorkbookParser
	•	TableBlockDetector
	•	Canonicalizer
	•	ParquetWriter

Validation here
	•	file type validation
	•	size limit validation
	•	parseability validation
	•	per-sheet/table structural validation

⸻

Stage 2: Semantic profiling and dataset model creation

Now build the semantic model.

For each table/sheet/block/column, infer:
	•	column name
	•	type
	•	null rate
	•	cardinality
	•	sample values
	•	likely semantic role:
	•	measure
	•	dimension
	•	time
	•	identifier
	•	hierarchy level
	•	granularity hints
	•	hierarchy candidates
	•	pivot-likeness / block-likeness

Stored where
	•	summary metadata in Postgres
	•	embeddings in pgvector
	•	semantic graph relationships in Neptune
	•	rich JSON dictionary in S3 if large

Classes
	•	SchemaProfiler
	•	ColumnRoleInferencer
	•	HierarchyInducer
	•	GranularityInferencer
	•	SemanticModelBuilder

Validation here
	•	type confidence thresholds
	•	hierarchy consistency checks
	•	semantic role confidence
	•	fallback/manual override for low-confidence critical fields

Why

This is the layer that lets the system decide whether the data should be treated as:
	•	flat tabular
	•	hierarchical
	•	pivot-like
	•	spreadsheet-block based

⸻

Stage 3: Semantic chunking

Now break the dataset into meaningful retrieval units.

Not arbitrary token chunks — meaningful subtables:
	•	row blocks
	•	structural blocks
	•	hierarchy slices
	•	sheet sections
	•	pivot-like aggregates if present

Each chunk stores:
	•	chunk_id
	•	parent dataset/table/sheet
	•	row range
	•	columns present
	•	semantic labels
	•	hierarchy labels
	•	granularity
	•	textual summary
	•	sample values

Stored where
	•	chunk metadata in Postgres
	•	chunk embeddings in pgvector
	•	chunk payload as Parquet or JSON in S3
	•	graph links in Neptune if needed

Classes
	•	SemanticChunker
	•	ChunkSummarizer
	•	ChunkLabeler
	•	ChunkStore

Validation here
	•	chunk schema validity
	•	row-range validity
	•	summary generation completeness
	•	label consistency

⸻

Stage 4: Multi-index retrieval

At query time, retrieve candidate evidence using multiple indexes in parallel.

Parallel retrieval channels
	•	lexical retrieval over summaries and labels
	•	dense retrieval over embeddings
	•	schema/column retrieval
	•	metadata retrieval
	•	hierarchy/path retrieval

Stored where
	•	lexical index: OpenSearch / BM25-style index or Postgres FTS
	•	embeddings: pgvector
	•	semantic graph: Neptune
	•	metadata: Postgres

Classes
	•	QueryNormalizer
	•	LexicalRetriever
	•	DenseRetriever
	•	SchemaRetriever
	•	HierarchyRetriever
	•	CandidateFusionRanker

Validation here
	•	candidate set size
	•	confidence / score spread
	•	evidence diversity check
	•	retrieval recall on benchmark queries

Why

This stage is optimized for high recall.
It only tries to answer:

where should I look?

Not:

what is the answer?

⸻

Stage 5: Methodology selection and plan compilation

This is the updated answer to the static DAG problem.

The planner receives:
	•	user query
	•	session state
	•	semantic model
	•	retrieved candidate chunks

Then:

5A. Query-family classification

Classify the question into:
	•	scalar lookup
	•	top-N
	•	comparison
	•	trend
	•	trend vs baseline
	•	contribution
	•	drill-down
	•	hierarchy roll-up / drill-down
	•	pivot-style slice/dice
	•	mixed reasoning

5B. Decide planning tier
	•	if a high-confidence DAG macro exists, use it
	•	otherwise compile a DAG from typed operators

5C. Fill typed slots

Resolve:
	•	metric
	•	dimension
	•	hierarchy path
	•	time grain
	•	filters
	•	ranking / comparison mode

Classes
	•	QueryFamilyClassifier
	•	DAGTemplateRegistry
	•	PlanCompiler
	•	SlotResolver
	•	PlanConfidenceScorer

Validation here
	•	all referenced fields exist
	•	operator sequence is type-safe
	•	DAG is acyclic
	•	required slots are filled
	•	if not, clarify or abstain

Why

This is the bridge between natural language and provable execution.

⸻

Stage 6: Fine localization

Now that a candidate plan exists, ground it precisely.

Tasks:
	•	select exact columns
	•	identify relevant row subsets
	•	create focused subtable(s)
	•	determine if multiple chunks must be joined

Classes
	•	ColumnResolver
	•	RowPredicateBuilder
	•	SubtableMaterializer
	•	JoinPlanner

Why

This is where you use the “column first, then rows” logic that H-STAR highlights, and the “query-aware zooming” logic that TableZoomer emphasizes.  ￼

⸻

Stage 7: Deterministic execution core

This is the main execution engine.

Engine choice

Use a DuckDB connection per worker/request, reading canonical Parquet from S3 and materializing temporary or persisted relations as needed.

This is the best practical choice because DuckDB:
	•	can read Parquet directly,
	•	has a lazy relational API,
	•	supports window functions,
	•	and supports ACID transactions with snapshot isolation in persistent mode.  ￼

How to use DuckDB

Recommended pattern:
	•	open DuckDB connection in the worker
	•	read_parquet(...) source relations
	•	create temp views / temp tables
	•	run deterministic operators
	•	write final result to Parquet
	•	store metadata + lineage separately

DuckDB’s Python relational API is lazy until execution is triggered, which is useful for composing intermediate query plans cheaply.  ￼

Operator classes
	•	ResolveColumnsOp
	•	ResolveHierarchyOp
	•	FilterRowsOp
	•	GroupAggregateOp
	•	JoinTablesOp
	•	WindowMetricOp
	•	ComparePeriodsOp
	•	RankWithinGroupOp
	•	DrillDownOp
	•	RollUpOp
	•	SelectOutputOp
	•	ValidateOutputOp
	•	ExplainLineageOp

Validation during execution

Before each operator:
	•	input schema check
	•	type check
	•	precondition check

After each operator:
	•	output schema check
	•	row-count sanity check
	•	null / cardinality sanity check
	•	invariant checks for hierarchy operators

DuckDB-specific note

Window functions like lag, lead, row_number, partitioned sums, and moving windows are built in, which makes DuckDB a strong engine for trend/baseline and rank-within-group workflows. DuckDB notes that window functions are blocking operators and can be memory-intensive, so this is where you should be careful with huge partitions.  ￼

⸻

Stage 8: ViewState materialization

After execution, create the result table.

This is the main truth object.

Persist
	•	result table to Parquet in S3
	•	ViewState metadata in Postgres
	•	active pointer in Redis
	•	lineage refs in Postgres and optionally Neptune

Classes
	•	ViewMaterializer
	•	LineageRecorder
	•	ViewStateStore

Validation here
	•	final schema validation
	•	lineage completeness
	•	confidence threshold
	•	proof / trace completeness

⸻

Stage 9: Response shaping

Now decide what the user actually needs.

Possible outputs:
	•	scalar
	•	result table
	•	textual explanation
	•	result table + chart
	•	clarification request
	•	abstention

Classes
	•	AnswerFormatter
	•	ScalarFormatter
	•	TableFormatter
	•	AbstentionFormatter

⸻

Stage 10: Optional charting handoff

If visualization was requested:
	•	chart planner receives only ViewState
	•	selects chart type / chart family
	•	creates semantic chart spec
	•	deterministic runtime binds actual data into config

Classes
	•	ChartIntentPlanner
	•	ChartSpecBuilder
	•	ChartRuntime
	•	ChartValidator

Important rule

Charting does not re-analyze raw source data.

It only visualizes ViewState.

⸻

C. What is stored where

Here is the clean mapping.

S3

Use for:
	•	raw uploads
	•	canonical parquet
	•	chunk payload parquet/json
	•	result-table parquet (ViewState blobs)
	•	chart config JSONs
	•	audit artifacts

PostgreSQL / Aurora PostgreSQL

Use for:
	•	sessions
	•	dataset metadata
	•	chunk metadata
	•	view metadata
	•	chart metadata
	•	lineage refs
	•	approvals
	•	version pointers
	•	confidence scores
	•	operator traces (lightweight)

Redis / ElastiCache

Use for:
	•	active session pointers
	•	hot caches
	•	in-progress job state
	•	recent retrieval caches
	•	current chart/view selection

pgvector

Use for:
	•	chunk embeddings
	•	column embeddings
	•	query embeddings
	•	semantic retrieval

Neptune

Use for:
	•	semantic graph
	•	hierarchy relationships
	•	column-to-column semantic edges
	•	block/table/hierarchy traversal

DuckDB

Use for:
	•	deterministic execution engine
	•	temporary subtable materialization
	•	joins
	•	groupbys
	•	window metrics
	•	final result-table materialization

⸻

D. What should be validated, and when

Validation must happen at multiple checkpoints.

Ingestion validation
	•	file type
	•	parseability
	•	schema sanity
	•	row/column counts

Semantic model validation
	•	role confidence
	•	hierarchy consistency
	•	granularity consistency
	•	required field presence

Retrieval validation
	•	top-k candidate quality
	•	evidence diversity
	•	confidence thresholds

Plan validation
	•	DAG acyclicity
	•	operator typing
	•	slot completeness
	•	unsupported query-family detection

Execution validation
	•	operator preconditions
	•	schema transitions
	•	hierarchy invariants
	•	aggregation validity
	•	join cardinality checks

Output validation
	•	result-table schema
	•	lineage completeness
	•	confidence threshold
	•	chartability if visualization requested

If any critical validation fails:
	•	clarify,
	•	abstain,
	•	or return a partial supported answer.

⸻

E. What can be parallelized

This is where you make the system fast.

Parallel at ingestion
	•	profiling
	•	hierarchy detection
	•	chunking
	•	summaries
	•	embeddings
	•	metadata graph construction

Parallel at retrieval
	•	lexical retrieval
	•	dense retrieval
	•	schema retrieval
	•	hierarchy retrieval
	•	candidate reranking

Parallel in plan execution

Use an LLMCompiler-style idea here:
	•	if the DAG has independent branches, execute them in parallel
	•	for example:
	•	current-period aggregate branch
	•	baseline aggregate branch
	•	then join/compare later

This is one of the main reasons to compile to DAGs rather than execute purely sequential ReAct loops. LLMCompiler explicitly reports latency gains from planner + task-fetching + parallel executor design.  ￼

Parallel in charting

Once ViewState exists:
	•	textual explanation
	•	chart-type suggestion
	•	theme resolution

can overlap.

⸻

F. A detailed flowchart

flowchart TD
    A[User Query + Session ID] --> B[State Bootstrap]
    B --> C[Load DatasetState / ViewState / ChartState / Preferences]

    C --> D[Query Normalizer]
    D --> E[Parallel Retrieval Layer]

    subgraph RETRIEVAL [Parallel Retrieval]
        E1[Lexical Retriever]
        E2[Dense Retriever / pgvector]
        E3[Schema Retriever]
        E4[Hierarchy Retriever / Neptune]
        E5[Metadata Retriever]
    end

    E --> E1
    E --> E2
    E --> E3
    E --> E4
    E --> E5

    E1 --> F[Candidate Fusion + Rerank]
    E2 --> F
    E3 --> F
    E4 --> F
    E5 --> F

    F --> G[Query Family Classifier]
    G --> H{Existing DAG Macro?}

    H -->|Yes| I[DAG Template Fill]
    H -->|No| J[Typed Plan Compiler]

    I --> K[Plan Verifier]
    J --> K

    K --> L{Plan Valid?}
    L -->|No| M[Clarify / Abstain]
    L -->|Yes| N[Fine Localization]

    N --> O[Materialize Focused Subtable(s)]
    O --> P[DuckDB Execution Engine]

    subgraph EXECUTION [Deterministic Typed Operators]
        P1[FilterRows]
        P2[GroupAggregate]
        P3[JoinTables]
        P4[WindowMetric]
        P5[ComparePeriods]
        P6[RankWithinGroup]
        P7[DrillDown / RollUp]
        P8[SelectOutput]
    end

    P --> P1 --> P2 --> P3 --> P4 --> P5 --> P6 --> P7 --> P8

    P8 --> Q[Execution Validator]
    Q --> R{Valid Output?}
    R -->|No| M
    R -->|Yes| S[Materialize ViewState]

    S --> T{Visualization Requested?}
    T -->|No| U[Format Table / Scalar / Textual Answer]
    T -->|Yes| V[Chart Planner + Chart Runtime]
    V --> W[Chart Validator]
    W --> X[Materialize ChartState]

    U --> Y[Return Response]
    X --> Y
    M --> Y


⸻

G. What this solves

This architecture solves both of your current concerns.

Concern 1: “What if the DAG library does not cover the query?”

Solved by:
	•	using DAGs as macros,
	•	not as the entire language,
	•	and falling back to a typed operator-graph compiler.

Concern 2: “The pipeline is still vague.”

Solved by making every stage explicit:
	•	classes,
	•	storage,
	•	validation,
	•	execution engine,
	•	flow,
	•	and parallelization.

⸻

H. My final recommendation

Use this as the current source of truth:

One orchestrator agent
with
multi-index retrieval,
DAG macros for common patterns,
dynamic typed DAG synthesis for unseen queries,
DuckDB as the deterministic execution engine,
S3 + Postgres + Redis + pgvector + Neptune as the storage/state substrate,
and
strict validation and abstention gates throughout.

That is the scalable version.

If you want, the next step should be to turn this into a repo-level class/module map with exact filenames and interfaces.