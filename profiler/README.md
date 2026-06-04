# Data Profiler Agent

A LangGraph-based agent that builds a complete structural and semantic understanding
of tabular data (CSV, Excel, SQL) before any downstream analyst agent answers questions.

## Why this exists

Users rarely describe their data accurately:

- Vague names: “show me customers” → which column(s)?
- Abbreviations: `txn_amt`, `cust_id`, `fico`, `gmv`
- Indirect questions: “how are we doing?” → revenue? churn? NPS?

The profiler runs first and produces a rich schema + compressed LLM summary that
downstream agents use to map user intent to actual column names and values.

## Project Structure

```
data_profiler_agent/
│
├── agent.py                  ← Main entry point / public API
├── state.py                  ← LangGraph TypedDict state
├── llm_config.py             ← Configurable Anthropic / OpenAI backend
├── requirements.txt
├── .env.example
├── example_usage.py          ← Integration examples
│
├── nodes/
│   ├── ingest.py             ← Node 1: Load CSV / Excel / SQL safely
│   ├── profile_columns.py    ← Node 2: Per-column stats, types, exemplars
│   ├── detect_relationships.py ← Node 3: FK map + join candidates + alias graph
│   ├── llm_summarise.py      ← Node 4: LLM generates compressed summary
│   └── write_outputs.py      ← Node 5: Save JSON + markdown + alias graph
│
└── utils/
    ├── type_inference.py     ← Semantic type detection + alias resolution
    └── exemplars.py          ← Strategic exemplar sampling
```

## Graph Flow

```
START → ingest → profile_columns → detect_relationships → llm_summarise → write_outputs → END
                     ↑ (abort if ingest fails)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env — set LLM_PROVIDER and API key
```

## Usage

```python
from agent import run_profiler

# CSV
result = run_profiler(
    source_type="csv",
    path="./data/orders.csv",
    knowledge_file="./data/orders_knowledge.md",  # optional
)

# Excel
result = run_profiler(
    source_type="excel",
    path="./data/report.xlsx",
    sheet_name="Sales",
)

# SQL
result = run_profiler(
    source_type="sql",
    connection_string="postgresql://user:pass@host:5432/db",
    table_name="orders",
)

# Result keys:
result["rich_profile"]           # Full RichProfile TypedDict (in-memory)
result["llm_summary"]            # Compressed markdown — inject into downstream agent
result["alias_graph"]            # {column → {concept, exemplars, type, flags}}
result["output_path_json"]       # Path to saved data_profile_rich.json
result["output_path_summary"]    # Path to saved data_profile_llm_summary.md
result["output_path_alias_graph"]# Path to saved data_profile_alias_graph.json
result["errors"]                 # List of any errors
result["completed"]              # bool
```

## Outputs

|File                           |Contents                                       |Use                                       |
|-------------------------------|-----------------------------------------------|------------------------------------------|
|`data_profile_rich.json`       |Full column stats, all exemplars, relationships|Storage, inspection                       |
|`data_profile_llm_summary.md`  |Compressed LLM-ready summary                   |Inject into downstream agent system prompt|
|`data_profile_alias_graph.json`|Column → concept map                           |Query routing                             |

## LLM Configuration

Set in `.env`:

```
LLM_PROVIDER=anthropic          # or "openai"
ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...
```

Optional overrides: `LLM_MODEL`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`

## Supported Sources

|Type                        |`source_type` value|Required params                  |
|----------------------------|-------------------|---------------------------------|
|CSV                         |`"csv"`            |`path`                           |
|TSV                         |`"tsv"`            |`path`                           |
|Excel (.xlsx/.xls/.ods)     |`"excel"`          |`path`, optionally `sheet_name`  |
|SQL (any SQLAlchemy dialect)|`"sql"`            |`connection_string`, `table_name`|

## Size Handling

|File size     |Strategy                                |
|--------------|----------------------------------------|
|< 50 MB       |Full load                               |
|50–500 MB     |Systematic 1-in-N row sample → ~50k rows|
|> 500 MB      |Head sample → 50k rows                  |
|SQL > 50k rows|TABLESAMPLE or LIMIT fallback           |

All sampled profiles are clearly flagged in output.