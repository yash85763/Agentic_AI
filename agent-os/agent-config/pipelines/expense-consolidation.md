# Pipeline: Expense Consolidation

## Purpose
Consolidate per-team monthly expense reports into a single, validated dataset with cross-team summaries, anomaly detection, and an executive narrative.

---

## Trigger Conditions

This pipeline runs when:
- User uploads ≥ 2 Excel/CSV files via the Workspace, OR
- User selects "expense consolidation" template, OR
- Scheduled monthly trigger fires (1st business day of each month)

---

## Inputs

- 2+ Excel/CSV files, one per team or cost center
- Each file expected to contain expense line items with columns:
  - `date`, `amount`, `category`, `description`, plus team/cost center identifier
- Optional: prior period file for variance comparison

---

## DAG

```
       ┌─────────────┐
       │ orchestrate │  Plan → break work into per-team transformations
       └──────┬──────┘
              │
       ┌──────▼──────┐
       │   ingest    │  Parallel: read each file, extract schema
       └──────┬──────┘
              │
       ┌──────▼──────┐
       │  understand │  Resolve columns vs. data-dictionary, validate types
       └──────┬──────┘
              │
       ┌──────▼──────┐
       │  transform  │  Parallel: clean + aggregate per file (sandbox)
       └──────┬──────┘
              │
       ┌──────▼──────┐
       │    merge    │  Combine team parquets into single dataset
       └──────┬──────┘
              │
       ┌──────▼──────┐
       │  validate   │  Cross-check totals, flag anomalies (sandbox)
       └──────┬──────┘
              │
       ┌──────▼──────┐
       │  visualize  │  Generate ECharts configs (5 charts default)
       └──────┬──────┘
              │
       ┌──────▼──────┐
       │   report    │  Executive summary + sections + Excel export
       └──────┬──────┘
              │
       ┌──────▼──────┐
       │  memorize   │  Persist schema, mappings, corrections
       └──────┬──────┘
              │
             END
```

---

## Step-by-Step

### 1. Orchestrate
- Load cognitive context (soul + skills + knowledge)
- Parse user task description
- Build PipelineState with job_id, file_ids, task
- Emit `agent_start` event

### 2. Ingest (parallel per file)
- For each file:
  - Detect format (xlsx/xls/csv)
  - Open in Docker sandbox using openpyxl/pandas
  - Classify sheets: data vs. summary vs. metadata
  - For each data sheet, extract:
    - Column names
    - First 100 rows as sample
    - Row count
    - Dtypes
  - Emit `code_generated`, `code_result` events

### 3. Understand
- For each manifest, map raw column names to canonical names:
  - Step 1: Lookup in `memory/column-mappings.json` (fast)
  - Step 2: If unmapped, prompt LLM with `data-dictionary.md` context
  - Step 3: Confirm dtype matches the canonical definition
- Validate schema completeness: every required canonical column present?
- Emit `validation` event if any file fails schema validation

### 4. Transform (parallel per file)
- Generate pandas code via WORKER_MODEL (qwen2.5-coder)
- Required transformations:
  - Apply column rename to canonical names
  - Clean values (currency, dates, percentages)
  - Apply business rules (currency conversion, fiscal calendar)
  - Aggregate by `(team, month, category)`
- Execute in Docker sandbox
- Write output parquet to `/tmp/jobs/{job_id}/team_{file_id}.parquet`

### 5. Merge
- Concatenate all team parquets
- Add `source_file_id` column for traceability
- Write to `/tmp/jobs/{job_id}/merged.parquet`

### 6. Validate
- Required checks:
  - Row count of merged = sum of source row counts
  - Sum of `amount` in merged = sum across sources (±$0.01)
  - No duplicate `(team, date, vendor, amount)` tuples
  - Every `cost_center` exists in hierarchy
  - Every `category` is in the allowed set
  - Every `currency` is in supported list
  - Apply business rule budget variance checks per team
  - Statistical: flag values >3σ from per-category median
- Output `ValidationReport` to state

### 7. Visualize
Default chart set:
1. **Bar chart**: Total expense by team (current period)
2. **Time series**: Daily / weekly trend across all teams
3. **Pie**: Spend distribution by category
4. **Treemap**: Team → Category breakdown
5. **Heatmap**: Team × Category matrix (if cells > 5×5)

Plus optional charts if data warrants:
- Budget vs. Actual bar chart per team
- Anomalies scatter (date vs. amount, colored by severity)

### 8. Report
Standard report sections:
1. **Executive Summary** — 3-5 sentences with key totals
2. **Overview** — period, scope, row count
3. **Key Findings** — 3-5 bullet points with numbers
4. **Team Breakdown** — table with totals, variance vs. budget
5. **Anomalies** — list of flagged items needing review
6. **Recommendations** — 3-5 actionable next steps

Export Excel with sheets: Summary, Sections, Data, Charts

### 9. Memorize
- Update `memory/schema-cache.json` with each file's schema fingerprint
- Update `memory/column-mappings.json` with any new alias resolutions discovered
- Don't auto-write to `corrections.md` — only humans add corrections

---

## Validation Rules (specific to this pipeline)

| Check | Threshold | Severity |
|---|---|---|
| Sum-of-sources matches merged | ±$0.01 | critical |
| Row count matches | exact | critical |
| Unknown cost center | any | critical |
| Unknown category | any | critical |
| Approval missing for amount > $500 | any | warning |
| Negative amount in non-refund category | any | warning |
| Variance vs. budget > 20% | per team | critical |
| Variance vs. budget 10-20% | per team | warning |
| Outlier > 3σ from category median | per row | info |

---

## Expected Outputs

After successful completion:
- `merged.parquet` — consolidated dataset (returned as MinIO object)
- `validation.json` — full validation report
- `charts[]` — array of 4-6 ECharts configs
- `report` — narrative dict with sections
- `report.xlsx` — Excel export with embedded data + charts
- Updated `memory/schema-cache.json` and `memory/column-mappings.json`

---

## Retry Behavior

- If `validate` reports `passed: false` AND the failures look transformation-related, route back to `transform` (max 2 retries)
- If `validate` reports `passed: false` after retries, set job status `needs_review` and end with the failed report
- If any agent throws an exception, emit `error` event and end pipeline
- Transient LLM errors retry automatically inside the agent (exponential backoff)

---

## Estimated Resource Usage

| Resource | Per Job (typical) |
|---|---|
| Docker containers | 1 per file (concurrent), then 1 each for merge + validate |
| Sandbox memory | 512MB × parallel files |
| LLM tokens | ~50k input + ~10k output |
| LLM cost (Claude + Ollama mix) | $0.10 - $0.50 |
| Wall-clock duration | 30-120 seconds depending on file size |
