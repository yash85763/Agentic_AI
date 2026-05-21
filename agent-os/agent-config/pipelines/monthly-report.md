# Pipeline: Monthly Report

## Purpose
Automated end-of-month financial report. Runs on the 1st business day of every month, consolidates the prior month's data, and distributes a polished narrative report to stakeholders.

---

## Schedule

- **Trigger**: 1st business day of each month, 06:00 UTC (Celery beat)
- **Period**: Previous calendar month
- **Override**: Manual trigger via API with `period=YYYY-MM` query parameter

---

## Inputs

Discovered automatically:
- All files in MinIO bucket `raw-files/expenses/YYYY-MM/` for the period
- Prior 3 months of consolidated data from `processed-files/` for trend comparison
- Previous month's report metadata for YoY/MoM comparison

---

## Pipeline Flow

This pipeline reuses **Expense Consolidation** under the hood, but adds:

1. **Auto-discovery** instead of explicit file upload
2. **Period filter** — only files dated in target month
3. **Comparison data** — pull prior 3 and 12 months for trend
4. **Distribution** — auto-deliver report to subscribers
5. **Snapshot persistence** — save report version-stamped

---

## Report Sections (extended vs. base consolidation)

1. **Executive Summary**
   - Total spend, headcount, key ratios
   - 3 highlights, 2 concerns

2. **Period Overview**
   - Month covered
   - Row counts
   - Data quality score

3. **Spend Analysis**
   - By team (bar chart)
   - By category (treemap)
   - Top 10 vendors (table)

4. **Trend Analysis**
   - MoM change (vs. previous month)
   - YoY change (vs. same month previous year)
   - 3-month rolling average
   - 12-month TTM

5. **Budget Performance**
   - Variance per team
   - Forecast end-of-quarter
   - YTD utilization

6. **Anomalies & Exceptions**
   - All `critical` anomalies
   - All approvals pending > 30 days
   - All vendors above materiality threshold ($10k)

7. **Recommendations**
   - 3-5 actionable items
   - Prioritized by financial impact

8. **Methodology** (footer)
   - Data sources
   - Conversion rates used
   - Assumptions

---

## Distribution

After successful completion, the report is delivered to subscribers via:

1. **Email** (if SMTP configured) — PDF attachment + summary in body
2. **Slack** (if webhook configured) — summary message with link
3. **API webhook** (per subscriber config)
4. **Persistent storage**: report saved to MinIO at `reports/YYYY-MM/report.json` and `reports/YYYY-MM/report.xlsx`

### Subscriber List

| Audience | Format | Channel |
|---|---|---|
| CFO | Excel + PDF | Email |
| Department Heads | PDF summary | Email |
| Finance Team | Excel + JSON | Slack + Email |
| Board (quarterly only) | PDF | Email |

---

## Comparison Data Requirements

For the trend section, the agent must pull:

- **t-1 month** (last month) — for MoM
- **t-12 months** (same month last year) — for YoY
- **t-1, t-2, t-3 months** — for 3-month rolling average
- **t-1 to t-12** — for TTM total

If any comparison data is missing, the section displays "Insufficient history — first month tracked" rather than failing.

---

## Validation Rules (in addition to base pipeline)

| Check | Threshold | Severity |
|---|---|---|
| Period coverage complete | All days in month | warning |
| MoM change in total | < ±25% | warning |
| MoM change in total | < ±50% | critical |
| Missing team data | any team with $0 | warning |
| New cost center appearance | any | warning (review) |

---

## Memorize Step Extras

In addition to the standard memorization, this pipeline writes:

- `memory/monthly-totals.json` — running record of each month's headline figures
- `memory/trend-baseline.json` — updated rolling baselines used for anomaly detection in subsequent runs

These help the next month's pipeline detect drift more accurately.

---

## SLAs

- **Latency**: complete within 10 minutes of trigger
- **Accuracy**: total within $1 of source-of-truth GL system
- **Availability**: skip month gracefully if upstream data is unavailable; alert ops

---

## Manual Override

Trigger an out-of-band run via the API:

```bash
curl -X POST http://localhost:8000/api/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Run monthly expense report for 2024-03",
    "pipeline": "monthly-report",
    "params": {"period": "2024-03"}
  }'
```

---

## Idempotency

Running the same month twice produces identical results (same input + same code = same output). The second run overwrites the first in MinIO storage; both runs are visible in job history.
