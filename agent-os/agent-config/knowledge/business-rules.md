# Business Rules

These rules govern how AgentOS interprets and validates data. The Orchestrator passes this file to every agent's context.

---

## Fiscal Calendar

- **Fiscal year**: October 1 — September 30
- **FY label**: the year of the *September* end-month. FY2024 = Oct 2023 → Sep 2024.

### Fiscal Quarters
| Fiscal Q | Calendar Months |
|---|---|
| Q1 | Oct, Nov, Dec |
| Q2 | Jan, Feb, Mar |
| Q3 | Apr, May, Jun |
| Q4 | Jul, Aug, Sep |

### Reporting Period Cutoffs
- Monthly reports: due by 5th business day of the following month
- Quarterly: due by 15th business day after quarter-end
- Annual: due by Nov 30 each year (60 days after FY end)

---

## Expense Approval Thresholds

| Amount Range (USD) | Approval Required |
|---|---|
| < $500 | Manager |
| $500 – $2,500 | Department Head |
| $2,500 – $10,000 | VP |
| $10,000 – $50,000 | CFO |
| > $50,000 | CEO + Board notification |

**Auto-reject**: any expense submitted >90 days after the transaction date.

---

## Budget Variance Rules

| Variance | Action |
|---|---|
| ≤ ±5% | Within tolerance — no action |
| ±5% to ±10% | Flag for review (`warning`) |
| ±10% to ±20% | Mandatory review by department head (`critical`) |
| > ±20% | Escalate to CFO (`critical`, requires explanation) |

Variance formula: `(actual - budget) / abs(budget) * 100`

---

## Currency Conversion

All amounts in non-USD currency must be converted to USD using the exchange rate on the **transaction date** (not the report date). Use end-of-month rates if exact-day rates are unavailable.

When converting:
1. Store the original amount and currency
2. Store the converted USD amount in a separate column
3. Record the FX rate used and the rate date
4. Never silently overwrite the original

---

## Expense Categorization Rules

### Travel
- Flight tickets, train, ground transport
- Includes baggage fees but NOT in-flight purchases (those = `meals`)

### Lodging
- Hotel rooms, Airbnb, room tax
- Excludes hotel meals (those = `meals`)
- Excludes hotel parking (that = `travel`)

### Meals
- Business meals with clients (must list attendees in description)
- Per-diem meals during travel
- Subject to IRS 50% deduction rule — flag total for tax reporting

### Tools
- Software subscriptions ≤ $5k/year per license
- Anything > $5k/year goes through procurement, not expense

### Hardware
- Equipment purchases (laptops, monitors)
- Must be capitalized if individual cost > $5,000

### Training
- Courses, conferences, books, certifications
- Pre-approval required for any item > $1,000

### Capital vs. Operating Expense (CapEx vs OpEx)
- Items with useful life > 1 year AND cost > $5,000 = CapEx, not OpEx
- These must be tagged for the asset register, not the expense ledger

---

## Cost Center Hierarchy

```
Company
├── 100 — Engineering
│   ├── 101 — Frontend
│   ├── 102 — Backend
│   ├── 103 — Infrastructure
│   └── 104 — Data Platform
├── 200 — Sales
│   ├── 201 — Enterprise
│   ├── 202 — Mid-Market
│   └── 203 — SMB
├── 300 — Marketing
│   ├── 301 — Demand Gen
│   ├── 302 — Brand
│   └── 303 — Content
├── 400 — Operations
│   ├── 401 — Customer Success
│   ├── 402 — Support
│   └── 403 — Facilities
└── 500 — G&A
    ├── 501 — Finance
    ├── 502 — HR
    └── 503 — Legal
```

Every expense must roll up to a leaf-level cost center (3-digit code). Reports aggregate at the 1-digit level by default.

---

## Data Quality Rules

### Hard rules (auto-reject row)
- `amount` is negative when category is not in the refund/credit list
- `date` is in the future
- `currency` is not in the supported list
- `cost_center` does not exist in the hierarchy
- `employee_id` does not exist in the employee directory

### Soft rules (flag but don't reject)
- Duplicate `(vendor, amount, date)` within 7-day window — possible double-submission
- Amount > 3× the 90-day rolling median for this category — possible outlier
- Description contains no spaces and is < 4 chars — possibly meaningless
- Transaction on weekend or public holiday — flag for review

---

## Privacy & PII Rules

- Employee `email` and `phone` may appear in raw data but must NOT appear in reports or chart legends
- Employee compensation (`salary`, `bonus`) is restricted — only HR & Finance users may view
- Customer names in expense memos must be redacted unless the user has client-data permission

---

## Period-over-Period Comparisons

- Always compare like-to-like periods: MoM = same month previous year, QoQ = same quarter previous year, YoY = same calendar position previous year
- Adjust for working-day count when comparing months (e.g. February has fewer days)
- Don't compare periods of unequal length without normalization

---

## Materiality Thresholds

For "key findings" in reports:
- Item must represent ≥ 1% of total reporting amount OR
- Item must exceed $10,000 absolute OR
- Item must represent a >10% change vs. comparison period

Below these thresholds, group items into "Other" rather than calling them out individually.

---

## Tie-Breaking for Conflicting Data

When the same logical record appears in multiple files with different values:
1. **Most recent** `updated_at` timestamp wins
2. **Otherwise** the file from the higher-precedence source wins (HRIS > expense system > spreadsheet)
3. **Always log the discrepancy** so a human can review

---

## Auto-Escalation Triggers

The Validation Agent escalates to a human reviewer (sets job status to `needs_review`) when:
- Any check has `severity: critical` AND `passed: false`
- Total row count differs from source by > 5%
- Sum of amounts differs from source by > $1,000 (absolute) or 1% (relative)
- Any negative amount appears in a category not on the refund list
- More than 10% of rows are flagged as anomalies
