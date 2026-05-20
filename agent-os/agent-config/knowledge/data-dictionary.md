# Data Dictionary

This file is the **canonical reference** for column names, aliases, and valid value sets across all data files processed by AgentOS. When the Understanding Agent resolves a raw column name from an uploaded file, it consults this dictionary to map it to a canonical name.

---

## Canonical Column Names & Aliases

| Canonical | Aliases | Type | Description |
|---|---|---|---|
| `employee_id` | emp_id, employeeid, staff_id, person_id | string | Unique employee identifier |
| `employee_name` | name, full_name, employee, staff_name | string | Full name |
| `department` | dept, dept_name, division | string | Department (see Team Directory) |
| `cost_center` | cc, cc_code, cost_ctr, costcenter | string | Cost center code (e.g. CC-101) |
| `team` | team_name, group, squad | string | Team within department |
| `manager` | mgr, manager_name, supervisor, line_manager | string | Direct manager |
| `email` | email_address, contact_email | string | Work email |
| `hire_date` | start_date, doh, date_hired | date | Date of hire |
| `termination_date` | end_date, dot, exit_date | date | Date of termination (if applicable) |
| `is_active` | active, status_active, employed | bool | Currently employed (Y/N) |
| `headcount` | hc, fte, fte_count | float | Headcount (may be fractional for part-time) |
| `salary` | base_salary, base_pay, annual_salary | float | Annual base salary (USD) |
| `currency` | ccy, curr | string | ISO 4217 currency code (USD, EUR, GBP) |
| `amount` | amt, value, total, sum, expense, spend, cost | float | Monetary amount in `currency` |
| `quantity` | qty, count, units | int | Item count |
| `unit_price` | price, unit_cost, rate | float | Per-unit price |
| `date` | transaction_date, txn_date, posted_date | date | Primary date of record |
| `month` | period_month, mo | string | Month in YYYY-MM format |
| `quarter` | qtr, period_quarter, q | string | Fiscal quarter (Q1-Q4) |
| `year` | yr, fy, fiscal_year | int | Fiscal year |
| `category` | cat, type, expense_type, transaction_type | string | Expense or transaction category |
| `subcategory` | subcat, sub_type | string | Finer-grained category |
| `description` | desc, memo, notes, comment | string | Free-text description |
| `vendor` | supplier, payee, merchant | string | Vendor name |
| `invoice_number` | inv_no, invoice_no, invoice_id | string | Invoice reference |
| `purchase_order` | po, po_number, po_id | string | Purchase order reference |
| `approval_status` | approved, status, approval | string | pending / approved / rejected |
| `approver` | approved_by, approval_manager | string | Who approved |
| `revenue` | rev, sales, income, turnover | float | Top-line revenue (USD) |
| `cogs` | cost_of_goods, cost_of_sales | float | Cost of goods sold |
| `gross_margin` | gm, gross_profit | float | Revenue minus COGS |
| `ebitda` | operating_profit | float | EBITDA (USD) |
| `budget` | budgeted_amount, budget_amt, planned | float | Budgeted amount for period |
| `actual` | actuals, actual_amount, spent | float | Actual amount for period |
| `variance` | var, diff, delta | float | Actual minus budget |
| `variance_pct` | var_pct, variance_percent, % variance | float | Variance as percentage |
| `region` | location, geo, market, country | string | Geographic region |
| `product` | sku, product_name, item, item_code | string | Product identifier |

---

## Valid Value Sets

### Departments
- `ENG` — Engineering
- `SALES` — Sales
- `MKTG` — Marketing
- `OPS` — Operations
- `FIN` — Finance
- `HR` — Human Resources
- `LEGAL` — Legal & Compliance

### Quarters
`Q1`, `Q2`, `Q3`, `Q4` — fiscal quarters (Oct = Q1, Jan = Q2, etc. — see business-rules.md)

### Expense Categories
- `travel` — flights, hotels, ground transport
- `lodging` — overnight stays
- `meals` — business meals & entertainment
- `tools` — software licenses, SaaS subscriptions
- `hardware` — equipment, devices
- `office` — office supplies, furniture
- `training` — courses, conferences, books
- `consulting` — external advisors
- `legal` — legal fees
- `marketing` — campaigns, ads, content
- `other` — anything else (requires description)

### Approval Status
- `pending`
- `approved`
- `rejected`
- `expired` — pending too long

### Currency Codes (ISO 4217)
`USD` (default), `EUR`, `GBP`, `JPY`, `CAD`, `AUD`

---

## Date Format Standards

All dates must be parsed to ISO format `YYYY-MM-DD` internally.

Common input formats to accept:
- `YYYY-MM-DD` (preferred)
- `MM/DD/YYYY` (US)
- `DD/MM/YYYY` (EU — flag ambiguity with US format)
- `DD-Mon-YYYY` (e.g. `15-Jan-2024`)
- Excel serial date (e.g. `45316`) — convert via `pd.to_datetime(x, unit='D', origin='1899-12-30')`

---

## Currency Precision Rules

- Display: 2 decimal places for `USD`, `EUR`, `GBP`, `CAD`, `AUD`
- Display: 0 decimal places for `JPY`
- Storage: 4 decimal places internally to avoid rounding in aggregations
- Always store the source currency in a separate column when converting

---

## Required vs Optional Fields

### Always Required
- `employee_id` OR `vendor` (one of)
- `amount` AND `currency`
- `date`
- `category`

### Required if Applicable
- `cost_center` — for any expense submission
- `manager` — for any employee record
- `approval_status` — for any expense > $500

### Optional (nice-to-have)
- `subcategory`
- `description`
- `invoice_number`
- `purchase_order`

When a required field is missing, the Validation Agent should flag the row as a `critical` anomaly, never silently impute.
