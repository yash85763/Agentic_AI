# Skill: DuckDB SQL Analysis

## Purpose
Use DuckDB for fast in-process SQL queries on parquet files and DataFrames. DuckDB is faster than pandas for joins, aggregations, and large data — and lets us write declarative SQL the way analysts already think.

---

## Setup

```python
import duckdb

# Connect (in-memory by default)
con = duckdb.connect(":memory:")

# Or with persisted state
con = duckdb.connect("/tmp/analysis.duckdb")
```

---

## Reading Parquet

```python
# Direct query — no table creation needed
df = con.execute("""
    SELECT team, SUM(amount) AS total
    FROM '/data/expenses.parquet'
    WHERE date >= '2024-01-01'
    GROUP BY team
    ORDER BY total DESC
""").fetch_df()

# Register a parquet directly as a view (read once, query many)
con.execute("CREATE VIEW expenses AS SELECT * FROM '/data/expenses.parquet'")
```

## Reading Multiple Parquet Files

```python
# Glob pattern: union all matching files
con.execute("""
    SELECT * FROM '/data/team_*.parquet'
""")

# With filename as a column
con.execute("""
    SELECT *, filename FROM read_parquet('/data/*.parquet', filename = true)
""")
```

## Reading CSV

```python
con.execute("""
    SELECT * FROM read_csv('/data/expenses.csv', header = true, auto_detect = true)
""")
```

---

## Common Aggregations

### Group-by with multiple metrics
```sql
SELECT
    team,
    COUNT(*) AS row_count,
    SUM(amount) AS total,
    AVG(amount) AS avg_amount,
    MIN(amount) AS min_amount,
    MAX(amount) AS max_amount,
    STDDEV(amount) AS std_amount,
    QUANTILE_CONT(amount, 0.5) AS median_amount,
FROM expenses
GROUP BY team
ORDER BY total DESC;
```

### Conditional aggregation
```sql
SELECT
    team,
    SUM(CASE WHEN category = 'travel' THEN amount ELSE 0 END) AS travel,
    SUM(CASE WHEN category = 'tools' THEN amount ELSE 0 END) AS tools,
    SUM(amount) AS total
FROM expenses
GROUP BY team;
```

### PIVOT
```sql
PIVOT expenses
ON quarter
USING SUM(amount)
GROUP BY team;
```

### UNPIVOT (wide → long)
```sql
UNPIVOT wide_table
ON q1, q2, q3, q4
INTO NAME quarter VALUE amount;
```

---

## Window Functions

### Running totals
```sql
SELECT
    date,
    team,
    amount,
    SUM(amount) OVER (PARTITION BY team ORDER BY date) AS running_total,
    AVG(amount) OVER (PARTITION BY team ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS ma3
FROM expenses;
```

### Ranking
```sql
SELECT
    team,
    department,
    amount,
    ROW_NUMBER() OVER (PARTITION BY team ORDER BY amount DESC) AS rank_in_team,
    PERCENT_RANK() OVER (ORDER BY amount) AS pct_rank
FROM expenses;
```

### Lag / Lead (period-over-period)
```sql
SELECT
    month,
    team,
    amount,
    LAG(amount) OVER (PARTITION BY team ORDER BY month) AS prev_month,
    amount - LAG(amount) OVER (PARTITION BY team ORDER BY month) AS month_over_month_change
FROM monthly_expenses;
```

---

## Date Functions

| Function | Result |
|---|---|
| `DATE_TRUNC('month', date)` | First day of month |
| `DATE_TRUNC('quarter', date)` | First day of quarter |
| `EXTRACT(YEAR FROM date)` | Year as integer |
| `EXTRACT(MONTH FROM date)` | Month 1-12 |
| `DATE_PART('week', date)` | ISO week number |
| `DATE_DIFF('day', a, b)` | Days between a and b |
| `STRFTIME(date, '%Y-%m')` | Format as string |
| `STRPTIME('2024-01-15', '%Y-%m-%d')` | Parse string to date |

### Fiscal year mapping (Oct 1 - Sep 30)
```sql
SELECT
    date,
    CASE
        WHEN EXTRACT(MONTH FROM date) >= 10 THEN EXTRACT(YEAR FROM date) + 1
        ELSE EXTRACT(YEAR FROM date)
    END AS fiscal_year,
    CASE
        WHEN EXTRACT(MONTH FROM date) IN (10, 11, 12) THEN 1
        WHEN EXTRACT(MONTH FROM date) IN (1, 2, 3) THEN 2
        WHEN EXTRACT(MONTH FROM date) IN (4, 5, 6) THEN 3
        ELSE 4
    END AS fiscal_quarter
FROM transactions;
```

---

## Joins

```sql
-- Inner join
SELECT e.*, t.budget
FROM expenses e
INNER JOIN teams t ON e.team = t.name;

-- Anti-join (rows in left, NOT in right)
SELECT *
FROM expenses e
LEFT JOIN approved_codes a ON e.cost_center = a.code
WHERE a.code IS NULL;

-- Asof join (latest record before a timestamp)
SELECT *
FROM events e
ASOF JOIN snapshots s ON e.entity = s.entity AND e.ts >= s.ts;
```

---

## Performance Tips

1. **Parquet > CSV**: 10-100× faster for large data. Always convert CSVs to parquet for repeated queries.
2. **Filter early**: push `WHERE` clauses as close to the source scan as possible.
3. **Project early**: `SELECT only_needed_cols` reduces I/O dramatically.
4. **Use views, not tables**: views avoid copying data.
5. **Partition large datasets**: `read_parquet('/data/team=*/year=*/file.parquet')` for partition pruning.

```python
# Check query plan
con.execute("EXPLAIN ANALYZE SELECT ...").fetch_df()
```

---

## DuckDB ↔ pandas Bridge

```python
# DataFrame → DuckDB table
con.register("mydf", df)
con.execute("SELECT * FROM mydf WHERE amount > 100").fetch_df()

# DuckDB result → pandas
result_df = con.execute("SELECT * FROM expenses").fetch_df()

# Arrow-zero-copy fetch (faster for large results)
result_arrow = con.execute("SELECT * FROM expenses").fetch_arrow_table()
```

---

## Safe Query Pattern (always parameterize)

```python
# Wrong — SQL injection risk if user input drives query
team = user_input
con.execute(f"SELECT * FROM expenses WHERE team = '{team}'")

# Right
con.execute("SELECT * FROM expenses WHERE team = ?", [user_input])
```

---

## Saving Results to Parquet

```sql
COPY (
    SELECT team, SUM(amount) AS total
    FROM expenses
    GROUP BY team
) TO '/out/team_totals.parquet' (FORMAT 'parquet');
```

Or from Python:
```python
con.execute(query).fetch_df().to_parquet('/out/result.parquet')
```
