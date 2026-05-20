# Skill: Financial Formula Translation

## Purpose
Translate Excel financial formulas into accurate Python equivalents using pandas, numpy, and DuckDB. Every translation must produce numerically identical output to the original Excel formula (within floating-point tolerance).

---

## Lookup Functions

### VLOOKUP / XLOOKUP → `pd.merge()`
```python
# =VLOOKUP(A2, Departments, 2, FALSE)
merged = main_df.merge(lookup_df[['key', 'value']], left_on='A', right_on='key', how='left')
```

### INDEX/MATCH → `pd.merge()` or `.map()`
```python
# =INDEX(values, MATCH(A2, keys, 0))
main_df['result'] = main_df['key_col'].map(lookup_df.set_index('key')['value'])
```

---

## Conditional Sums

### SUMIF → `df[condition][col].sum()` or `groupby().sum()`
```python
# =SUMIF(A:A, "ENG", B:B)
total = df.loc[df['A'] == 'ENG', 'B'].sum()

# =SUMIF(A:A, ">1000", B:B)
total = df.loc[df['A'] > 1000, 'B'].sum()
```

### SUMIFS → multi-condition boolean
```python
# =SUMIFS(C:C, A:A, "ENG", B:B, "Q1")
total = df.loc[(df['A'] == 'ENG') & (df['B'] == 'Q1'), 'C'].sum()
```

### COUNTIF / COUNTIFS
```python
# =COUNTIF(A:A, "ENG")
count = (df['A'] == 'ENG').sum()

# =COUNTIFS(A:A, "ENG", B:B, ">100")
count = ((df['A'] == 'ENG') & (df['B'] > 100)).sum()
```

### AVERAGEIF
```python
# =AVERAGEIF(A:A, "ENG", B:B)
avg = df.loc[df['A'] == 'ENG', 'B'].mean()
```

---

## Conditional Logic

### IF → `np.where()`
```python
# =IF(A2 > 1000, "high", "low")
df['flag'] = np.where(df['A'] > 1000, 'high', 'low')

# Nested =IF(A2 > 1000, "high", IF(A2 > 500, "mid", "low"))
df['flag'] = np.select(
    [df['A'] > 1000, df['A'] > 500],
    ['high', 'mid'],
    default='low'
)
```

### IFS → `np.select()`
```python
df['category'] = np.select(
    condlist=[df['amt'] < 100, df['amt'] < 1000, df['amt'] < 10000],
    choicelist=['small', 'medium', 'large'],
    default='huge'
)
```

### IFERROR → `try/except` or `.fillna()`
```python
# =IFERROR(A2/B2, 0)
df['ratio'] = (df['A'] / df['B']).replace([np.inf, -np.inf], np.nan).fillna(0)
```

---

## Rounding

| Excel | Python |
|---|---|
| `=ROUND(A, 2)` | `df['A'].round(2)` |
| `=ROUNDUP(A, 0)` | `np.ceil(df['A']).astype(int)` |
| `=ROUNDDOWN(A, 0)` | `np.floor(df['A']).astype(int)` |
| `=ROUND(A, -2)` | `(df['A'] / 100).round() * 100` |
| `=MROUND(A, 5)` | `(df['A'] / 5).round() * 5` |

---

## Dates

| Excel | Python |
|---|---|
| `=TODAY()` | `pd.Timestamp.today().normalize()` |
| `=NOW()` | `pd.Timestamp.now()` |
| `=EOMONTH(A, 0)` | `df['A'].dt.to_period('M').dt.to_timestamp('M').dt.normalize() + pd.offsets.MonthEnd(0)` |
| `=EOMONTH(A, 1)` | `df['A'] + pd.offsets.MonthEnd(1)` |
| `=YEAR(A)` | `df['A'].dt.year` |
| `=MONTH(A)` | `df['A'].dt.month` |
| `=DAY(A)` | `df['A'].dt.day` |
| `=WEEKDAY(A)` | `df['A'].dt.dayofweek` |
| `=NETWORKDAYS(A, B)` | `np.busday_count(df['A'].values.astype('datetime64[D]'), df['B'].values.astype('datetime64[D]'))` |
| `=DATEDIF(A, B, "d")` | `(df['B'] - df['A']).dt.days` |
| `=DATEDIF(A, B, "m")` | `(df['B'].dt.year - df['A'].dt.year) * 12 + (df['B'].dt.month - df['A'].dt.month)` |

---

## Running Totals & Cumulatives

```python
# Excel: =SUM($B$2:B2)  (drag down)
df['running_total'] = df['B'].cumsum()

# Reset cumsum per group
df['team_running'] = df.groupby('team')['amount'].cumsum()

# Running average
df['running_avg'] = df['B'].expanding().mean()

# 3-month rolling
df['ma3'] = df['B'].rolling(window=3).mean()
```

---

## Pivot Tables

```python
# Excel pivot: rows=team, cols=quarter, values=expense, agg=sum
pivot = pd.pivot_table(
    df,
    index='team',
    columns='quarter',
    values='expense',
    aggfunc='sum',
    fill_value=0,
    margins=True,
    margins_name='Total',
)
```

---

## Financial Functions

### NPV
```python
# Excel: =NPV(rate, cashflows...)
def npv(rate: float, cashflows: list[float]) -> float:
    return sum(cf / (1 + rate) ** (i + 1) for i, cf in enumerate(cashflows))
```

### IRR
```python
from scipy.optimize import brentq
def irr(cashflows: list[float]) -> float:
    f = lambda r: sum(cf / (1 + r) ** i for i, cf in enumerate(cashflows))
    return brentq(f, -0.99, 10.0)
```

### PMT (Loan Payment)
```python
# =PMT(rate, nper, pv)
def pmt(rate: float, nper: int, pv: float) -> float:
    return -(rate * pv) / (1 - (1 + rate) ** -nper) if rate else -pv / nper
```

---

## Accuracy Checklist
Before submitting transformed data, verify:
1. ✅ Row count matches original (or expected difference is documented)
2. ✅ Sum of a numeric column matches Excel within $0.01 tolerance
3. ✅ NaN counts don't increase unless explicitly imputed
4. ✅ Date columns are dtype `datetime64[ns]`, not strings
5. ✅ Currency columns are float64, not strings with `$` prefix
