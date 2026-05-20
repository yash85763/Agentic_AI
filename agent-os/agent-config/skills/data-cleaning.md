# Skill: Data Cleaning

## Purpose
Normalize messy real-world data into clean, analyzable form. Apply consistently across all files so downstream merge operations align cleanly.

---

## Column Name Normalization

```python
def normalize_columns(df):
    df.columns = (
        df.columns
        .str.strip()                               # remove leading/trailing whitespace
        .str.lower()                               # case-insensitive
        .str.replace(r'[^\w\s]', '', regex=True)   # drop punctuation
        .str.replace(r'\s+', '_', regex=True)      # spaces → underscore
        .str.replace(r'_+', '_', regex=True)       # collapse repeats
        .str.strip('_')
    )
    return df
```

Examples:
- `"  Employee ID "` → `"employee_id"`
- `"$Amount (USD)"` → `"amount_usd"`
- `"Q1-2024 Total"` → `"q1_2024_total"`

---

## Type Coercion

### Currency Strings → Float
```python
def parse_currency(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    # Remove currency symbols and thousands separators
    s = re.sub(r'[$£€¥,]', '', s)
    # Handle parens for negatives: "(1,234)" → -1234
    if s.startswith('(') and s.endswith(')'):
        s = '-' + s[1:-1]
    return float(s) if s else np.nan

df['amount'] = df['amount'].apply(parse_currency)
```

### Date Strings → datetime64
```python
df['date'] = pd.to_datetime(df['date'], errors='coerce', infer_datetime_format=True)

# Multiple possible formats:
for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%d-%b-%Y'):
    mask = df['date'].isna()
    if not mask.any():
        break
    df.loc[mask, 'date'] = pd.to_datetime(df.loc[mask, 'date_raw'], format=fmt, errors='coerce')
```

### Percentage Strings → Float
```python
def parse_pct(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip().rstrip('%')
    return float(s) / 100 if s else np.nan
```

### Boolean from "Y/N", "Yes/No", "1/0"
```python
truthy = {'y', 'yes', 'true', '1', 't'}
df['is_active'] = df['is_active'].astype(str).str.lower().str.strip().isin(truthy)
```

---

## Deduplication

### Exact Duplicates
```python
before = len(df)
df = df.drop_duplicates()
print(f"Removed {before - len(df)} exact duplicates")
```

### Duplicates on Key Columns (keep latest)
```python
df = df.sort_values('updated_at').drop_duplicates(subset=['employee_id'], keep='last')
```

### Fuzzy Duplicates (manual confirmation required)
Don't auto-drop fuzzy matches. Flag them for human review:
```python
df['name_normalized'] = df['name'].str.lower().str.strip()
dupes = df[df.duplicated(subset=['name_normalized'], keep=False)]
print(f"Possible fuzzy duplicates: {len(dupes)} rows — review manually")
```

---

## Missing Value Handling

| Column type | Default rule |
|---|---|
| Numeric (transactional) | Leave as NaN — don't impute |
| Numeric (categorical key) | Drop the row — invalid |
| Currency / amount | Replace with `0.0` if business rule says "no entry = no spend" |
| Date | Leave as NaT — don't impute |
| Categorical | Replace with `"Unknown"` |
| Text descriptions | Leave as NaN, downstream code handles |

**Critical rule**: Never silently impute missing values. Always log the action:
```python
missing_count = df['team'].isna().sum()
if missing_count:
    print(f"WARNING: filled {missing_count} missing teams with 'Unknown'")
    df['team'] = df['team'].fillna('Unknown')
```

---

## Outlier Detection

### Univariate (>3σ)
```python
mean = df['amount'].mean()
std = df['amount'].std()
df['outlier_z'] = ((df['amount'] - mean) / std).abs() > 3
```

### IQR-based
```python
q1, q3 = df['amount'].quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
df['outlier_iqr'] = ~df['amount'].between(lower, upper)
```

### Domain-specific thresholds
Always prefer business-rule thresholds over statistical ones when available:
```python
# From business-rules.md: any expense over $50k requires escalation
df['needs_escalation'] = df['amount'] > 50000
```

---

## Cross-File Consistency Checks

When merging multiple files, verify:
1. **Same column types**: `df_a['date'].dtype == df_b['date'].dtype`
2. **Same value sets**: `set(df_a['team']) == set(df_b['team'])`
3. **No orphan IDs**: every `df_b['team']` exists in `df_a['team']`
4. **Date ranges align**: no rows from outside the expected period
5. **No double-counting**: same `(team, period, category)` doesn't appear twice across files

---

## Encoding Issues

### Detect non-UTF-8 files
```python
import chardet
with open(path, 'rb') as f:
    detection = chardet.detect(f.read(10000))
encoding = detection['encoding']  # often 'cp1252' or 'iso-8859-1'
df = pd.read_csv(path, encoding=encoding)
```

### Strip BOM
```python
df.columns = df.columns.str.replace('﻿', '')
```

---

## Whitespace & Special Characters

```python
# Trim string columns
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str).str.strip()

# Remove control characters
df['notes'] = df['notes'].str.replace(r'[\x00-\x1f\x7f]', '', regex=True)
```

---

## Cleaning Pipeline Order

Run cleaning operations in this exact order to avoid downstream issues:
1. Drop fully empty rows (`df.dropna(how='all')`)
2. Normalize column names
3. Strip whitespace from string columns
4. Type coercion (dates, currency, percentages)
5. Drop exact duplicates
6. Validate against data dictionary (resolve aliases)
7. Handle missing values per the rules table above
8. Flag outliers (don't drop)
9. Cross-file consistency check (if merging)

---

## Logging Every Cleaning Action

Always record what changed so the Memory Agent can learn:
```python
cleaning_log = {
    'rows_in': len(df_raw),
    'rows_out': len(df_clean),
    'columns_renamed': dict(zip(df_raw.columns, df_clean.columns)),
    'duplicates_removed': len(df_raw) - len(df_clean.drop_duplicates()),
    'nulls_imputed': {col: df_raw[col].isna().sum() for col in df_clean.columns},
    'outliers_flagged': df_clean.get('outlier_z', pd.Series()).sum(),
}
```
