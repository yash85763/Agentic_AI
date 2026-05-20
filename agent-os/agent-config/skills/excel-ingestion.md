# Skill: Excel Ingestion

> How to correctly ingest Excel workbooks of arbitrary structure into clean
> pandas DataFrames ready for analysis.

---

## 1. Opening a Workbook

Always use `openpyxl` as the engine for `.xlsx` files and `xlrd` for legacy `.xls`.

```python
import pandas as pd
import openpyxl

# Inspect sheet names before reading
wb = openpyxl.load_workbook("file.xlsx", read_only=True, data_only=True)
sheet_names = wb.sheetnames
print(sheet_names)
wb.close()

# Read a specific sheet with pandas
df = pd.read_excel(
    "file.xlsx",
    sheet_name="Revenue",
    engine="openpyxl",
    header=None,   # defer header detection — we do it manually
)
```

---

## 2. Sheet Classification

Before processing, classify each sheet to determine handling:

| Class | Description | Action |
|-------|-------------|--------|
| `data` | Contains row-level transactional or detail records | Ingest fully |
| `summary` | Aggregated totals, pivot-style | Ingest for validation cross-checks |
| `metadata` | File info, period, owner, version | Extract key-value pairs |
| `ignored` | Charts-only, cover page, notes, empty | Skip |

### Classification Heuristics

```python
def classify_sheet(ws) -> str:
    """Classify an openpyxl worksheet."""
    max_row = ws.max_row or 0
    max_col = ws.max_column or 0

    # Empty sheet
    if max_row < 2 or max_col < 1:
        return "ignored"

    # Count non-null cells in first 10 rows
    sample_values = []
    for row in ws.iter_rows(min_row=1, max_row=min(10, max_row), values_only=True):
        sample_values.extend([v for v in row if v is not None])

    if not sample_values:
        return "ignored"

    # Metadata sheets are small (< 20 rows, < 5 cols) with label-value pairs
    if max_row < 20 and max_col <= 4:
        string_count = sum(1 for v in sample_values if isinstance(v, str))
        if string_count / len(sample_values) > 0.6:
            return "metadata"

    # Summary sheets often have "Total", "Grand Total", "Subtotal" in first col
    first_col_values = [
        str(ws.cell(r, 1).value or "").lower()
        for r in range(1, min(20, max_row) + 1)
    ]
    summary_keywords = {"total", "grand total", "subtotal", "summary"}
    if any(any(kw in v for kw in summary_keywords) for v in first_col_values):
        return "summary"

    # Default: treat as data
    return "data"
```

---

## 3. Detecting Header Rows

Headers are not always on row 1. Common patterns:
- Row 1–3: company name / report title / file metadata
- Row 4 or 5: actual column headers
- Sometimes preceded by blank rows

### Detection Algorithm

```python
import numpy as np

def detect_header_row(df_raw: pd.DataFrame) -> int:
    """
    Detect the 0-indexed row number that contains column headers.
    Strategy: find first row where:
    - Most values are non-null strings
    - Values look like labels (not dates or numbers)
    - The next row has a different dtype pattern
    """
    for i, row in df_raw.iterrows():
        non_null = row.dropna()
        if len(non_null) < 2:
            continue
        str_fraction = sum(isinstance(v, str) for v in non_null) / len(non_null)
        num_fraction = sum(isinstance(v, (int, float)) for v in non_null) / len(non_null)

        # Good header: mostly strings, few numbers
        if str_fraction >= 0.6 and num_fraction <= 0.2:
            # Check that the following row has numeric data
            if i + 1 < len(df_raw):
                next_row = df_raw.iloc[i + 1].dropna()
                next_num = sum(isinstance(v, (int, float)) for v in next_row)
                if next_num / max(len(next_row), 1) >= 0.3:
                    return i
    return 0  # fallback to first row


def read_with_detected_header(path: str, sheet: str) -> pd.DataFrame:
    df_raw = pd.read_excel(path, sheet_name=sheet, header=None, engine="openpyxl")
    header_row = detect_header_row(df_raw)
    df = pd.read_excel(
        path,
        sheet_name=sheet,
        header=header_row,
        engine="openpyxl",
    )
    return df
```

---

## 4. Handling Merged Cells

Merged cells in Excel appear as a value in the top-left cell and `None` in all
other cells of the merge region. `openpyxl` does not auto-unmerge; you must do it.

```python
def unmerge_and_fill(path: str, sheet_name: str) -> pd.DataFrame:
    """Read a sheet, unmerge cells by forward-filling, return DataFrame."""
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[sheet_name]

    # Unmerge all merged cell regions and forward-fill their values
    for merge_range in list(ws.merged_cells.ranges):
        min_row, min_col = merge_range.min_row, merge_range.min_col
        fill_value = ws.cell(min_row, min_col).value
        ws.unmerge_cells(str(merge_range))
        for row in ws.iter_rows(
            min_row=merge_range.min_row, max_row=merge_range.max_row,
            min_col=merge_range.min_col, max_col=merge_range.max_col,
        ):
            for cell in row:
                if cell.value is None:
                    cell.value = fill_value

    # Convert to DataFrame
    data = list(ws.values)
    wb.close()

    if not data:
        return pd.DataFrame()

    headers = data[0]
    return pd.DataFrame(data[1:], columns=headers)
```

---

## 5. Recognizing Table Boundaries

Excel files often contain multiple tables on the same sheet separated by blank
rows or rows containing only labels.

```python
def find_table_boundaries(df_raw: pd.DataFrame) -> list[tuple[int, int]]:
    """
    Return list of (start_row, end_row) tuples for each distinct data table.
    A table boundary is detected by a row where ALL values are null.
    """
    boundaries = []
    in_table = False
    start = 0

    for i in range(len(df_raw)):
        row_all_null = df_raw.iloc[i].isna().all()
        if not in_table and not row_all_null:
            start = i
            in_table = True
        elif in_table and row_all_null:
            boundaries.append((start, i - 1))
            in_table = False

    if in_table:
        boundaries.append((start, len(df_raw) - 1))

    return boundaries
```

---

## 6. Multi-Sheet Workbook Strategy

```python
def ingest_workbook(path: str) -> dict[str, pd.DataFrame]:
    """
    Ingest all data sheets from an Excel workbook.
    Returns dict of {sheet_name: DataFrame}.
    """
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    result = {}

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        classification = classify_sheet(ws)

        if classification == "ignored":
            print(f"  Sheet '{sheet_name}': ignored")
            continue

        print(f"  Sheet '{sheet_name}': {classification}")
        df = unmerge_and_fill(path, sheet_name)
        df = read_with_detected_header_from_df(df)

        if classification == "metadata":
            result[f"_meta_{sheet_name}"] = df
        else:
            result[sheet_name] = df

    wb.close()
    return result
```

---

## 7. Common Excel Anti-Patterns

### 7.1 Totals Rows Embedded in Data

Many Excel files include sub-total rows interspersed with data rows.
These must be excluded from analysis to avoid double-counting.

```python
def remove_total_rows(df: pd.DataFrame, first_col: str) -> pd.DataFrame:
    """Remove rows where the first column contains total/subtotal keywords."""
    keywords = r"(?i)^(total|sub.?total|grand\s+total|sum|net\s+total)"
    mask = df[first_col].astype(str).str.match(keywords)
    total_count = mask.sum()
    if total_count > 0:
        print(f"  Removed {total_count} total/subtotal rows from '{first_col}'")
    return df[~mask].reset_index(drop=True)
```

### 7.2 Mixed Data Types in a Column

Excel allows a column to contain both numbers and strings (e.g., "N/A", "TBD").

```python
def coerce_numeric_column(series: pd.Series) -> pd.Series:
    """Attempt numeric coercion; non-numeric values become NaN."""
    return pd.to_numeric(series, errors="coerce")
```

### 7.3 Currency Strings

Values like `"$1,234.56"` or `"(500.00)"` (negative in accounting notation).

```python
import re

def parse_currency(value) -> float | None:
    if pd.isna(value):
        return None
    s = str(value).strip()
    # Remove currency symbols and commas
    s = re.sub(r"[$£€,\s]", "", s)
    # Handle accounting negative: (1234.56) -> -1234.56
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except ValueError:
        return None

def parse_currency_column(series: pd.Series) -> pd.Series:
    return series.map(parse_currency)
```

### 7.4 Percentage Strings

```python
def parse_percentage(value) -> float | None:
    """Convert '12.5%' or 0.125 (Excel fraction) to decimal fraction."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        # Excel stores % as fraction if cell is formatted as %
        return float(value)
    s = str(value).strip().rstrip("%")
    try:
        return float(s) / 100
    except ValueError:
        return None
```

### 7.5 Inconsistent Date Formats

```python
from dateutil import parser as dateutil_parser

def parse_date_flexible(value) -> pd.Timestamp | None:
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value
    if isinstance(value, float):
        # Excel serial date number
        return pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(value))
    try:
        return pd.Timestamp(dateutil_parser.parse(str(value)))
    except Exception:
        return None
```

### 7.6 Trailing Whitespace in Column Names

```python
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, lowercase, convert spaces to underscores."""
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    return df
```

---

## 8. Validation After Ingestion

After loading every sheet, run these checks:

```python
def validate_ingested_df(df: pd.DataFrame, sheet_name: str) -> dict:
    issues = []

    # Check for completely empty DataFrame
    if df.empty:
        issues.append(f"Sheet '{sheet_name}' produced empty DataFrame")

    # Check for duplicate column names
    dupe_cols = df.columns[df.columns.duplicated()].tolist()
    if dupe_cols:
        issues.append(f"Duplicate column names: {dupe_cols}")

    # Report null rates per column
    null_rates = (df.isna().sum() / len(df) * 100).round(1)
    high_null = null_rates[null_rates > 80].to_dict()
    if high_null:
        issues.append(f"High null rate columns (>80%): {high_null}")

    return {
        "sheet": sheet_name,
        "rows": len(df),
        "columns": len(df.columns),
        "null_rates": null_rates.to_dict(),
        "issues": issues,
    }
```

---

*Last updated: 2026-05-20*
