# Formula Library

Domain-specific formulas the Transformation Agent applies during analysis. Each formula is implemented as a Python function that runs in the Docker sandbox.

---

## Financial Performance

### Gross Margin
```python
def gross_margin(revenue: float, cogs: float) -> float:
    if revenue == 0:
        return 0.0
    return (revenue - cogs) / revenue
```
Output as percentage; report to 2 decimal places.

### Operating Margin
```python
def operating_margin(revenue: float, opex: float) -> float:
    if revenue == 0:
        return 0.0
    return (revenue - opex) / revenue
```

### Net Margin
```python
def net_margin(revenue: float, net_income: float) -> float:
    if revenue == 0:
        return 0.0
    return net_income / revenue
```

### EBITDA
```python
def ebitda(net_income: float, interest: float, tax: float, depreciation: float, amortization: float) -> float:
    return net_income + interest + tax + depreciation + amortization
```

---

## Growth Rates

### YoY Growth
```python
def yoy_growth(current: float, prior: float) -> float:
    if prior == 0:
        return float('inf') if current > 0 else 0.0
    return (current - prior) / prior
```

### CAGR (Compound Annual Growth Rate)
```python
def cagr(start_value: float, end_value: float, years: int) -> float:
    if start_value <= 0 or years <= 0:
        return 0.0
    return (end_value / start_value) ** (1 / years) - 1
```

### Month-over-Month Run Rate
```python
def annualized_run_rate(latest_month_value: float) -> float:
    return latest_month_value * 12
```

---

## Headcount & Payroll

### Fully-Loaded Cost per Head
A common "fully-loaded" multiplier captures the true cost of an employee beyond base salary.

```python
def fully_loaded_cost(base_salary: float, multiplier: float = 1.35) -> float:
    """
    Multiplier of 1.35 = +35% to cover:
    - Payroll taxes (~7.65% in US)
    - Benefits (~15-20%)
    - Equipment, software, training (~5-10%)
    - Office space, utilities (~5-10%)
    """
    return base_salary * multiplier
```

Use industry-appropriate multiplier:
- US tech: 1.30-1.40
- US finance: 1.40-1.60
- EU: 1.25-1.35 (different tax structure)

### Headcount-Adjusted Productivity
```python
def revenue_per_employee(total_revenue: float, avg_headcount: float) -> float:
    if avg_headcount == 0:
        return 0.0
    return total_revenue / avg_headcount

def expense_per_employee(total_expense: float, avg_headcount: float) -> float:
    if avg_headcount == 0:
        return 0.0
    return total_expense / avg_headcount
```

### Average Headcount (for ratios)
```python
def average_headcount(start_hc: float, end_hc: float) -> float:
    return (start_hc + end_hc) / 2
```

---

## Cash Flow & Burn

### Burn Rate (monthly cash consumption)
```python
def burn_rate(cash_start: float, cash_end: float, months: int) -> float:
    if months == 0:
        return 0.0
    return (cash_start - cash_end) / months
```

### Runway (months of cash remaining)
```python
def runway_months(cash_balance: float, monthly_burn: float) -> float:
    if monthly_burn <= 0:
        return float('inf')
    return cash_balance / monthly_burn
```

### Net Burn vs Gross Burn
- **Gross burn**: monthly operating expenses (cash out)
- **Net burn**: gross burn minus monthly revenue

```python
def net_burn(operating_expenses: float, revenue: float) -> float:
    return max(0.0, operating_expenses - revenue)
```

---

## Budget Utilization

### Utilization Percentage
```python
def budget_utilization(actual: float, budget: float) -> float:
    if budget == 0:
        return 0.0
    return actual / budget
```

### Variance & Variance Percentage
```python
def variance(actual: float, budget: float) -> float:
    return actual - budget

def variance_pct(actual: float, budget: float) -> float:
    if budget == 0:
        return 0.0
    return (actual - budget) / abs(budget)
```

### Forecast End-of-Period
Project where actuals will land given current burn rate:
```python
def projected_end_of_period(
    actual_to_date: float,
    days_elapsed: int,
    days_in_period: int,
) -> float:
    if days_elapsed <= 0:
        return 0.0
    return actual_to_date * (days_in_period / days_elapsed)
```

---

## Rolling Averages

### 3-Month Rolling Average
```python
df['ma3'] = df['amount'].rolling(window=3, min_periods=1).mean()
```

### 6-Month Rolling Average
```python
df['ma6'] = df['amount'].rolling(window=6, min_periods=1).mean()
```

### Trailing Twelve Months (TTM)
```python
df['ttm'] = df['amount'].rolling(window=12, min_periods=1).sum()
```

---

## Ratios & Multiples

### Quick Ratio
```python
def quick_ratio(cash: float, receivables: float, current_liabilities: float) -> float:
    if current_liabilities == 0:
        return float('inf')
    return (cash + receivables) / current_liabilities
```

### Current Ratio
```python
def current_ratio(current_assets: float, current_liabilities: float) -> float:
    if current_liabilities == 0:
        return float('inf')
    return current_assets / current_liabilities
```

### Days Sales Outstanding (DSO)
```python
def dso(accounts_receivable: float, revenue: float, days: int = 365) -> float:
    if revenue == 0:
        return 0.0
    return (accounts_receivable / revenue) * days
```

---

## SaaS Metrics

### ARR (Annual Recurring Revenue)
```python
def arr(mrr: float) -> float:
    return mrr * 12
```

### Net Revenue Retention (NRR)
```python
def nrr(start_arr: float, expansions: float, contractions: float, churn: float) -> float:
    if start_arr == 0:
        return 0.0
    return (start_arr + expansions - contractions - churn) / start_arr
```

### Gross Revenue Retention (GRR)
```python
def grr(start_arr: float, contractions: float, churn: float) -> float:
    if start_arr == 0:
        return 0.0
    return (start_arr - contractions - churn) / start_arr
```

### Customer Acquisition Cost (CAC)
```python
def cac(sales_marketing_spend: float, new_customers: int) -> float:
    if new_customers == 0:
        return float('inf')
    return sales_marketing_spend / new_customers
```

### LTV / CAC Ratio
```python
def ltv_cac_ratio(avg_revenue_per_account: float, gross_margin_pct: float, churn_rate: float, cac_value: float) -> float:
    if cac_value == 0 or churn_rate == 0:
        return float('inf')
    ltv = (avg_revenue_per_account * gross_margin_pct) / churn_rate
    return ltv / cac_value
```

---

## Statistical Helpers

### Coefficient of Variation
```python
def cv(series) -> float:
    """Standardized measure of dispersion; useful for comparing variability across scales."""
    m = series.mean()
    if m == 0:
        return 0.0
    return series.std() / m
```

### Z-Score
```python
def z_score(value: float, series) -> float:
    s = series.std()
    if s == 0:
        return 0.0
    return (value - series.mean()) / s
```

---

## All formulas above are deterministic
Every formula here is a pure function with no side effects and no LLM-generated logic. The Transformation Agent must apply these as-is — never approximate or "guess" the formula.
