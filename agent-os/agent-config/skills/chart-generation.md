# Skill: Apache ECharts Configuration

## Purpose
Generate Apache ECharts configurations that visualize data **truthfully**. Every numeric value in any chart MUST come from sandbox-validated data — never from LLM generation.

---

## The Accuracy Contract

⚠️ **CRITICAL RULE**: When generating a chart config, the LLM produces only the layout/structure. The numeric values in `series.data` must be substituted in from the actual sandbox JSON output. Never type numbers into a chart config.

**Wrong** (LLM hallucinated values):
```python
config = {
    "series": [{"type": "bar", "data": [120, 200, 150, 80, 70]}]
}
```

**Right** (values come from sandbox):
```python
sandbox_output = {"sales_by_team": {"ENG": 250000, "SALES": 180000, "MKTG": 95000}}
config = {
    "xAxis": {"data": list(sandbox_output["sales_by_team"].keys())},
    "series": [{"type": "bar", "data": list(sandbox_output["sales_by_team"].values())}],
}
```

---

## Chart Type Selection

| Goal | Chart type | When |
|---|---|---|
| Trend over time | `line` or `time_series` | dates + 1+ numeric series |
| Compare categories | `bar` | <15 categories, single metric |
| Distribution shape | `histogram` (bar w/ binning) | continuous numeric data |
| Composition | `pie` or `treemap` | parts-of-a-whole, ≤7 slices |
| Correlation | `scatter` | two numeric variables |
| Matrix / 2-D distribution | `heatmap` | two categorical axes, one metric |
| Hierarchical breakdown | `treemap` or `sankey` | nested categories |
| Stock / OHLC | `candlestick` | financial OHLC data |
| Flow between states | `sankey` | source → target with volumes |

---

## Color Palette

Always use this palette for consistency with the AgentOS UI:

```python
PALETTE = [
    "#3B82F6",  # blue (primary)
    "#10B981",  # green (positive, success)
    "#F59E0B",  # amber (warning, attention)
    "#EF4444",  # red (negative, error)
    "#8B5CF6",  # purple
    "#06B6D4",  # cyan
    "#EC4899",  # pink
    "#84CC16",  # lime
]
```

For sequential data (heatmap, choropleth), use a gradient:
```python
"visualMap": {
    "inRange": {"color": ["#1e3a8a", "#3b82f6", "#bfdbfe"]}
}
```

For diverging data (e.g., variance from budget), red-amber-green:
```python
"visualMap": {
    "inRange": {"color": ["#ef4444", "#f59e0b", "#10b981"]}
}
```

---

## Dark-Theme Defaults

Every chart MUST include these dark-theme styling defaults to match the AgentOS UI:

```python
base = {
    "backgroundColor": "transparent",
    "textStyle": {"color": "#cbd5e1"},
    "title": {
        "textStyle": {"color": "#e2e8f0", "fontSize": 14, "fontWeight": "bold"},
        "left": "center",
    },
    "legend": {
        "textStyle": {"color": "#cbd5e1"},
        "bottom": 0,
    },
    "tooltip": {"trigger": "axis"},
    "color": PALETTE,
    "grid": {"left": 50, "right": 30, "top": 50, "bottom": 60},
    "xAxis": {
        "axisLabel": {"color": "#94a3b8"},
        "axisLine": {"lineStyle": {"color": "#475569"}},
    },
    "yAxis": {
        "axisLabel": {"color": "#94a3b8"},
        "axisLine": {"lineStyle": {"color": "#475569"}},
        "splitLine": {"lineStyle": {"color": "#334155"}},
    },
}
```

---

## Template: Bar Chart

```python
config = {
    **base,
    "title": {"text": "Total Expense by Team — Q1 2024", "left": "center"},
    "xAxis": {"type": "category", "data": categories},  # from sandbox
    "yAxis": {"type": "value", "name": "USD"},
    "series": [{
        "name": "Expense",
        "type": "bar",
        "data": values,  # from sandbox
        "itemStyle": {"borderRadius": [4, 4, 0, 0]},
    }],
}
```

## Template: Multi-Series Line

```python
config = {
    **base,
    "title": {"text": "Monthly Revenue by Region"},
    "xAxis": {"type": "category", "data": months, "boundaryGap": False},
    "yAxis": {"type": "value", "name": "USD"},
    "series": [
        {"name": region, "type": "line", "data": values, "smooth": True, "areaStyle": {"opacity": 0.15}}
        for region, values in regional_data.items()
    ],
}
```

## Template: Pie

```python
config = {
    **base,
    "tooltip": {"trigger": "item", "formatter": "{b}: ${c} ({d}%)"},
    "series": [{
        "name": "Spend Distribution",
        "type": "pie",
        "radius": ["35%", "65%"],
        "data": [{"name": k, "value": v} for k, v in category_totals.items()],
        "itemStyle": {"borderColor": "#0f172a", "borderWidth": 2},
        "label": {"color": "#cbd5e1"},
    }],
}
```

## Template: Heatmap

```python
config = {
    **base,
    "tooltip": {"position": "top"},
    "xAxis": {"type": "category", "data": x_labels, "splitArea": {"show": True}},
    "yAxis": {"type": "category", "data": y_labels, "splitArea": {"show": True}},
    "visualMap": {
        "min": min_val, "max": max_val,
        "calculable": True, "orient": "horizontal",
        "left": "center", "bottom": 0,
        "textStyle": {"color": "#94a3b8"},
        "inRange": {"color": ["#1e3a8a", "#3b82f6", "#bfdbfe"]},
    },
    "series": [{
        "name": "Density",
        "type": "heatmap",
        "data": heatmap_data,  # [[x_idx, y_idx, value], ...]
        "label": {"show": True, "color": "#0f172a"},
    }],
}
```

## Template: Candlestick (financial)

```python
config = {
    **base,
    "xAxis": {"type": "category", "data": dates},
    "yAxis": {"scale": True},
    "series": [{
        "type": "candlestick",
        "data": ohlc,  # [[open, close, low, high], ...]
        "itemStyle": {
            "color": "#10b981",       # bullish (up)
            "color0": "#ef4444",      # bearish (down)
            "borderColor": "#10b981",
            "borderColor0": "#ef4444",
        },
    }],
}
```

## Template: Treemap

```python
config = {
    **base,
    "series": [{
        "type": "treemap",
        "data": [
            {"name": "ENG", "value": 250000, "children": [
                {"name": "Salaries", "value": 180000},
                {"name": "Tools", "value": 70000},
            ]},
            # ...
        ],
        "label": {"show": True, "color": "#fff"},
        "leafDepth": 2,
        "breadcrumb": {"show": False},
    }],
}
```

---

## Self-Check Before Submitting Chart

Before emitting `chart_ready`, verify:
1. ✅ Every value in `series[*].data` is present in the sandbox output dict
2. ✅ Title accurately describes what the chart shows
3. ✅ Axis labels include units (USD, %, count, days, etc.)
4. ✅ Color palette is from PALETTE constants
5. ✅ Dark theme styling applied
6. ✅ Tooltip is enabled
7. ✅ Legend shown when there are multiple series
8. ✅ No more than 12 distinct colors in a single chart (visual overload)
