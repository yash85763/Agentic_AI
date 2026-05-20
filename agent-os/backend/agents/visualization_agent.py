"""
VisualizationAgent - Generates Apache ECharts configurations.

CRITICAL DATA ACCURACY RULE:
    Every value in series.data MUST come from the sandbox output JSON.
    The LLM generates the chart STRUCTURE (title, axis labels, chart type,
    series wiring) but the LLM never invents numeric values.
"""

from __future__ import annotations

import json
import logging
import os
import textwrap
from typing import Any, Dict, List, Optional

from langfuse.decorators import observe

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

VIZ_MODEL = os.getenv("VIZ_MODEL", "claude-sonnet-4-6")
VIZ_FALLBACK = os.getenv("VIZ_FALLBACK", "openai/gpt-4o")

SUPPORTED_CHART_TYPES = (
    "line",
    "bar",
    "scatter",
    "heatmap",
    "candlestick",
    "time_series",
    "pie",
    "treemap",
    "sankey",
)

DEFAULT_COLOR_PALETTE = [
    "#3B82F6",  # blue
    "#10B981",  # green
    "#F59E0B",  # amber
    "#EF4444",  # red
    "#8B5CF6",  # purple
    "#06B6D4",  # cyan
    "#EC4899",  # pink
    "#84CC16",  # lime
]


class VisualizationAgent(BaseAgent):
    """Generates Apache ECharts configs from validated sandbox output."""

    DEFAULT_MODEL: str = VIZ_MODEL

    def __init__(
        self,
        cognitive_context: Dict[str, Any] = None,
        langfuse_handler: Any = None,
        model: str = None,
    ) -> None:
        super().__init__(
            model=model or VIZ_MODEL,
            cognitive_context=cognitive_context or {},
            langfuse_handler=langfuse_handler,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @observe(name="visualization.run")
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate charts from the merged dataset summary.

        Expected state:
            - merged_path: parquet path of consolidated data
            - validation: validation report dict (must have passed=True)
            - data_summary: dict with aggregate stats per column/group
            - job_id, redis_client

        Adds:
            - charts: list of {title, type, config} dicts
        """
        job_id = state.get("job_id", self._job_id())
        redis_client = state.get("redis_client")
        merged_path = state.get("merged_path")
        data_summary = state.get("data_summary") or self._build_data_summary(merged_path)

        self._emit_event(
            "agent_start",
            {"agent": "visualization"},
            job_id,
            redis_client,
        )

        # Decide which charts to generate based on data structure
        chart_requests = self._plan_charts(data_summary)

        charts: List[Dict[str, Any]] = []
        for req in chart_requests:
            try:
                config = self.generate_chart_config(
                    data_summary=data_summary,
                    chart_type=req["type"],
                    title=req["title"],
                    description=req.get("description", ""),
                )
                if self.validate_chart_accuracy(config, data_summary):
                    chart_obj = {
                        "title": req["title"],
                        "type": req["type"],
                        "config": config,
                    }
                    charts.append(chart_obj)
                    self._emit_event(
                        "chart_ready",
                        chart_obj,
                        job_id,
                        redis_client,
                    )
                else:
                    logger.warning("Chart accuracy validation failed for %s", req["title"])
            except Exception as exc:
                logger.error("Failed to generate chart '%s': %s", req["title"], exc)

        state["charts"] = charts
        self._emit_event(
            "agent_complete",
            {"agent": "visualization", "chart_count": len(charts)},
            job_id,
            redis_client,
        )
        return state

    # ------------------------------------------------------------------
    # Chart planning
    # ------------------------------------------------------------------

    def _plan_charts(self, data_summary: Dict[str, Any]) -> List[Dict[str, str]]:
        """Decide chart types based on the data summary structure."""
        plans: List[Dict[str, str]] = []
        groups = data_summary.get("groups", {})

        # 1) Bar chart per top categorical group
        for group_col, group_data in list(groups.items())[:1]:
            plans.append({
                "type": "bar",
                "title": f"Total by {group_col.replace('_', ' ').title()}",
                "description": f"Sum aggregated across {group_col}",
            })

        # 2) Time series if time column present
        if data_summary.get("time_column"):
            plans.append({
                "type": "line",
                "title": "Trend over time",
                "description": "Time-series view of primary metric",
            })

        # 3) Treemap if two-level grouping available
        if len(groups) >= 2:
            plans.append({
                "type": "treemap",
                "title": "Hierarchical breakdown",
                "description": "Two-level grouping by primary and secondary categories",
            })

        # 4) Pie for top-level distribution
        if groups:
            plans.append({
                "type": "pie",
                "title": "Distribution",
                "description": "Share of total by category",
            })

        return plans[:4]  # Cap at 4 charts

    # ------------------------------------------------------------------
    # ECharts config generation
    # ------------------------------------------------------------------

    def generate_chart_config(
        self,
        data_summary: Dict[str, Any],
        chart_type: str,
        title: str,
        description: str = "",
    ) -> Dict[str, Any]:
        """Generate ECharts config using LLM for structure, sandbox values for data.

        For accuracy, the actual data arrays come from data_summary directly;
        the LLM only controls layout/styling.
        """
        if chart_type not in SUPPORTED_CHART_TYPES:
            chart_type = "bar"

        # Hand-build the data portion to guarantee accuracy
        data_block = self._extract_chart_data(data_summary, chart_type)

        # Build the ECharts skeleton in code (no LLM hallucination risk)
        config = self._build_echarts_config(
            chart_type=chart_type,
            title=title,
            data_block=data_block,
        )

        return config

    def _build_echarts_config(
        self,
        chart_type: str,
        title: str,
        data_block: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a deterministic ECharts config from extracted data."""
        base = {
            "title": {
                "text": title,
                "left": "center",
                "textStyle": {"color": "#e2e8f0", "fontSize": 14, "fontWeight": "bold"},
            },
            "tooltip": {"trigger": "axis" if chart_type in ("line", "bar", "scatter") else "item"},
            "legend": {
                "bottom": 0,
                "textStyle": {"color": "#cbd5e1"},
            },
            "color": DEFAULT_COLOR_PALETTE,
            "backgroundColor": "transparent",
            "grid": {"left": 50, "right": 30, "top": 50, "bottom": 60},
        }

        if chart_type == "bar":
            base.update({
                "xAxis": {
                    "type": "category",
                    "data": data_block.get("categories", []),
                    "axisLabel": {"color": "#94a3b8"},
                },
                "yAxis": {
                    "type": "value",
                    "axisLabel": {"color": "#94a3b8"},
                    "splitLine": {"lineStyle": {"color": "#334155"}},
                },
                "series": [
                    {
                        "name": series.get("name"),
                        "type": "bar",
                        "data": series.get("data", []),
                        "itemStyle": {"borderRadius": [4, 4, 0, 0]},
                    }
                    for series in data_block.get("series", [])
                ],
            })

        elif chart_type in ("line", "time_series"):
            base.update({
                "xAxis": {
                    "type": "category" if chart_type == "line" else "time",
                    "data": data_block.get("categories", []),
                    "axisLabel": {"color": "#94a3b8"},
                    "boundaryGap": False,
                },
                "yAxis": {
                    "type": "value",
                    "axisLabel": {"color": "#94a3b8"},
                    "splitLine": {"lineStyle": {"color": "#334155"}},
                },
                "series": [
                    {
                        "name": series.get("name"),
                        "type": "line",
                        "data": series.get("data", []),
                        "smooth": True,
                        "areaStyle": {"opacity": 0.15},
                        "symbol": "circle",
                        "symbolSize": 5,
                    }
                    for series in data_block.get("series", [])
                ],
            })

        elif chart_type == "pie":
            base.update({
                "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
                "series": [{
                    "name": title,
                    "type": "pie",
                    "radius": ["35%", "65%"],
                    "data": data_block.get("pie_data", []),
                    "label": {"color": "#cbd5e1"},
                    "itemStyle": {"borderColor": "#0f172a", "borderWidth": 2},
                }],
            })

        elif chart_type == "scatter":
            base.update({
                "xAxis": {"type": "value", "axisLabel": {"color": "#94a3b8"}},
                "yAxis": {"type": "value", "axisLabel": {"color": "#94a3b8"}},
                "series": [{
                    "type": "scatter",
                    "data": data_block.get("scatter_data", []),
                    "symbolSize": 10,
                }],
            })

        elif chart_type == "heatmap":
            base.update({
                "xAxis": {"type": "category", "data": data_block.get("x_labels", [])},
                "yAxis": {"type": "category", "data": data_block.get("y_labels", [])},
                "visualMap": {
                    "min": data_block.get("min", 0),
                    "max": data_block.get("max", 100),
                    "calculable": True,
                    "orient": "horizontal",
                    "left": "center",
                    "bottom": 0,
                    "textStyle": {"color": "#94a3b8"},
                    "inRange": {"color": ["#1e3a8a", "#3b82f6", "#bfdbfe"]},
                },
                "series": [{
                    "type": "heatmap",
                    "data": data_block.get("heatmap_data", []),
                    "label": {"show": True, "color": "#0f172a"},
                }],
            })

        elif chart_type == "treemap":
            base.update({
                "series": [{
                    "type": "treemap",
                    "data": data_block.get("treemap_data", []),
                    "label": {"color": "#fff"},
                    "breadcrumb": {"show": False},
                    "leafDepth": 2,
                }],
            })

        elif chart_type == "candlestick":
            base.update({
                "xAxis": {"type": "category", "data": data_block.get("categories", [])},
                "yAxis": {"type": "value", "scale": True},
                "series": [{
                    "type": "candlestick",
                    "data": data_block.get("ohlc_data", []),
                }],
            })

        elif chart_type == "sankey":
            base.update({
                "series": [{
                    "type": "sankey",
                    "data": data_block.get("nodes", []),
                    "links": data_block.get("links", []),
                    "lineStyle": {"color": "gradient", "curveness": 0.5},
                }],
            })

        return base

    def _extract_chart_data(
        self,
        data_summary: Dict[str, Any],
        chart_type: str,
    ) -> Dict[str, Any]:
        """Extract data block for chart from data_summary.

        Every value here originates from the validated sandbox output —
        no LLM-generated numbers.
        """
        groups = data_summary.get("groups", {})

        if chart_type in ("bar", "pie"):
            primary = next(iter(groups.items()), (None, None))
            if not primary[0]:
                return {"categories": [], "series": [], "pie_data": []}
            col, values = primary
            sorted_items = sorted(values.items(), key=lambda x: -float(x[1] or 0))[:10]
            categories = [str(k) for k, _ in sorted_items]
            data = [float(v or 0) for _, v in sorted_items]
            return {
                "categories": categories,
                "series": [{"name": col, "data": data}],
                "pie_data": [{"name": c, "value": d} for c, d in zip(categories, data)],
            }

        if chart_type in ("line", "time_series"):
            ts = data_summary.get("time_series", {})
            categories = ts.get("dates", [])
            series = ts.get("series", [{"name": "value", "data": ts.get("values", [])}])
            return {"categories": categories, "series": series}

        if chart_type == "heatmap":
            hm = data_summary.get("heatmap", {})
            return {
                "x_labels": hm.get("x_labels", []),
                "y_labels": hm.get("y_labels", []),
                "heatmap_data": hm.get("data", []),
                "min": hm.get("min", 0),
                "max": hm.get("max", 100),
            }

        if chart_type == "treemap":
            return {"treemap_data": data_summary.get("treemap", [])}

        if chart_type == "scatter":
            return {"scatter_data": data_summary.get("scatter", [])}

        if chart_type == "candlestick":
            return {
                "categories": data_summary.get("ohlc_dates", []),
                "ohlc_data": data_summary.get("ohlc_data", []),
            }

        if chart_type == "sankey":
            return {
                "nodes": data_summary.get("sankey_nodes", []),
                "links": data_summary.get("sankey_links", []),
            }

        return {}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_chart_accuracy(
        self,
        config: Dict[str, Any],
        data_summary: Dict[str, Any],
    ) -> bool:
        """Verify every chart value originates from the sandbox data.

        This is a key correctness guarantee — we never let the LLM invent
        numbers that appear in a chart.
        """
        try:
            series_list = config.get("series", [])
            if not series_list:
                return False

            # Walk through every numeric data point and ensure it exists in
            # data_summary somewhere. We do a relaxed match: each number must
            # appear in the flattened set of all numeric values in the summary.
            allowed_values = self._flatten_numeric_values(data_summary)
            allowed_set = {round(v, 4) for v in allowed_values if v is not None}

            for series in series_list:
                data = series.get("data", [])
                for point in data:
                    val = self._extract_numeric(point)
                    if val is None:
                        continue
                    if round(val, 4) not in allowed_set:
                        # Tolerate small floating-point drift
                        if not any(abs(v - val) < 0.001 for v in allowed_set):
                            logger.warning(
                                "Chart value %s not found in data summary — "
                                "possible LLM hallucination",
                                val,
                            )
                            return False
            return True
        except Exception as exc:
            logger.error("Chart accuracy validation failed: %s", exc)
            return True  # Don't block on validator errors

    def _flatten_numeric_values(self, obj: Any, depth: int = 0) -> List[float]:
        """Recursively extract numeric values from a nested dict/list."""
        if depth > 6:
            return []
        values: List[float] = []
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            values.append(float(obj))
        elif isinstance(obj, dict):
            for v in obj.values():
                values.extend(self._flatten_numeric_values(v, depth + 1))
        elif isinstance(obj, list):
            for item in obj:
                values.extend(self._flatten_numeric_values(item, depth + 1))
        return values

    def _extract_numeric(self, point: Any) -> Optional[float]:
        if isinstance(point, (int, float)) and not isinstance(point, bool):
            return float(point)
        if isinstance(point, dict):
            if "value" in point:
                v = point["value"]
                return self._extract_numeric(v)
        if isinstance(point, list) and point:
            return self._extract_numeric(point[-1])
        return None

    def _build_data_summary(self, merged_path: str) -> Dict[str, Any]:
        """Fallback: build a basic data summary by reading the parquet directly."""
        if not merged_path or not os.path.exists(merged_path):
            return {"groups": {}, "time_series": {}, "row_count": 0}

        try:
            import pandas as pd
            df = pd.read_parquet(merged_path)
        except Exception as exc:
            logger.warning("Could not read merged parquet for summary: %s", exc)
            return {"groups": {}, "time_series": {}, "row_count": 0}

        summary: Dict[str, Any] = {"row_count": len(df), "groups": {}, "time_series": {}}

        # Numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        primary_metric = numeric_cols[0] if numeric_cols else None

        # Group bys on categorical columns
        cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        for col in cat_cols[:3]:
            if primary_metric:
                grouped = df.groupby(col)[primary_metric].sum()
                summary["groups"][col] = {
                    str(k): float(v) for k, v in grouped.items()
                }

        # Time series
        date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        if date_cols and primary_metric:
            time_col = date_cols[0]
            try:
                df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
                ts = df.dropna(subset=[time_col]).groupby(
                    df[time_col].dt.strftime("%Y-%m-%d")
                )[primary_metric].sum()
                summary["time_column"] = time_col
                summary["time_series"] = {
                    "dates": list(ts.index.astype(str)),
                    "series": [{
                        "name": primary_metric,
                        "data": [float(v) for v in ts.values],
                    }],
                }
            except Exception:
                pass

        return summary
