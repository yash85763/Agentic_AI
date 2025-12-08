"""
Deterministic Highcharts config builder (targeting Highcharts 12.4.0).

- Defines a strict input spec (ChartSpec and subtypes).
- Validates inputs with Pydantic.
- Builds Highcharts-ready config dicts deterministically.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Union, Tuple, Dict, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator


# ============================================================
# 1. Canonical input spec (your app's schema)
# ============================================================

ChartType = Literal["line", "column", "bar", "pie", "heatmap"]


class BaseChartSpec(BaseModel):
    """Common fields for all chart types."""
    id: str
    title: Optional[str] = None
    subtitle: Optional[str] = None
    x_label: Optional[str] = Field(default=None, alias="xLabel")
    y_label: Optional[str] = Field(default=None, alias="yLabel")

    class Config:
        allow_population_by_field_name = True  # support both x_label and xLabel


class CategoricalSeries(BaseModel):
    id: str
    name: str
    data: List[float]


class XYPoint(BaseModel):
    x: float
    y: float


class XYSeries(BaseModel):
    id: str
    name: str
    data: List[Tuple[float, float]]


class LineBarSpec(BaseChartSpec):
    type: Literal["line", "column", "bar"]
    categories: Optional[List[str]] = None
    series: List[CategoricalSeries]

    @validator("series")
    def validate_series_nonempty(cls, v: List[CategoricalSeries]) -> List[CategoricalSeries]:
        if len(v) == 0:
            raise ValueError("Line/column/bar spec must have at least one series.")
        return v

    @validator("categories", always=True)
    def validate_categories_length(cls, cats: Optional[List[str]], values: Dict[str, Any]) -> Optional[List[str]]:
        """
        If categories are provided, ensure they match the length of series data,
        or at least ensure all series share same length (for safety).
        """
        series: List[CategoricalSeries] = values.get("series", [])
        if not series:
            return cats

        # Check all series lengths equal
        lengths = {len(s.data) for s in series}
        if len(lengths) > 1:
            raise ValueError(
                f"All series data arrays must have the same length. Got lengths: {lengths}"
            )

        if cats is not None and len(cats) != next(iter(lengths)):
            raise ValueError(
                f"categories length ({len(cats)}) must match series data length ({next(iter(lengths))})."
            )
        return cats


class PiePoint(BaseModel):
    name: str
    y: float


class PieSpec(BaseChartSpec):
    type: Literal["pie"]
    data: List[PiePoint]

    @validator("data")
    def validate_data_nonempty(cls, v: List[PiePoint]) -> List[PiePoint]:
        if len(v) == 0:
            raise ValueError("Pie spec must have at least one data point.")
        return v


class HeatmapPoint(BaseModel):
    x: int
    y: int
    value: float


class HeatmapSpec(BaseChartSpec):
    type: Literal["heatmap"]
    x_categories: Optional[List[str]] = Field(default=None, alias="xCategories")
    y_categories: Optional[List[str]] = Field(default=None, alias="yCategories")
    data: List[HeatmapPoint]

    class Config:
        allow_population_by_field_name = True

    @validator("data")
    def validate_data_nonempty(cls, v: List[HeatmapPoint]) -> List[HeatmapPoint]:
        if len(v) == 0:
            raise ValueError("Heatmap spec must have at least one data point.")
        return v

    @validator("x_categories", "y_categories", always=True)
    def validate_categories_optional(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        # We allow None or non-empty lists.
        if v is not None and len(v) == 0:
            raise ValueError("xCategories/yCategories cannot be empty lists; use None or non-empty list.")
        return v


ChartSpec = Union[LineBarSpec, PieSpec, HeatmapSpec]


# ============================================================
# 2. Base templates (deterministic Highcharts configs)
#    These are your default options. We only fill data & labels.
# ============================================================

BASE_LINE_BAR_TEMPLATE: Dict[str, Any] = {
    "chart": {
        "type": "line",  # will override for 'column' / 'bar'
    },
    "title": {"text": ""},
    "subtitle": {"text": ""},
    "xAxis": {
        "categories": [],
        "title": {"text": None},
    },
    "yAxis": {
        "title": {"text": None},
    },
    "legend": {"enabled": True},
    "tooltip": {"shared": True},
    "plotOptions": {
        "series": {
            "dataLabels": {"enabled": False},
            "animation": True,
        }
    },
    "series": [],
}

BASE_PIE_TEMPLATE: Dict[str, Any] = {
    "chart": {"type": "pie"},
    "title": {"text": ""},
    "subtitle": {"text": ""},
    "tooltip": {
        "pointFormat": "{series.name}: <b>{point.percentage:.1f}%</b>"
    },
    "accessibility": {
        "point": {"valueSuffix": "%"}
    },
    "plotOptions": {
        "pie": {
            "allowPointSelect": True,
            "cursor": "pointer",
            "dataLabels": {
                "enabled": True,
                "format": "<b>{point.name}</b>: {point.percentage:.1f} %",
            },
        }
    },
    "series": [],
}

BASE_HEATMAP_TEMPLATE: Dict[str, Any] = {
    "chart": {"type": "heatmap"},
    "title": {"text": ""},
    "subtitle": {"text": ""},
    "xAxis": {
        "categories": [],
        "title": {"text": None},
    },
    "yAxis": {
        "categories": [],
        "title": {"text": None},
        "reversed": True,
    },
    "colorAxis": {
        "min": 0,
        # You can tweak colors; keep them deterministic
        "stops": [
            [0.0, "#FFFFFF"],
            [0.5, "#7CB5EC"],
            [1.0, "#0000FF"],
        ],
    },
    "legend": {
        "align": "right",
        "layout": "vertical",
        "margin": 0,
        "verticalAlign": "top",
        "symbolHeight": 200,
    },
    "series": [],
}


# ============================================================
# 3. Builder functions per chart family
# ============================================================

def build_line_bar_options(spec: LineBarSpec) -> Dict[str, Any]:
    """
    Build deterministic Highcharts options for line/column/bar charts.

    :param spec: Validated LineBarSpec instance.
    :return: dict compatible with Highcharts.Options.
    """
    # Start from base template (copy to avoid mutation side effects)
    options: Dict[str, Any] = {
        "chart": dict(BASE_LINE_BAR_TEMPLATE["chart"]),
        "title": dict(BASE_LINE_BAR_TEMPLATE["title"]),
        "subtitle": dict(BASE_LINE_BAR_TEMPLATE["subtitle"]),
        "xAxis": dict(BASE_LINE_BAR_TEMPLATE["xAxis"]),
        "yAxis": dict(BASE_LINE_BAR_TEMPLATE["yAxis"]),
        "legend": dict(BASE_LINE_BAR_TEMPLATE["legend"]),
        "tooltip": dict(BASE_LINE_BAR_TEMPLATE["tooltip"]),
        "plotOptions": {
            "series": dict(BASE_LINE_BAR_TEMPLATE["plotOptions"]["series"])
        },
        "series": [],
    }

    options["chart"]["type"] = spec.type
    options["title"]["text"] = spec.title or ""
    options["subtitle"]["text"] = spec.subtitle or ""

    options["xAxis"]["categories"] = spec.categories or []
    options["xAxis"]["title"]["text"] = spec.x_label

    options["yAxis"]["title"]["text"] = spec.y_label

    # Deterministic series mapping, preserve order
    series_list: List[Dict[str, Any]] = []
    for s in spec.series:
        series_list.append(
            {
                "type": spec.type,  # explicit; Highcharts also infers from chart.type
                "name": s.name,
                "data": s.data,
                # You can add deterministic style options here if needed
            }
        )

    options["series"] = series_list
    return options


def build_pie_options(spec: PieSpec) -> Dict[str, Any]:
    """
    Build deterministic Highcharts options for pie charts.

    :param spec: Validated PieSpec instance.
    :return: dict compatible with Highcharts.Options.
    """
    options: Dict[str, Any] = {
        "chart": dict(BASE_PIE_TEMPLATE["chart"]),
        "title": dict(BASE_PIE_TEMPLATE["title"]),
        "subtitle": dict(BASE_PIE_TEMPLATE["subtitle"]),
        "tooltip": dict(BASE_PIE_TEMPLATE["tooltip"]),
        "accessibility": dict(BASE_PIE_TEMPLATE["accessibility"]),
        "plotOptions": {
            "pie": dict(BASE_PIE_TEMPLATE["plotOptions"]["pie"])
        },
        "series": [],
    }

    options["title"]["text"] = spec.title or ""
    options["subtitle"]["text"] = spec.subtitle or ""

    series_data = [
        {"name": p.name, "y": p.y} for p in spec.data
    ]

    options["series"] = [
        {
            "type": "pie",
            "name": spec.title or "Share",
            "data": series_data,
        }
    ]
    return options


def build_heatmap_options(spec: HeatmapSpec) -> Dict[str, Any]:
    """
    Build deterministic Highcharts options for heatmaps.

    :param spec: Validated HeatmapSpec instance.
    :return: dict compatible with Highcharts.Options.
    """
    options: Dict[str, Any] = {
        "chart": dict(BASE_HEATMAP_TEMPLATE["chart"]),
        "title": dict(BASE_HEATMAP_TEMPLATE["title"]),
        "subtitle": dict(BASE_HEATMAP_TEMPLATE["subtitle"]),
        "xAxis": dict(BASE_HEATMAP_TEMPLATE["xAxis"]),
        "yAxis": dict(BASE_HEATMAP_TEMPLATE["yAxis"]),
        "colorAxis": dict(BASE_HEATMAP_TEMPLATE["colorAxis"]),
        "legend": dict(BASE_HEATMAP_TEMPLATE["legend"]),
        "series": [],
    }

    options["title"]["text"] = spec.title or ""
    options["subtitle"]["text"] = spec.subtitle or ""

    options["xAxis"]["categories"] = spec.x_categories or []
    options["xAxis"]["title"]["text"] = spec.x_label

    options["yAxis"]["categories"] = spec.y_categories or []
    options["yAxis"]["title"]["text"] = spec.y_label

    # Highcharts heatmap expects [x, y, value] tuples in data
    series_data = [[p.x, p.y, p.value] for p in spec.data]

    options["series"] = [
        {
            "type": "heatmap",
            "data": series_data,
        }
    ]
    return options


# ============================================================
# 4. Public entry point (factory)
# ============================================================

def build_highcharts_options(raw_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point you call from your app.

    - Accepts a raw dict (e.g., from JSON / request body).
    - Validates it into the correct ChartSpec subtype.
    - Dispatches to the appropriate builder.

    This is deterministic: same input dict => same output dict.
    """
    chart_type = raw_spec.get("type")
    if chart_type not in ("line", "column", "bar", "pie", "heatmap"):
        raise ValueError(f"Unsupported chart type: {chart_type!r}")

    # Validate & parse into specific spec model
    if chart_type in ("line", "column", "bar"):
        spec = LineBarSpec.parse_obj(raw_spec)
        return build_line_bar_options(spec)
    elif chart_type == "pie":
        spec = PieSpec.parse_obj(raw_spec)
        return build_pie_options(spec)
    elif chart_type == "heatmap":
        spec = HeatmapSpec.parse_obj(raw_spec)
        return build_heatmap_options(spec)

    # This is just for mypy / static check completeness
    raise ValueError(f"Unsupported chart type: {chart_type!r}")


# ============================================================
# 5. Example usage & basic determinism tests
# ============================================================

if __name__ == "__main__":
    # --- Example: line chart ---
    line_spec = {
        "id": "sales-line",
        "type": "line",
        "title": "Monthly Sales",
        "subtitle": "FY 2025",
        "xLabel": "Month",
        "yLabel": "Sales (k USD)",
        "categories": ["Jan", "Feb", "Mar"],
        "series": [
            {"id": "s1", "name": "Product A", "data": [10, 12, 15]},
            {"id": "s2", "name": "Product B", "data": [8, 9, 11]},
        ],
    }

    line_opts_1 = build_highcharts_options(line_spec)
    line_opts_2 = build_highcharts_options(line_spec)
    assert line_opts_1 == line_opts_2, "Line chart builder must be deterministic."

    print("Line chart options:")
    print(line_opts_1)

    # --- Example: pie chart ---
    pie_spec = {
        "id": "market-share-pie",
        "type": "pie",
        "title": "Market Share",
        "subtitle": "Q1 2025",
        "data": [
            {"name": "Brand A", "y": 45.0},
            {"name": "Brand B", "y": 30.0},
            {"name": "Brand C", "y": 25.0},
        ],
    }

    pie_opts = build_highcharts_options(pie_spec)
    print("\nPie chart options:")
    print(pie_opts)

    # --- Example: heatmap ---
    heatmap_spec = {
        "id": "performance-heatmap",
        "type": "heatmap",
        "title": "Performance Heatmap",
        "subtitle": "Scores",
        "xCategories": ["Jan", "Feb", "Mar"],
        "yCategories": ["Team A", "Team B"],
        "xLabel": "Month",
        "yLabel": "Team",
        "data": [
            {"x": 0, "y": 0, "value": 75.0},
            {"x": 1, "y": 0, "value": 80.0},
            {"x": 2, "y": 0, "value": 90.0},
            {"x": 0, "y": 1, "value": 65.0},
            {"x": 1, "y": 1, "value": 70.0},
            {"x": 2, "y": 1, "value": 85.0},
        ],
    }

    heatmap_opts = build_highcharts_options(heatmap_spec)
    print("\nHeatmap options:")
    print(heatmap_opts)

    print("\nAll example configs built deterministically and validated.")