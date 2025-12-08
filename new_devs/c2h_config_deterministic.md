# Best Solution for Highcharts Config Generation

You’re right to separate this as the final, deterministic step. Let me show you the **best practical approach** that balances simplicity with robustness.

-----

## The Problem Breakdown

When generating a Highcharts config, you need to handle:

1. **Different data structures per chart type**

- Line chart: `series: [{data: [[x, y], [x, y]]}]`
- Pie chart: `series: [{data: [{name: 'A', y: 10}]}]`
- Heatmap: `series: [{data: [[x, y, value]]}]`

1. **Data transformations**

- Aggregation (sum, average, count)
- Pivoting (wide to long format)
- Filtering (top N, date ranges)
- Sorting

1. **Edge cases**

- Too many categories (>50) → sample or group “Others”
- Missing values → how to handle nulls
- Large datasets (>1000 points) → enable boost module
- Mixed data types → coerce or filter

1. **Highcharts-specific requirements**

- DateTime axis needs timestamps or formatted dates
- Categories array vs. numeric axis
- Multiple series handling

-----

## Best Solution: Schema-Driven Config Builder

Instead of simple placeholders, use a **data schema** that defines how to transform CSV → Highcharts format for each chart type.

-----

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Input:                                                  │
│  - chart_type: "line"                                    │
│  - data: DataFrame                                       │
│  - mapping: {x: "date", y: "revenue", series: "category"}│
│  - options: {aggregate: "sum", top_n: 20}               │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Chart Schema Loader                                     │
│  Load chart-specific schema from schemas/line.json      │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Data Transformer                                        │
│  Transform DataFrame based on schema requirements        │
│  - Apply aggregations                                    │
│  - Handle missing data                                   │
│  - Sort/filter                                           │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Config Builder                                          │
│  Build Highcharts config using transformed data         │
│  - Load base template                                    │
│  - Insert data in correct format                         │
│  - Apply edge case handlers                              │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Validator                                               │
│  Validate against Highcharts schema                      │
│  - Check required fields                                 │
│  - Validate data types                                   │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
                 Valid Highcharts Config
```

-----

## Component 1: Chart Schemas (Not Just Templates)

Instead of templates with placeholders, define **schemas** that describe data requirements and transformations:

### `schemas/line.json`

```json
{
  "chart_type": "line",
  "highcharts_type": "line",
  "description": "Line chart for temporal or sequential data",
  
  "data_requirements": {
    "x_axis": {
      "required": true,
      "types": ["temporal", "numeric", "sequential"],
      "role": "independent"
    },
    "y_axis": {
      "required": true,
      "types": ["numeric"],
      "role": "dependent"
    },
    "series_by": {
      "required": false,
      "types": ["categorical"],
      "max_categories": 20,
      "role": "grouping"
    }
  },
  
  "data_transformations": {
    "sort_by": "x_axis",
    "sort_order": "ascending",
    "handle_nulls": "connect",  // or "break", "zero"
    "aggregation": {
      "group_by": ["x_axis", "series_by"],
      "aggregate": "sum",  // default, can be overridden
      "agg_column": "y_axis"
    }
  },
  
  "data_format": {
    "series_structure": "array_of_series",
    "data_point_format": "tuple",  // [x, y] format
    "x_format": "auto"  // "datetime", "category", "numeric"
  },
  
  "config_template": {
    "chart": {
      "type": "line",
      "zoomType": "x"
    },
    "xAxis": {
      "type": "{{x_axis_type}}",
      "categories": "{{x_categories}}",  // if categorical
      "title": {"text": "{{x_title}}"}
    },
    "yAxis": {
      "title": {"text": "{{y_title}}"}
    },
    "series": "{{series_data}}",
    "plotOptions": {
      "line": {
        "marker": {"enabled": false}
      }
    }
  },
  
  "edge_case_handlers": {
    "too_many_points": {
      "threshold": 1000,
      "action": "enable_boost"
    },
    "too_many_series": {
      "threshold": 10,
      "action": "warn_user"
    }
  }
}
```

### `schemas/bar.json`

```json
{
  "chart_type": "bar",
  "highcharts_type": "bar",
  
  "data_requirements": {
    "x_axis": {
      "required": true,
      "types": ["categorical"],
      "role": "independent"
    },
    "y_axis": {
      "required": true,
      "types": ["numeric"],
      "role": "dependent"
    }
  },
  
  "data_transformations": {
    "sort_by": "y_axis",
    "sort_order": "descending",
    "aggregation": {
      "group_by": ["x_axis"],
      "aggregate": "sum",
      "agg_column": "y_axis"
    },
    "top_n": 20
  },
  
  "data_format": {
    "series_structure": "single_series",
    "data_point_format": "value_only"  // just y values
  },
  
  "config_template": {
    "chart": {"type": "bar"},
    "xAxis": {
      "categories": "{{x_categories}}",
      "title": {"text": null}
    },
    "yAxis": {
      "title": {"text": "{{y_title}}"}
    },
    "series": [{
      "name": "{{y_title}}",
      "data": "{{y_values}}"
    }],
    "legend": {"enabled": false}
  },
  
  "edge_case_handlers": {
    "too_many_categories": {
      "threshold": 30,
      "action": "take_top_n_and_group_rest"
    }
  }
}
```

### `schemas/pie.json`

```json
{
  "chart_type": "pie",
  "highcharts_type": "pie",
  
  "data_requirements": {
    "name_field": {
      "required": true,
      "types": ["categorical"],
      "max_categories": 7
    },
    "value_field": {
      "required": true,
      "types": ["numeric"]
    }
  },
  
  "data_transformations": {
    "sort_by": "value_field",
    "sort_order": "descending",
    "aggregation": {
      "group_by": ["name_field"],
      "aggregate": "sum",
      "agg_column": "value_field"
    },
    "top_n": 7
  },
  
  "data_format": {
    "series_structure": "single_series",
    "data_point_format": "name_value_object"  // {name: 'X', y: 10}
  },
  
  "config_template": {
    "chart": {"type": "pie"},
    "title": {"text": "{{title}}"},
    "series": [{
      "name": "{{series_name}}",
      "data": "{{pie_data}}"
    }],
    "plotOptions": {
      "pie": {
        "dataLabels": {
          "enabled": true,
          "format": "{point.name}: {point.percentage:.1f}%"
        }
      }
    }
  },
  
  "edge_case_handlers": {
    "too_many_slices": {
      "threshold": 7,
      "action": "group_smallest_as_other"
    }
  }
}
```

-----

## Component 2: Data Transformer

This component reads the schema and transforms the DataFrame accordingly:

```python
class DataTransformer:
    def __init__(self, schema):
        self.schema = schema
    
    def transform(self, df, column_mapping):
        """
        Transform DataFrame according to schema requirements
        
        Args:
            df: pandas DataFrame
            column_mapping: {
                "x_axis": "date",
                "y_axis": "revenue", 
                "series_by": "category"
            }
        
        Returns:
            Transformed DataFrame ready for config building
        """
        # Step 1: Extract relevant columns
        df = self._extract_columns(df, column_mapping)
        
        # Step 2: Handle nulls
        df = self._handle_nulls(df)
        
        # Step 3: Apply aggregations if specified
        if "aggregation" in self.schema["data_transformations"]:
            df = self._aggregate(df, column_mapping)
        
        # Step 4: Sort
        df = self._sort(df, column_mapping)
        
        # Step 5: Apply top_n filtering if specified
        if "top_n" in self.schema["data_transformations"]:
            df = self._apply_top_n(df, column_mapping)
        
        # Step 6: Handle edge cases
        df = self._handle_edge_cases(df, column_mapping)
        
        return df
    
    def _extract_columns(self, df, mapping):
        """Extract only the columns we need"""
        needed_cols = [col for col in mapping.values() if col]
        return df[needed_cols].copy()
    
    def _handle_nulls(self, df):
        """Handle null values based on schema"""
        null_strategy = self.schema["data_transformations"].get("handle_nulls", "drop")
        
        if null_strategy == "drop":
            return df.dropna()
        elif null_strategy == "zero":
            return df.fillna(0)
        elif null_strategy == "connect":
            # For line charts, keep nulls (Highcharts will connect)
            return df
        
        return df
    
    def _aggregate(self, df, mapping):
        """Apply aggregation as specified in schema"""
        agg_config = self.schema["data_transformations"]["aggregation"]
        
        group_by_cols = [
            mapping[field] 
            for field in agg_config["group_by"] 
            if field in mapping and mapping[field]
        ]
        
        agg_column = mapping[agg_config["agg_column"]]
        agg_func = agg_config["aggregate"]
        
        if not group_by_cols:
            # No grouping needed
            return df
        
        # Perform aggregation
        grouped = df.groupby(group_by_cols, as_index=False)[agg_column].agg(agg_func)
        
        return grouped
    
    def _sort(self, df, mapping):
        """Sort data as specified in schema"""
        sort_config = self.schema["data_transformations"]
        sort_by_field = sort_config.get("sort_by")
        sort_order = sort_config.get("sort_order", "ascending")
        
        if not sort_by_field:
            return df
        
        sort_column = mapping.get(sort_by_field)
        if not sort_column or sort_column not in df.columns:
            return df
        
        ascending = (sort_order == "ascending")
        return df.sort_values(sort_column, ascending=ascending)
    
    def _apply_top_n(self, df, mapping):
        """Take top N rows"""
        top_n = self.schema["data_transformations"].get("top_n")
        if top_n and len(df) > top_n:
            return df.head(top_n)
        return df
    
    def _handle_edge_cases(self, df, mapping):
        """Handle edge cases like too many categories"""
        handlers = self.schema.get("edge_case_handlers", {})
        
        # Too many categories
        if "too_many_categories" in handlers:
            threshold = handlers["too_many_categories"]["threshold"]
            x_col = mapping.get("x_axis")
            
            if x_col and x_col in df.columns and len(df) > threshold:
                action = handlers["too_many_categories"]["action"]
                if action == "take_top_n_and_group_rest":
                    df = self._group_others(df, x_col, threshold)
        
        return df
    
    def _group_others(self, df, category_col, top_n):
        """Group smallest categories into 'Others'"""
        top_categories = df.nlargest(top_n - 1, df.columns[-1])
        
        # Sum all other rows
        others_sum = df.iloc[top_n-1:].sum(numeric_only=True)
        others_row = pd.DataFrame([{
            category_col: "Others",
            **{col: others_sum[col] for col in df.columns if col != category_col}
        }])
        
        return pd.concat([top_categories, others_row], ignore_index=True)
```

-----

## Component 3: Config Builder

This takes transformed data and builds the Highcharts config:

```python
class HighchartsConfigBuilder:
    def __init__(self, schema):
        self.schema = schema
    
    def build(self, transformed_df, column_mapping, options=None):
        """
        Build Highcharts config from transformed data
        
        Args:
            transformed_df: Transformed DataFrame
            column_mapping: Original column mapping
            options: User preferences (colors, titles, etc.)
        
        Returns:
            Complete Highcharts config dict
        """
        options = options or {}
        
        # Start with template
        config = copy.deepcopy(self.schema["config_template"])
        
        # Build series data based on chart type
        series_data = self._build_series_data(transformed_df, column_mapping)
        
        # Replace placeholders in config
        config = self._populate_config(config, series_data, column_mapping, options)
        
        # Apply edge case configurations
        config = self._apply_edge_case_configs(config, transformed_df, series_data)
        
        # Apply user preferences
        if options.get("colors"):
            config["colors"] = options["colors"]
        
        return config
    
    def _build_series_data(self, df, mapping):
        """Build series data in format required by chart type"""
        data_format = self.schema["data_format"]
        structure = data_format["series_structure"]
        
        if structure == "array_of_series":
            # Line chart with multiple series
            return self._build_multi_series(df, mapping, data_format)
        
        elif structure == "single_series":
            # Bar/column chart
            return self._build_single_series(df, mapping, data_format)
        
        return []
    
    def _build_multi_series(self, df, mapping, data_format):
        """Build multiple series (for line charts, multi-bar, etc.)"""
        x_col = mapping["x_axis"]
        y_col = mapping["y_axis"]
        series_by = mapping.get("series_by")
        
        point_format = data_format["data_point_format"]
        
        if series_by and series_by in df.columns:
            # Multiple series
            series_list = []
            
            for series_name in df[series_by].unique():
                series_df = df[df[series_by] == series_name]
                
                if point_format == "tuple":
                    # [[x, y], [x, y]] format
                    data = series_df[[x_col, y_col]].values.tolist()
                else:
                    # [{x: ..., y: ...}] format
                    data = series_df[[x_col, y_col]].to_dict('records')
                
                series_list.append({
                    "name": str(series_name),
                    "data": data
                })
            
            return series_list
        
        else:
            # Single series
            if point_format == "tuple":
                data = df[[x_col, y_col]].values.tolist()
            else:
                data = df[[x_col, y_col]].to_dict('records')
            
            return [{
                "name": y_col,
                "data": data
            }]
    
    def _build_single_series(self, df, mapping, data_format):
        """Build single series (for bar, column, pie)"""
        point_format = data_format["data_point_format"]
        
        if point_format == "name_value_object":
            # Pie chart format: [{name: 'X', y: 10}]
            x_col = mapping.get("name_field") or mapping.get("x_axis")
            y_col = mapping.get("value_field") or mapping.get("y_axis")
            
            data = [
                {"name": str(row[x_col]), "y": float(row[y_col])}
                for _, row in df.iterrows()
            ]
            
            return [{
                "name": y_col,
                "data": data
            }]
        
        elif point_format == "value_only":
            # Bar chart format: just array of values
            y_col = mapping["y_axis"]
            return [{
                "name": y_col,
                "data": df[y_col].tolist()
            }]
        
        return []
    
    def _populate_config(self, config, series_data, mapping, options):
        """Replace placeholders in config with actual data"""
        
        # Replace series
        if "{{series_data}}" in str(config):
            config["series"] = series_data
        elif isinstance(config.get("series"), list) and len(config["series"]) > 0:
            if "{{pie_data}}" in str(config["series"]):
                config["series"][0]["data"] = series_data[0]["data"]
            elif "{{y_values}}" in str(config["series"]):
                config["series"][0]["data"] = series_data[0]["data"]
        
        # Replace axis info
        x_col = mapping.get("x_axis")
        y_col = mapping.get("y_axis")
        
        # X-axis
        if config.get("xAxis"):
            if "{{x_categories}}" in str(config["xAxis"]):
                # Get unique x values for categories
                config["xAxis"]["categories"] = series_data[0]["data"]  # Will be extracted properly
            
            if "{{x_title}}" in str(config["xAxis"]):
                config["xAxis"]["title"]["text"] = x_col or ""
        
        # Y-axis
        if config.get("yAxis"):
            if "{{y_title}}" in str(config["yAxis"]):
                config["yAxis"]["title"]["text"] = y_col or ""
        
        # Title
        if config.get("title") and "{{title}}" in str(config["title"]):
            config["title"]["text"] = options.get("title", "")
        
        return config
    
    def _apply_edge_case_configs(self, config, df, series_data):
        """Apply configuration changes for edge cases"""
        handlers = self.schema.get("edge_case_handlers", {})
        
        # Too many data points - enable boost
        if "too_many_points" in handlers:
            threshold = handlers["too_many_points"]["threshold"]
            total_points = sum(len(s.get("data", [])) for s in series_data)
            
            if total_points > threshold:
                config["boost"] = {
                    "useGPUTranslations": True,
                    "usePreAllocated": True
                }
                for series in config["series"]:
                    series["boostThreshold"] = 1
        
        return config
```

-----

## Component 4: Main API Function

```python
@app.post("/generate-config")
async def generate_config(
    data_id: str,
    chart_type: str,
    column_mapping: dict,
    options: Optional[dict] = None
):
    """
    Generate Highcharts config
    
    Example request:
    {
        "data_id": "abc123",
        "chart_type": "line",
        "column_mapping": {
            "x_axis": "date",
            "y_axis": "revenue",
            "series_by": "category"
        },
        "options": {
            "title": "Revenue Over Time",
            "colors": ["#FF6384", "#36A2EB"]
        }
    }
    """
    
    # Load data
    df = cache[data_id]["df"]
    
    # Load schema for chart type
    schema = load_schema(chart_type)
    
    # Transform data
    transformer = DataTransformer(schema)
    transformed_df = transformer.transform(df, column_mapping)
    
    # Build config
    builder = HighchartsConfigBuilder(schema)
    config = builder.build(transformed_df, column_mapping, options)
    
    # Validate (optional but recommended)
    validation_result = validate_config(config, chart_type)
    
    if not validation_result["valid"]:
        raise HTTPException(400, detail=validation_result["errors"])
    
    return {
        "config": config,
        "metadata": {
            "chart_type": chart_type,
            "data_points": len(transformed_df),
            "series_count": len(config["series"])
        }
    }
```

-----

## Why This Is Better Than Simple Templates

|Approach                |Simple Templates       |Schema-Driven (This)        |
|------------------------|-----------------------|----------------------------|
|**Data transformations**|Manual, error-prone    |Automatic, defined in schema|
|**Edge cases**          |If-else spaghetti      |Declarative handlers        |
|**Validation**          |Hard to validate       |Schema defines requirements |
|**Extensibility**       |Add code for each chart|Add JSON schema             |
|**Debugging**           |Debug Python code      |Debug data, not code        |
|**Testing**             |Mock data in tests     |Test schemas independently  |

-----

## Example: Generate Multiple Configs

```python
@app.post("/generate-all-recommended")
async def generate_all_configs(data_id: str, recommendations: list):
    """
    Generate configs for all recommended chart types
    """
    configs = []
    
    for rec in recommendations[:3]:  # Top 3
        chart_type = rec["chart_type"]
        column_mapping = rec["data_requirements"]  # From recommendation
        
        try:
            config = await generate_config(
                data_id=data_id,
                chart_type=chart_type,
                column_mapping=column_mapping
            )
            configs.append({
                "chart_type": chart_type,
                "config": config["config"],
                "reasoning": rec["reasoning"]
            })
        except Exception as e:
            # Log but continue with other charts
            print(f"Failed to generate {chart_type}: {e}")
    
    return {"configs": configs}
```

-----

## Summary: Your Solution vs. Best Solution

**Your Solution:**

- Templates with placeholders ✅
- Manual data insertion ⚠️
- Limited transformations ⚠️
- Hard to handle edge cases ❌

**Best Solution (Schema-Driven):**

- Schemas define data requirements ✅
- Automatic transformations ✅
- Declarative edge case handling ✅
- Easy to extend (add JSON, not code) ✅
- Testable and debuggable ✅

**The key difference:** Schemas encode **how to transform data**, not just **where to put it**.

Would you like me to show you example schemas for 5-6 common chart types to get you started?​​​​​​​​​​​​​​​​