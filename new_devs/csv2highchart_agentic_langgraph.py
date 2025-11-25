"""
CSV to Highcharts - LangGraph Implementation
Production-ready graph-based agent for analyzing CSV data and generating Highcharts configurations

Dependencies:
    langchain==0.3.0
    langchain-openai==0.2.0
    langchain-core==0.3.0
    langgraph==0.2.34
    pandas==2.2.0
    openai==1.54.0
"""

import os
import json
import pandas as pd
from typing import Optional, List, Dict, Any, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


# ============================================================
# Data Analyzer Class
# ============================================================

class CSVDataAnalyzer:
    """Provides tools for analyzing CSV data"""
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.csv_path: Optional[str] = None
    
    def load_csv(self, csv_path: str) -> pd.DataFrame:
        """Load CSV file into memory"""
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        return self.df
    
    def get_basic_info(self) -> str:
        """Get basic information about the dataset"""
        if self.df is None:
            return json.dumps({"error": "No data loaded"})
        
        info = {
            "total_rows": int(len(self.df)),
            "total_columns": int(len(self.df.columns)),
            "column_names": list(self.df.columns),
            "memory_usage_mb": round(self.df.memory_usage(deep=True).sum() / 1024**2, 2)
        }
        return json.dumps(info, indent=2)
    
    def get_data_types(self) -> str:
        """Get data types of all columns"""
        if self.df is None:
            return json.dumps({"error": "No data loaded"})
        
        dtypes = {}
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            if dtype in ['int64', 'float64'] and self.df[col].nunique() < 10:
                dtype = f"{dtype} (possibly categorical)"
            elif dtype == 'object':
                try:
                    pd.to_datetime(self.df[col], errors='raise')
                    dtype = "object (possibly datetime)"
                except:
                    pass
            dtypes[col] = dtype
        
        return json.dumps(dtypes, indent=2)
    
    def get_column_statistics(self, column_name: str) -> str:
        """Get detailed statistics for a specific column"""
        if self.df is None:
            return json.dumps({"error": "No data loaded"})
        
        if column_name not in self.df.columns:
            return json.dumps({
                "error": f"Column '{column_name}' not found",
                "available_columns": list(self.df.columns)
            })
        
        col = self.df[column_name]
        stats = {
            "column_name": column_name,
            "data_type": str(col.dtype),
            "non_null_count": int(col.count()),
            "null_count": int(col.isnull().sum()),
            "unique_values": int(col.nunique())
        }
        
        if pd.api.types.is_numeric_dtype(col):
            stats.update({
                "min": float(col.min()) if not pd.isna(col.min()) else None,
                "max": float(col.max()) if not pd.isna(col.max()) else None,
                "mean": float(col.mean()) if not pd.isna(col.mean()) else None,
                "median": float(col.median()) if not pd.isna(col.median()) else None,
                "std": float(col.std()) if not pd.isna(col.std()) else None
            })
        
        if col.nunique() < 20:
            value_counts = col.value_counts().head(10).to_dict()
            stats["top_values"] = {str(k): int(v) for k, v in value_counts.items()}
        
        return json.dumps(stats, indent=2)
    
    def get_sample_data(self) -> str:
        """Get sample rows from the dataset"""
        if self.df is None:
            return json.dumps({"error": "No data loaded"})
        
        n_rows = min(5, len(self.df))
        sample = self.df.head(n_rows).to_dict(orient='records')
        return json.dumps(sample, indent=2, default=str)
    
    def get_correlation_info(self) -> str:
        """Get correlation information for numeric columns"""
        if self.df is None:
            return json.dumps({"error": "No data loaded"})
        
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if len(numeric_cols) < 2:
            return json.dumps({
                "message": "Not enough numeric columns for correlation analysis",
                "numeric_columns": numeric_cols
            })
        
        corr_matrix = self.df[numeric_cols].corr()
        
        strong_corr = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_corr.append({
                        "column1": numeric_cols[i],
                        "column2": numeric_cols[j],
                        "correlation": round(float(corr_val), 3)
                    })
        
        result = {
            "numeric_columns": numeric_cols,
            "strong_correlations": strong_corr
        }
        return json.dumps(result, indent=2)
    
    def detect_time_series(self) -> str:
        """Detect if dataset contains time series data"""
        if self.df is None:
            return json.dumps({"error": "No data loaded"})
        
        time_columns = []
        for col in self.df.columns:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'year', 'month', 'day']):
                try:
                    pd.to_datetime(self.df[col], errors='raise')
                    time_columns.append(col)
                except:
                    pass
        
        result = {
            "has_time_series": len(time_columns) > 0,
            "time_columns": time_columns
        }
        return json.dumps(result, indent=2)
    
    def get_all_columns_summary(self) -> str:
        """Get a quick summary of all columns"""
        if self.df is None:
            return json.dumps({"error": "No data loaded"})
        
        summary = []
        for col in self.df.columns:
            col_data = self.df[col]
            summary.append({
                "column": col,
                "dtype": str(col_data.dtype),
                "non_null": int(col_data.count()),
                "unique": int(col_data.nunique()),
                "null_percentage": round(col_data.isnull().sum() / len(col_data) * 100, 2)
            })
        
        return json.dumps(summary, indent=2)


# ============================================================
# Graph State Definition
# ============================================================

class GraphState(TypedDict):
    """State of the LangGraph workflow"""
    messages: List
    csv_path: str
    analysis_complete: bool
    tool_calls_count: int
    final_recommendation: Optional[Dict[str, Any]]


# ============================================================
# Main LangGraph Agent Class
# ============================================================

class CSVToHighchartsGraph:
    """LangGraph-based implementation for CSV analysis and Highcharts recommendation"""
    
    def __init__(self, api_key: Optional[str] = OPENAI_API_KEY, model: str = "gpt-4"):
        """
        Initialize the graph-based agent
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env variable)
            model: Model to use (gpt-4, gpt-3.5-turbo, etc.)
        """
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Initialize data analyzer
        self.analyzer = CSVDataAnalyzer()
        
        # Initialize LLM
        self.llm = ChatOpenAI(model=model, temperature=0.3)
        
        # Create tools
        self.tools = self._create_tools()
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Create the graph
        self.graph = self._create_graph()
    
    def _create_tools(self) -> List:
        """Create tools that the LLM can use"""
        
        @tool
        def get_basic_info() -> str:
            """Get basic information about the dataset including total rows, columns, column names, and memory usage. Use this first to understand the dataset structure."""
            return self.analyzer.get_basic_info()
        
        @tool
        def get_data_types() -> str:
            """Get data types of all columns. This helps identify numeric, categorical, datetime, and other column types."""
            return self.analyzer.get_data_types()
        
        @tool
        def get_column_statistics(column_name: str) -> str:
            """Get detailed statistics for a SPECIFIC column. Input should be the exact column name. Returns min, max, mean, median, std for numeric columns, or value counts for categorical."""
            return self.analyzer.get_column_statistics(column_name)
        
        @tool
        def get_sample_data() -> str:
            """Get sample rows (first 5) from the dataset. Use this to see actual data values and understand data patterns."""
            return self.analyzer.get_sample_data()
        
        @tool
        def get_correlation_info() -> str:
            """Get correlation information between numeric columns. Use this to identify relationships between variables."""
            return self.analyzer.get_correlation_info()
        
        @tool
        def detect_time_series() -> str:
            """Detect if the dataset contains time series data. Identifies columns with dates or timestamps."""
            return self.analyzer.detect_time_series()
        
        @tool
        def get_all_columns_summary() -> str:
            """Get a quick summary of ALL columns at once. Shows data type, null count, and unique values for each column."""
            return self.analyzer.get_all_columns_summary()
        
        return [
            get_basic_info,
            get_data_types,
            get_column_statistics,
            get_sample_data,
            get_correlation_info,
            detect_time_series,
            get_all_columns_summary
        ]
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow"""
        
        # Create the graph
        workflow = StateGraph(GraphState)
        
        # Define nodes
        workflow.add_node("agent", self.agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("synthesize", self.synthesize_node)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "continue": "tools",
                "synthesize": "synthesize",
                "end": END
            }
        )
        
        workflow.add_edge("tools", "agent")
        workflow.add_edge("synthesize", END)
        
        return workflow.compile()
    
    def agent_node(self, state: GraphState) -> GraphState:
        """Agent decides what to do next"""
        messages = state["messages"]
        
        # Invoke the LLM with tools
        response = self.llm_with_tools.invoke(messages)
        
        # Update state
        state["messages"].append(response)
        
        return state
    
    def should_continue(self, state: GraphState) -> str:
        """Determine if we should continue with tools or synthesize results"""
        last_message = state["messages"][-1]
        
        # Check if the LLM wants to use tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            # Count tool calls
            state["tool_calls_count"] += len(last_message.tool_calls)
            return "continue"
        
        # Check if we've made enough tool calls to synthesize (at least 3)
        if state["tool_calls_count"] >= 3:
            return "synthesize"
        
        # If no tool calls and not enough data, ask for more
        if state["tool_calls_count"] < 3:
            return "continue"
        
        return "end"
    
    def synthesize_node(self, state: GraphState) -> GraphState:
        """Synthesize all findings into a final recommendation"""
        
        synthesis_prompt = """Based on all the tool calls you've made and the data you've analyzed, 
        provide a final recommendation for the best Highcharts configuration.
        
        Your response MUST be a valid JSON object with this EXACT structure:
        {
            "chartType": "line|column|bar|pie|scatter|area|spline|areaspline",
            "reasoning": "Detailed explanation based on your analysis of the data patterns, types, and relationships",
            "highchartsConfig": {
                "chart": {
                    "type": "chart_type_here"
                },
                "title": {
                    "text": "Appropriate title based on the data"
                },
                "xAxis": {
                    "categories": ["list", "of", "categories"],
                    "title": {"text": "X axis label"}
                },
                "yAxis": {
                    "title": {"text": "Y axis label"}
                },
                "series": [
                    {
                        "name": "Series name",
                        "data": [actual, data, values]
                    }
                ],
                "credits": {
                    "enabled": false
                }
            },
            "dataInsights": {
                "rowCount": 123,
                "keyColumns": ["col1", "col2"],
                "dataPattern": "time-series|categorical|correlation|mixed"
            }
        }
        
        CRITICAL REQUIREMENTS:
        - Use ACTUAL data from the CSV in the highchartsConfig (from sample_data or statistics)
        - The chart must be immediately usable with real data
        - Choose the chart type based on data patterns you discovered
        - For time series: use line or area charts
        - For categorical comparisons: use column or bar charts
        - For proportions/parts of whole: use pie charts
        - For correlations: use scatter plots
        - Return ONLY the JSON object, no markdown code blocks or extra text
        """
        
        state["messages"].append(HumanMessage(content=synthesis_prompt))
        
        # Get final response
        final_response = self.llm.invoke(state["messages"])
        
        # Parse the JSON
        try:
            content = final_response.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            result = json.loads(content)
            state["final_recommendation"] = result
            state["analysis_complete"] = True
            
        except json.JSONDecodeError as e:
            print(f"\n⚠️  Warning: Could not parse JSON: {e}")
            print(f"Raw response: {final_response.content[:500]}...")
            
            # Create a fallback response
            state["final_recommendation"] = {
                "error": "Failed to parse recommendation",
                "raw_content": final_response.content,
                "chartType": "line",
                "reasoning": "Error in parsing, defaulting to line chart"
            }
            state["analysis_complete"] = True
        
        return state
    
    def analyze_csv(self, csv_path: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Main method to analyze CSV and get Highcharts recommendation
        
        Args:
            csv_path: Path to CSV file
            verbose: Whether to print progress information
            
        Returns:
            Dictionary with chart recommendation and config
        """
        # Load CSV
        if verbose:
            print(f"\n📂 Loading CSV: {csv_path}")
        
        self.analyzer.load_csv(csv_path)
        
        if verbose:
            print(f"✓ Loaded {len(self.analyzer.df)} rows, {len(self.analyzer.df.columns)} columns\n")
        
        # Initialize state
        initial_state = {
            "messages": [
                SystemMessage(content="""You are a data visualization expert. Analyze the CSV dataset 
                using the available tools to recommend the best Highcharts configuration.
                
                ANALYSIS PROCESS:
                1. Start with get_basic_info and get_all_columns_summary to understand structure
                2. Use detect_time_series to check for temporal data
                3. Use get_sample_data to see actual values
                4. For key columns, use get_column_statistics for detailed stats
                5. Use get_correlation_info if there are multiple numeric columns
                6. Based on patterns, determine the optimal chart type
                
                CHART TYPE SELECTION:
                - Line/Area charts: Time series data, trends over time
                - Column/Bar charts: Categorical comparisons, grouped data
                - Pie charts: Part-to-whole relationships (max 7-8 categories)
                - Scatter plots: Correlation between two numeric variables
                - Multiple series: When comparing 2-4 related metrics
                
                Be thorough but efficient. Gather enough information to make a confident recommendation."""),
                
                HumanMessage(content=f"""Analyze the loaded CSV dataset at {csv_path}.
                
                Use the available tools to understand:
                - Dataset structure and size
                - Column types and patterns
                - Presence of time series
                - Key statistics
                - Correlations between variables
                
                Start your analysis now by calling the appropriate tools.""")
            ],
            "csv_path": csv_path,
            "analysis_complete": False,
            "tool_calls_count": 0,
            "final_recommendation": None
        }
        
        # Run the graph
        if verbose:
            print("🤖 LangGraph Agent analyzing data...")
            print("="*70 + "\n")
        
        final_state = self.graph.invoke(initial_state)
        
        if verbose:
            print("\n" + "="*70)
            print(f"✓ Analysis complete!")
            print(f"✓ Tool calls made: {final_state['tool_calls_count']}")
            print()
        
        return final_state["final_recommendation"]
    
    def save_result(self, result: Dict[str, Any], output_path: str = "highcharts_config.json"):
        """Save the result to a JSON file"""
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"✓ Configuration saved to: {output_path}")


# ============================================================
# Example Usage
# ============================================================

def main():
    """Main function with example usage"""
    print("="*70)
    print("CSV to Highcharts - LangGraph Implementation")
    print("="*70)
    
    # Create sample CSV
    print("\n📝 Creating sample CSV...")
    sample_data = pd.DataFrame({
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug'],
        'Sales': [12000, 15000, 14500, 18000, 21000, 19500, 23000, 25000],
        'Expenses': [8000, 9500, 9000, 11000, 13000, 12000, 14500, 15500],
        'Profit': [4000, 5500, 5500, 7000, 8000, 7500, 8500, 9500]
    })
    sample_data.to_csv('sample_sales_data.csv', index=False)
    print("✓ Sample CSV created: sample_sales_data.csv\n")
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set!")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'\n")
        return
    
    # Initialize the graph-based agent
    print("🚀 Initializing LangGraph agent...")
    try:
        agent = CSVToHighchartsGraph(model="gpt-4")
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        print("\nMake sure you have the correct package versions:")
        print("  pip install langchain==0.3.0 langchain-openai==0.2.0 langgraph==0.2.34")
        return
    
    # Analyze CSV
    try:
        result = agent.analyze_csv('sample_sales_data.csv')
        
        # Display results
        if result and "error" not in result:
            print("\n" + "="*70)
            print("📊 ANALYSIS RESULTS")
            print("="*70)
            print(f"\n✓ Chart Type: {result.get('chartType', 'N/A')}")
            print(f"\n✓ Reasoning:\n{result.get('reasoning', 'N/A')}")
            
            if 'dataInsights' in result:
                print(f"\n✓ Data Insights:")
                print(f"  - Row Count: {result['dataInsights'].get('rowCount', 'N/A')}")
                print(f"  - Key Columns: {', '.join(result['dataInsights'].get('keyColumns', []))}")
                print(f"  - Pattern: {result['dataInsights'].get('dataPattern', 'N/A')}")
            
            if 'highchartsConfig' in result:
                print(f"\n✓ Highcharts Config Generated")
                print(f"  (See highcharts_config.json for full configuration)")
            
            # Save result
            agent.save_result(result)
            
            print("\n" + "="*70)
            print("✅ Success! You can now:")
            print("  1. View highcharts_config.json")
            print("  2. Use the config in your web application")
            print("  3. Open example_highcharts.html to see visualization")
            print("="*70)
            
        else:
            print(f"\n⚠️  Error occurred: {result.get('error', 'Unknown error')}")
            if 'raw_content' in result:
                print(f"Raw output: {result['raw_content'][:200]}...")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()