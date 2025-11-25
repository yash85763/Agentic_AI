import os
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.agents.agent import AgentExecutor
from langchain.agents.openai_functions_agent.base import create_openai_functions_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage


class CSVDataAnalyzer:
    """
    Provides tools for LLM to analyze CSV data
    """

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
            return "No data loaded"

        info = {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "column_names": list(self.df.columns),
            "memory_usage_mb": round(self.df.memory_usage(deep=True).sum() / 1024**2, 2)
        }
        return json.dumps(info, indent=2)

    def get_data_types(self) -> str:
        """Get data types of all columns"""
        if self.df is None:
            return "No data loaded"

        dtypes = {}
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            # Detect if numeric column might be categorical
            if dtype in ['int64', 'float64'] and self.df[col].nunique() < 10:
                dtype = f"{dtype} (possibly categorical)"
            # Detect if object column might be datetime
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
            return "No data loaded"

        if column_name not in self.df.columns:
            return f"Column '{column_name}' not found. Available columns: {list(self.df.columns)}"

        col = self.df[column_name]
        stats = {
            "column_name": column_name,
            "data_type": str(col.dtype),
            "non_null_count": int(col.count()),
            "null_count": int(col.isnull().sum()),
            "unique_values": int(col.nunique())
        }

        # Numeric statistics
        if pd.api.types.is_numeric_dtype(col):
            stats.update({
                "min": float(col.min()) if not pd.isna(col.min()) else None,
                "max": float(col.max()) if not pd.isna(col.max()) else None,
                "mean": float(col.mean()) if not pd.isna(col.mean()) else None,
                "median": float(col.median()) if not pd.isna(col.median()) else None,
                "std": float(col.std()) if not pd.isna(col.std()) else None
            })

        # Categorical statistics
        if col.nunique() < 20:  # If less than 20 unique values, show value counts
            value_counts = col.value_counts().head(10).to_dict()
            stats["top_values"] = {str(k): int(v) for k, v in value_counts.items()}

        return json.dumps(stats, indent=2)

    def get_sample_data(self, n_rows: int = 5) -> str:
        """Get sample rows from the dataset"""
        if self.df is None:
            return "No data loaded"

        n_rows = min(n_rows, len(self.df))
        sample = self.df.head(n_rows).to_dict(orient='records')
        return json.dumps(sample, indent=2, default=str)

    def get_correlation_info(self) -> str:
        """Get correlation information for numeric columns"""
        if self.df is None:
            return "No data loaded"

        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        if len(numeric_cols) < 2:
            return "Not enough numeric columns for correlation analysis"

        corr_matrix = self.df[numeric_cols].corr()

        # Find strong correlations (> 0.5 or < -0.5)
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
            return "No data loaded"

        time_columns = []
        for col in self.df.columns:
            # Check if column name suggests time
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
            return "No data loaded"

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


class CSVToHighchartsAgent:
    """
    Agent-based chatbot that uses tools to analyze CSV and recommend Highcharts config
    """

    def __init__(self, api_key: Optional[str] = OPENAI_API_KEY, model: str = "gpt-4"):
        """
        Initialize the agent with tools

        Args:
            api_key: OpenAI API key
            model: Model to use (gpt-4 or gpt-3.5-turbo)
        """
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        # Initialize data analyzer
        self.analyzer = CSVDataAnalyzer()

        # Initialize LLM
        self.llm = ChatOpenAI(model=model, temperature=0.3)

        # Create tools for the agent
        self.tools = self._create_tools()

        # Create the agent
        self.agent_executor = self._create_agent()

    def _create_tools(self) -> List[Tool]:
        """Create tools that the LLM can use to analyze data"""
        return [
            Tool(
                name="get_basic_info",
                func=self.analyzer.get_basic_info,
                description="""Get basic information about the dataset including:
                - Total number of rows
                - Total number of columns
                - Column names
                - Memory usage
                Use this first to understand the dataset structure."""
            ),
            Tool(
                name="get_data_types",
                func=self.analyzer.get_data_types,
                description="""Get data types of all columns in the dataset.
                This helps identify numeric, categorical, datetime, and other column types.
                Use this to understand what kind of data each column contains."""
            ),
            Tool(
                name="get_column_statistics",
                func=self.analyzer.get_column_statistics,
                description="""Get detailed statistics for a SPECIFIC column.
                Input should be the exact column name as a string.
                Returns: min, max, mean, median, std for numeric columns,
                or value counts for categorical columns.
                Call this for each important column you want to analyze."""
            ),
            Tool(
                name="get_sample_data",
                func=lambda: self.analyzer.get_sample_data(5),
                description="""Get sample rows (first 5) from the dataset.
                Use this to see actual data values and understand the data better.
                Helpful for understanding data patterns and relationships."""
            ),
            Tool(
                name="get_correlation_info",
                func=self.analyzer.get_correlation_info,
                description="""Get correlation information between numeric columns.
                Use this to identify relationships between variables.
                Helps determine if scatter plots or multi-series charts would be useful."""
            ),
            Tool(
                name="detect_time_series",
                func=self.analyzer.detect_time_series,
                description="""Detect if the dataset contains time series data.
                Identifies columns that contain dates or timestamps.
                Use this to determine if line or area charts would be appropriate."""
            ),
            Tool(
                name="get_all_columns_summary",
                func=self.analyzer.get_all_columns_summary,
                description="""Get a quick summary of ALL columns at once.
                Shows data type, null count, and unique values for each column.
                Use this for a comprehensive overview before diving into specifics."""
            )
        ]

    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent with tools"""

        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data visualization expert. Your task is to analyze CSV data
            using the available tools and recommend the best Highcharts configuration.

            ANALYSIS PROCESS:
            1. Start by using get_basic_info and get_all_columns_summary to understand the dataset
            2. Use get_data_types to understand column types
            3. Use detect_time_series to check for time-based data
            4. Use get_sample_data to see actual values
            5. For key columns, use get_column_statistics to get detailed stats
            6. Use get_correlation_info if there are multiple numeric columns
            7. Based on your analysis, determine the best chart type

            CHART TYPE SELECTION GUIDELINES:
            - Line/Area charts: Time series data, trends over time
            - Column/Bar charts: Categorical comparisons, grouped data
            - Pie/Donut charts: Part-to-whole relationships, percentages (max 7-8 categories)
            - Scatter plots: Correlation between two numeric variables
            - Multiple series: When comparing 2-4 related metrics

            FINAL OUTPUT:
            After your analysis, provide your recommendation as a JSON object with this EXACT structure:
            {{
                "chartType": "line|column|bar|pie|scatter|area|etc",
                "reasoning": "Detailed explanation based on the data analysis",
                "highchartsConfig": {{
                    // Complete Highcharts configuration
                }},
                "dataInsights": {{
                    "rowCount": 123,
                    "keyColumns": ["col1", "col2"],
                    "dataPattern": "time-series|categorical|correlation"
                }}
            }}

            IMPORTANT: Use the tools systematically. Don't skip tools - each provides valuable insights."""),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        # Create agent
        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        # Create agent executor
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,  # Set to True to see tool calls
            max_iterations=15,  # Allow multiple tool calls
            handle_parsing_errors=True
        )

    def analyze_csv(self, csv_path: str) -> Dict[str, Any]:
        """
        Analyze CSV file and get Highcharts recommendation

        Args:
            csv_path: Path to CSV file

        Returns:
            Dictionary with chart recommendation and config
        """
        # Load the CSV data
        print(f"\n📂 Loading CSV: {csv_path}")
        self.analyzer.load_csv(csv_path)
        print(f"✓ Loaded {len(self.analyzer.df)} rows, {len(self.analyzer.df.columns)} columns\n")

        # Prepare the input for the agent
        input_text = f"""Analyze the loaded CSV dataset and recommend the best Highcharts visualization.

The CSV file is already loaded. Use the available tools to:
1. Understand the data structure and types
2. Analyze key columns and their statistics
3. Identify patterns (time-series, categorical, correlations)
4. Recommend the optimal chart type
5. Generate a complete Highcharts configuration

Be thorough in your analysis - use multiple tools to gather comprehensive insights."""

        # Run the agent
        print("🤖 Agent analyzing data...\n")
        print("="*60)
        result = self.agent_executor.invoke({"input": input_text})
        print("="*60)

        # Parse the output
        output = result['output']

        # Try to extract JSON from the output
        try:
            # Look for JSON in the response
            if '```json' in output:
                json_start = output.find('```json') + 7
                json_end = output.find('```', json_start)
                json_str = output[json_start:json_end].strip()
            elif '```' in output:
                json_start = output.find('```') + 3
                json_end = output.find('```', json_start)
                json_str = output[json_start:json_end].strip()
            else:
                # Try to find JSON object directly
                json_start = output.find('{')
                json_end = output.rfind('}') + 1
                json_str = output[json_start:json_end]

            parsed_result = json.loads(json_str)
            return parsed_result

        except json.JSONDecodeError as e:
            print(f"\n⚠️  Warning: Could not parse JSON from agent output")
            print(f"Raw output:\n{output}\n")
            # Return a structured error response
            return {
                "error": "Failed to parse JSON",
                "raw_output": output,
                "chartType": "unknown",
                "reasoning": "Error in parsing agent response"
            }

    def save_result(self, result: Dict[str, Any], output_path: str = "highcharts_config.json"):
        """Save the result to a JSON file"""
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Configuration saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("CSV to Highcharts Agent - With Tool Calling")
    print("="*60)

    # Create sample CSV
    print("\n📝 Creating sample CSV...")
    sample_data = pd.DataFrame({
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug'],
        'Sales': [12000, 15000, 14500, 18000, 21000, 19500, 23000, 25000],
        'Expenses': [8000, 9500, 9000, 11000, 13000, 12000, 14500, 15500],
        'Profit': [4000, 5500, 5500, 7000, 8000, 7500, 8500, 9500]
    })
    sample_data.to_csv('sample_sales_data.csv', index=False)
    print("✓ Sample CSV created: sample_sales_data.csv")

    # Initialize agent
    print("\n🚀 Initializing agent...")
    agent = CSVToHighchartsAgent(model="gpt-4")

    # Analyze CSV
    try:
        result = agent.analyze_csv('sample_sales_data.csv')

        # Display results
        print("\n" + "="*60)
        print("📊 ANALYSIS RESULTS")
        print("="*60)

        if "error" not in result:
            print(f"\n✓ Chart Type: {result.get('chartType', 'N/A')}")
            print(f"\n✓ Reasoning:\n{result.get('reasoning', 'N/A')}")

            if 'dataInsights' in result:
                print(f"\n✓ Data Insights:")
                print(json.dumps(result['dataInsights'], indent=2))

            print(f"\n✓ Highcharts Config Generated:")
            print(json.dumps(result.get('highchartsConfig', {}), indent=2)[:500] + "...")

            # Save result
            agent.save_result(result)
        else:
            print(f"\n⚠️  Error occurred: {result.get('error')}")
            print(f"Raw output: {result.get('raw_output', 'N/A')[:500]}")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nMake sure to set OPENAI_API_KEY environment variable!")