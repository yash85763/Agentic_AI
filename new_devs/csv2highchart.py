import os
import json
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

class CSVToHighchartsBot:
“””
A chatbot that analyzes CSV data and recommends Highcharts configuration
“””

```
def __init__(self, api_key=None, model="gpt-4"):
    """
    Initialize the chatbot with LangChain LLM
    
    Args:
        api_key: OpenAI API key (or set OPENAI_API_KEY env variable)
        model: Model to use (default: gpt-4, can use gpt-3.5-turbo for cheaper option)
    """
    # Set API key
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Initialize LLM
    self.llm = ChatOpenAI(
        model=model,
        temperature=0.3,  # Lower temperature for more consistent JSON output
    )
    
    # Create the prompt template
    self.prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a data visualization expert. Your task is to analyze CSV data 
        and recommend the best Highcharts configuration for visualizing it.
        
        You must return ONLY valid JSON in this exact format:
        {{
            "chartType": "line|bar|column|pie|scatter|area|etc",
            "reasoning": "Brief explanation of why this chart type is best",
            "highchartsConfig": {{
                // Complete Highcharts configuration object
            }}
        }}
        
        Consider:
        - Data types (numeric, categorical, time-series)
        - Number of series
        - Data distribution
        - Best practices for data visualization
        
        Return ONLY the JSON, no markdown code blocks or extra text."""),
        ("human", """Here is the CSV data:
```

Column Names: {columns}
Data Types: {dtypes}
Sample Data (first 5 rows):
{sample_data}

Full Data Summary:
{data_summary}

Please analyze this data and provide the best Highcharts configuration as JSON.”””)
])

```
def read_csv(self, csv_path):
    """
    Read and analyze CSV file
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        pandas DataFrame
    """
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        raise Exception(f"Error reading CSV: {str(e)}")

def analyze_csv(self, csv_path):
    """
    Main function to analyze CSV and get Highcharts config
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        dict: JSON response with chart type, reasoning, and Highcharts config
    """
    # Read CSV
    df = self.read_csv(csv_path)
    
    # Prepare data summary
    columns = list(df.columns)
    dtypes = df.dtypes.to_dict()
    dtypes_str = {k: str(v) for k, v in dtypes.items()}
    sample_data = df.head(5).to_string()
    data_summary = df.describe(include='all').to_string()
    
    # Create the prompt
    formatted_prompt = self.prompt.format_messages(
        columns=columns,
        dtypes=dtypes_str,
        sample_data=sample_data,
        data_summary=data_summary
    )
    
    # Get response from LLM
    response = self.llm.invoke(formatted_prompt)
    
    # Parse JSON response
    try:
        # Clean the response (remove markdown code blocks if present)
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        result = json.loads(content)
        return result
    except json.JSONDecodeError as e:
        print("Raw response:", response.content)
        raise Exception(f"Failed to parse JSON response: {str(e)}")

def save_result(self, result, output_path="highcharts_config.json"):
    """
    Save the result to a JSON file
    
    Args:
        result: Dictionary containing the Highcharts config
        output_path: Path to save the JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"✓ Highcharts configuration saved to: {output_path}")
```

# Example usage

if **name** == “**main**”:
# Example 1: Create sample CSV for demonstration
print(“Creating sample CSV data…”)
sample_data = pd.DataFrame({
‘Month’: [‘Jan’, ‘Feb’, ‘Mar’, ‘Apr’, ‘May’, ‘Jun’],
‘Sales’: [1200, 1900, 1500, 2100, 2400, 2000],
‘Expenses’: [800, 1100, 900, 1300, 1500, 1200]
})
sample_data.to_csv(‘sample_data.csv’, index=False)
print(“✓ Sample CSV created: sample_data.csv\n”)

```
# Initialize the chatbot
# Option 1: Pass API key directly
# bot = CSVToHighchartsBot(api_key="your-api-key-here")

# Option 2: Use environment variable (recommended)
# Set OPENAI_API_KEY in your environment
bot = CSVToHighchartsBot(model="gpt-4")

# Analyze CSV and get Highcharts config
print("Analyzing CSV data...")
try:
    result = bot.analyze_csv('sample_data.csv')
    
    # Display results
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    print(f"\nRecommended Chart Type: {result['chartType']}")
    print(f"\nReasoning: {result['reasoning']}")
    print("\nHighcharts Configuration:")
    print(json.dumps(result['highchartsConfig'], indent=2))
    
    # Save to file
    bot.save_result(result)
    
except Exception as e:
    print(f"Error: {str(e)}")
    print("\nMake sure to set your OPENAI_API_KEY environment variable!")
    print("Example: export OPENAI_API_KEY='your-key-here'")
```