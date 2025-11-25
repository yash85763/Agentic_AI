“””
Example: Using the CSV to Highcharts Chatbot
This demonstrates different use cases
“””

from csv_to_highcharts import CSVToHighchartsBot
import pandas as pd
import json

# Initialize the bot

bot = CSVToHighchartsBot(model=“gpt-4”)

# ============================================================

# EXAMPLE 1: Sales Data (Time Series)

# ============================================================

print(”\n” + “=”*60)
print(“EXAMPLE 1: Monthly Sales Data”)
print(”=”*60)

sales_data = pd.DataFrame({
‘Month’: [‘Jan’, ‘Feb’, ‘Mar’, ‘Apr’, ‘May’, ‘Jun’, ‘Jul’, ‘Aug’],
‘Sales’: [12000, 15000, 14500, 18000, 21000, 19500, 23000, 25000],
‘Target’: [15000, 15000, 15000, 20000, 20000, 20000, 25000, 25000]
})
sales_data.to_csv(‘sales_data.csv’, index=False)

result = bot.analyze_csv(‘sales_data.csv’)
print(f”\n✓ Recommended: {result[‘chartType’]}”)
print(f”✓ Reasoning: {result[‘reasoning’]}”)
bot.save_result(result, ‘sales_chart_config.json’)

# ============================================================

# EXAMPLE 2: Product Distribution (Categorical)

# ============================================================

print(”\n” + “=”*60)
print(“EXAMPLE 2: Product Market Share”)
print(”=”*60)

product_data = pd.DataFrame({
‘Product’: [‘Product A’, ‘Product B’, ‘Product C’, ‘Product D’, ‘Product E’],
‘Market_Share’: [35, 25, 20, 12, 8]
})
product_data.to_csv(‘product_data.csv’, index=False)

result = bot.analyze_csv(‘product_data.csv’)
print(f”\n✓ Recommended: {result[‘chartType’]}”)
print(f”✓ Reasoning: {result[‘reasoning’]}”)
bot.save_result(result, ‘product_chart_config.json’)

# ============================================================

# EXAMPLE 3: Correlation Data (Scatter Plot)

# ============================================================

print(”\n” + “=”*60)
print(“EXAMPLE 3: Temperature vs Ice Cream Sales”)
print(”=”*60)

correlation_data = pd.DataFrame({
‘Temperature_F’: [65, 70, 75, 80, 85, 90, 95, 75, 68, 72],
‘Ice_Cream_Sales’: [150, 200, 280, 350, 420, 480, 550, 270, 180, 220]
})
correlation_data.to_csv(‘correlation_data.csv’, index=False)

result = bot.analyze_csv(‘correlation_data.csv’)
print(f”\n✓ Recommended: {result[‘chartType’]}”)
print(f”✓ Reasoning: {result[‘reasoning’]}”)
bot.save_result(result, ‘correlation_chart_config.json’)

# ============================================================

# EXAMPLE 4: Using Your Own CSV

# ============================================================

print(”\n” + “=”*60)
print(“EXAMPLE 4: Using Your Own CSV”)
print(”=”*60)
print(”””
To use your own CSV file:

1. Make sure your CSV has clear column names
1. Call the bot with your file:
   
   result = bot.analyze_csv(‘your_file.csv’)
1. Get the Highcharts config:
   
   config = result[‘highchartsConfig’]
1. Use it in your HTML:
   
    <script src="https://code.highcharts.com/highcharts.js"></script>
   
    <div id="container"></div>
    <script>
        Highcharts.chart('container', YOUR_CONFIG_HERE);
    </script>

“””)

print(”\n✓ All examples completed!”)
print(“✓ Check the generated JSON files for Highcharts configurations”)