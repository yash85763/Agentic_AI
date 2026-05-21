Yes — then don’t call it a “coding agent” in the general sense.

Call it:

Data Analysis Agent

Its job is not to build software.

Its job is:

Given a PDF, Excel, CSV, or structured file, understand it, write safe analysis code when needed, run that code in a sandbox, and answer the user’s question with evidence.

This is much simpler and more useful.

⸻

1. New Agent Definition

Data Analysis Agent =
  File Understanding Core
  + Query Understanding
  + Sandbox Code Executor
  + PDF Reader
  + Excel/Table Engine
  + Validation Layer
  + Answer Generator

The agent should only operate on uploaded input files.

No git.
No repo editing.
No app development.
No shell freedom except inside controlled sandbox.

⸻

2. Agent Architecture

                  User Query
                      |
                      v
              Query Understanding
                      |
          ┌───────────┴────────────┐
          |                        |
       PDF File                Excel/File
          |                        |
          v                        v
  PDF Extraction Engine     Table Profiling Engine
          |                        |
          v                        v
  Document Understanding    Data Understanding
          |                        |
          └───────────┬────────────┘
                      v
              Analysis Planner
                      |
                      v
              Sandbox Code Runner
                      |
                      v
              Result Validator
                      |
                      v
              Final Answer

⸻

3. The Agent Core Should Be Small

class DataAnalysisAgent:
    def __init__(
        self,
        file_reader,
        query_planner,
        sandbox,
        validator,
        answer_generator,
        memory=None,
        knowledge=None,
    ):
        self.file_reader = file_reader
        self.query_planner = query_planner
        self.sandbox = sandbox
        self.validator = validator
        self.answer_generator = answer_generator
        self.memory = memory
        self.knowledge = knowledge
    def run(self, user_query: str, uploaded_file: str):
        file_profile = self.file_reader.profile(uploaded_file)
        plan = self.query_planner.create_plan(
            query=user_query,
            file_profile=file_profile
        )
        result = self.sandbox.execute(plan)
        validated_result = self.validator.validate(
            result=result,
            file_profile=file_profile,
            plan=plan
        )
        return self.answer_generator.generate(
            query=user_query,
            result=validated_result,
            file_profile=file_profile
        )

⸻

4. Required Plug-in Ports

A. File Reader Port

Handles file type detection.

class FileReaderPort:
    def profile(self, file_path: str) -> dict:
        ...

Implementations:

PDFReader
ExcelReader
CSVReader
ParquetReader
JSONReader

⸻

B. PDF Reader

PDF needs two separate jobs:

PDF Reading = extracting raw text/tables/images
PDF Understanding = interpreting structure and meaning

PDF module:

PDFReader
  - detect text-based vs scanned
  - extract text
  - extract tables
  - preserve page numbers
  - preserve headings
  - chunk content
  - create searchable document index

For PDFs, the agent should answer with page-level evidence.

Example:

According to page 4, the contract caps travel reimbursement at $5,000 per quarter.

⸻

C. Excel/Table Reader

Excel module should inspect:

sheets
columns
data types
merged cells
hidden rows/columns
formulas
named ranges
empty rows
tables
pivots if available
cell coordinates

For Excel, the agent needs two modes:

Cell Mode:
  “What is in cell B14?”
Table Mode:
  “Pivot this by region and month”

⸻

5. Excel Capabilities

The Excel agent should support:

read specific cell
read row/column range
extract subtable
filter rows
sort data
groupby aggregation
pivot table
join sheets
detect headers
clean missing values
date parsing
formula inspection
statistical summary
correlation
regression
z-score
IQR outlier detection
forecasting, later

⸻

6. Sandbox Code Runner

The sandbox is the most important part.

The LLM should not calculate directly.

It should produce a structured execution plan.

Example:

{
  "operation": "pivot_table",
  "file_type": "excel",
  "sheet": "Sales",
  "index": ["Region"],
  "columns": ["Month"],
  "values": ["Revenue"],
  "aggfunc": "sum"
}

Then your backend converts this into deterministic Python code:

import pandas as pd
df = pd.read_excel(file_path, sheet_name="Sales")
result = pd.pivot_table(
    df,
    index=["Region"],
    columns=["Month"],
    values=["Revenue"],
    aggfunc="sum"
)
print(result.to_json())

The best design is:

LLM creates intent/plan.
System creates safe code.
Sandbox executes code.
Validator checks result.
LLM explains result.

Not:

LLM writes arbitrary Python freely.

⸻

7. Safer Execution Pattern

Use this:

Natural Language Query
        ↓
Structured Analysis Plan
        ↓
Plan Validation
        ↓
Code Generation from Trusted Templates
        ↓
Sandbox Execution
        ↓
Result Validation
        ↓
Natural Language Answer

This gives you accuracy and safety.

⸻

8. Agent Should Have These Tools Only

read_pdf_text
read_pdf_tables
search_pdf_chunks
read_excel_metadata
read_excel_cell
read_excel_range
profile_table
run_pandas_operation
run_statistical_analysis
generate_chart_data
execute_in_sandbox
validate_result

No general tools unless needed.

⸻

9. Skills Should Be Modular

Skills are procedures, not tools.

For this agent:

PDF Question Answering Skill
Excel Cell Lookup Skill
Table Aggregation Skill
Pivot Analysis Skill
Statistical Analysis Skill
Data Cleaning Skill
Chart Preparation Skill
Validation Skill

Example skill:

name: pivot_analysis
description: Answer pivot-style analytical questions over tabular files.
workflow:
  - identify target sheet
  - identify grouping columns
  - identify value column
  - identify aggregation function
  - generate structured pivot plan
  - run pivot in sandbox
  - validate row counts and totals
  - explain result

⸻

10. Memory Should Be Minimal at First

Do not build complex human memory yet.

For this agent, memory should only store:

current uploaded file profile
sheet summaries
detected table schemas
user’s previous questions in current session
successful analysis plans
known file quirks

Example:

Memory:
  - Sheet "Sales 2024" uses row 3 as header.
  - Column "Amt" means "Revenue".
  - User prefers results as tables.

That is enough.

⸻

11. Knowledge Should Be Optional

Knowledge can include:

domain glossary
company data dictionary
metric definitions
formula definitions
accounting rules
statistical method explanations

Example:

"Net Revenue = Gross Revenue - Refunds - Discounts"

But the agent should still work without knowledge.

⸻

12. Final Simplified Agent

InputDataAgent
  Ports:
    File Port
    Analysis Plan Port
    Sandbox Port
    Validation Port
    Memory Port
    Knowledge Port
  File Types:
    PDF
    Excel
    CSV
    JSON
    Parquet
  Main Jobs:
    read
    understand
    extract
    transform
    compute
    validate
    answer

⸻

Better Name

I would not call it Coding Agent.

Better names:

Data Analyst Agent
Document + Data Agent
Sandbox Analyst Agent
Input Intelligence Agent
File Analysis Agent

My preferred name:

Sandbox Analyst Agent

Because the core innovation is:

It can understand files, plan analysis, execute code safely in sandbox, and answer from computed evidence.