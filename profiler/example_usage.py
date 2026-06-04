"""
example_usage.py — Integration examples for the Data Profiler Agent.

Shows how to:
  1. Profile a CSV file
  2. Profile an Excel file
  3. Profile a SQL table
  4. Use the outputs with a downstream LangChain agent
"""

import os
from dotenv import load_dotenv
load_dotenv()

from agent import run_profiler


# ─────────────────────────────────────────────────────────────────────────── #
# Example 1: Profile a CSV                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

def example_csv():
    result = run_profiler(
        source_type="csv",
        path="./sample_data/orders.csv",
        knowledge_file="./sample_data/orders_knowledge.md",  # optional
    )

    print("=== Profiling complete ===")
    print(f"Completed: {result['completed']}")
    print(f"Errors:    {result['errors']}")
    print(f"Files:     {result['output_path_json']}")
    print()
    print("=== LLM Summary (inject this into your downstream agent) ===")
    print(result["llm_summary"])

    return result


# ─────────────────────────────────────────────────────────────────────────── #
# Example 2: Profile an Excel file                                            #
# ─────────────────────────────────────────────────────────────────────────── #

def example_excel():
    result = run_profiler(
        source_type="excel",
        path="./sample_data/sales_report.xlsx",
        sheet_name="Sheet1",   # or 0 for first sheet
    )
    print(result["llm_summary"])
    return result


# ─────────────────────────────────────────────────────────────────────────── #
# Example 3: Profile a SQL table                                              #
# ─────────────────────────────────────────────────────────────────────────── #

def example_sql():
    result = run_profiler(
        source_type="sql",
        connection_string="postgresql://user:password@localhost:5432/mydb",
        table_name="orders",
    )
    print(result["llm_summary"])
    return result


# ─────────────────────────────────────────────────────────────────────────── #
# Example 4: Use profile output with a downstream LangChain agent            #
# ─────────────────────────────────────────────────────────────────────────── #

def example_downstream_agent(profile_result: dict):
    """
    Shows how to inject the profiler output into a downstream analyst agent.
    The profiler's llm_summary and alias_graph become part of the system prompt.
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    from llm_config import get_llm

    llm_summary = profile_result["llm_summary"]
    alias_graph = profile_result["alias_graph"]

    # Build compact alias section for system prompt
    alias_lines = []
    for col, meta in alias_graph.items():
        concept = meta.get("likely_concept", "")
        if concept:
            alias_lines.append(f"  - `{col}` → {concept}")
    alias_section = "\n".join(alias_lines) if alias_lines else "  (none detected)"

    system_prompt = f"""You are a data analyst assistant. The user's dataset has been
fully profiled below. Use this profile to answer questions accurately even when the
user's language is vague, abbreviated, or indirect.

{llm_summary}

## Column Aliases
{alias_section}

## Query Routing Instructions
- Map vague user terms to actual column names using the profile above.
- When the user says "customers" look for customer-related columns.
- When the user says "revenue" look for amount/value columns.
- When exemplars are truncated, note that the full dataset may contain more values.
- Always reference actual column names in your answers, not the user's informal terms.
"""

    llm = get_llm()

    # Example query
    user_question = "What's the average transaction amount and which customers have the highest spend?"

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_question),
    ])

    print("=== Downstream Agent Response ===")
    print(response.content)


# ─────────────────────────────────────────────────────────────────────────── #
# Run                                                                         #
# ─────────────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    # Change to whichever example fits your data
    result = example_csv()
    example_downstream_agent(result)
