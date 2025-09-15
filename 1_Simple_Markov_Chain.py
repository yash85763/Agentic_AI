"""
ReAct Agent Implementation using LangGraph
A simple reasoning and acting agent that can use tools to answer questions.
"""

from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.prebuilt.tool_executor import ToolExecutor
import json

# Define the agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
# Define some example tools
@tool
def search_web(query: str) -> str:
    """Search the web for information about a query."""
    # This is a mock implementation - replace with actual web search
    return f"Search results for '{query}': This is mock search data. In reality, this would return web search results."

@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions safely."""
    try:
        # Simple calculator - in production, use a safer evaluation method
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    # Mock weather data - replace with actual weather API
    return f"Weather in {location}: Sunny, 72°F with light clouds"

# Initialize the LLM and tools
llm = ChatOpenAI(model="gpt-4", temperature=0)
tools = [search_web, calculator, get_weather]
tool_executor = ToolExecutor(tools)

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)

# Define the agent node
def agent(state: AgentState):
    """The main agent that decides whether to use tools or respond."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Define the tool execution node
def execute_tools(state: AgentState):
    """Execute the tools that the agent decided to use."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Execute each tool call
    tool_results = []
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_result = tool_executor.invoke(
                ToolInvocation(
                    tool=tool_call["name"],
                    tool_input=tool_call["args"]
                )
            )
            tool_results.append({
                "tool_call_id": tool_call["id"],
                "name": tool_call["name"],
                "content": str(tool_result)
            })
    
    # Create tool messages
    tool_messages = []
    for result in tool_results:
        tool_messages.append({
            "role": "tool",
            "content": result["content"],
            "tool_call_id": result["tool_call_id"]
        })
    
    return {"messages": tool_messages}

# Define the routing logic
def should_continue(state: AgentState) -> str:
    """Determine if we should continue with tool execution or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If there are tool calls, execute tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    # Otherwise, end the conversation
    return END

# Create the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", agent)
workflow.add_node("tools", execute_tools)

# Set entry point
workflow.set_entry_point("agent")

# Add edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)
workflow.add_edge("tools", "agent")

# Compile the graph
app = workflow.compile()

# Helper function to run the agent
def run_react_agent(query: str) -> str:
    """Run the ReAct agent with a query."""
    print(f"\n🤖 User Query: {query}")
    print("=" * 50)
    
    # Initialize the state
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }
    
    # Run the agent
    result = app.invoke(initial_state)
    
    # Extract and return the final response
    final_message = result["messages"][-1]
    
    # Print the reasoning process
    print("\n💭 Agent Reasoning Process:")
    for i, message in enumerate(result["messages"]):
        if hasattr(message, 'content'):
            if i == 0:
                print(f"  Human: {message.content}")
            elif hasattr(message, 'tool_calls') and message.tool_calls:
                print(f"  Agent: I need to use tools to answer this.")
                for tool_call in message.tool_calls:
                    print(f"    🔧 Calling {tool_call['name']} with: {tool_call['args']}")
            elif hasattr(message, 'role') and message.role == 'tool':
                print(f"    📋 Tool result: {message.content}")
            else:
                print(f"  Agent: {message.content}")
    
    return final_message.content if hasattr(final_message, 'content') else str(final_message)

# Example usage and testing
if __name__ == "__main__":
    # Set your OpenAI API key
    import os
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
    # Test queries that demonstrate ReAct pattern
    test_queries = [
        "What's the weather like in New York?",
        "Calculate 25 * 47 + 123",
        "Search for information about Python programming",
        "What's 15% of 240 and what's the weather in London?"
    ]
    
    print("🚀 ReAct Agent Demo")
    print("This agent can reason about queries and use tools when needed.\n")
    
    for query in test_queries:
        try:
            response = run_react_agent(query)
            print(f"\n✅ Final Answer: {response}")
            print("\n" + "="*70)
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Make sure to set your OPENAI_API_KEY environment variable")
            break

# Additional utility: Interactive mode
def interactive_mode():
    """Run the agent in interactive mode."""
    print("\n🎮 Interactive ReAct Agent")
    print("Type 'quit' to exit\n")
    
    while True:
        query = input("👤 You: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if query:
            try:
                response = run_react_agent(query)
                print(f"\n🤖 Agent: {response}\n")
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Make sure to set your OPENAI_API_KEY environment variable\n")

# Uncomment to run interactive mode:
# interactive_mode()