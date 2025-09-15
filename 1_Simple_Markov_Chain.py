"""
ReAct Agent Implementation using LangGraph (Latest API)
A simple reasoning and acting agent that can use tools to answer questions.
"""

from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
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
        result = eval(expression.replace('^', '**'))  # Handle exponents
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

# Method 1: Using the prebuilt create_react_agent (Simplest)
def create_simple_react_agent():
    """Create a ReAct agent using the prebuilt function."""
    return create_react_agent(llm, tools)

# Method 2: Custom implementation with latest API
def create_custom_react_agent():
    """Create a custom ReAct agent with full control."""
    
    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)
    
    def agent_node(state: AgentState):
        """The main agent that decides whether to use tools or respond."""
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def tool_node(state: AgentState):
        """Execute the tools that the agent decided to use."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # Execute each tool call
        tool_messages = []
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                # Find the tool by name
                selected_tool = None
                for tool in tools:
                    if tool.name == tool_call["name"]:
                        selected_tool = tool
                        break
                
                if selected_tool:
                    try:
                        # Invoke the tool directly
                        tool_result = selected_tool.invoke(tool_call["args"])
                        
                        # Create tool message
                        tool_message = ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_call["id"]
                        )
                        tool_messages.append(tool_message)
                        
                    except Exception as e:
                        # Handle tool execution errors
                        error_message = ToolMessage(
                            content=f"Error executing {tool_call['name']}: {str(e)}",
                            tool_call_id=tool_call["id"]
                        )
                        tool_messages.append(error_message)
        
        return {"messages": tool_messages}
    
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
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
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
    
    return workflow.compile()

# Helper function to run the agent
def run_react_agent(query: str, use_prebuilt: bool = True) -> str:
    """Run the ReAct agent with a query."""
    print(f"\n🤖 User Query: {query}")
    print("=" * 50)
    
    # Choose which agent to use
    if use_prebuilt:
        app = create_simple_react_agent()
        print("Using prebuilt ReAct agent")
    else:
        app = create_custom_react_agent()
        print("Using custom ReAct agent")
    
    # Initialize the state
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }
    
    # Run the agent
    result = app.invoke(initial_state)
    
    # Print the reasoning process
    print("\n💭 Agent Reasoning Process:")
    for i, message in enumerate(result["messages"]):
        if isinstance(message, HumanMessage):
            print(f"  Human: {message.content}")
        elif isinstance(message, AIMessage):
            if hasattr(message, 'tool_calls') and message.tool_calls:
                print(f"  Agent: I need to use tools to answer this.")
                for tool_call in message.tool_calls:
                    print(f"    🔧 Calling {tool_call['name']} with: {tool_call['args']}")
            else:
                print(f"  Agent: {message.content}")
        elif isinstance(message, ToolMessage):
            print(f"    📋 Tool result: {message.content}")
    
    # Extract and return the final response
    final_message = result["messages"][-1]
    return final_message.content if hasattr(final_message, 'content') else str(final_message)

# Streaming version for real-time output
def run_react_agent_streaming(query: str, use_prebuilt: bool = True):
    """Run the ReAct agent with streaming output."""
    print(f"\n🤖 User Query: {query}")
    print("=" * 50)
    
    # Choose which agent to use
    if use_prebuilt:
        app = create_simple_react_agent()
    else:
        app = create_custom_react_agent()
    
    # Initialize the state
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }
    
    print("\n💭 Agent Process (Streaming):")
    
    # Stream the execution
    for chunk in app.stream(initial_state):
        for node_name, node_output in chunk.items():
            print(f"\n--- {node_name.upper()} ---")
            for message in node_output.get("messages", []):
                if isinstance(message, HumanMessage):
                    print(f"Human: {message.content}")
                elif isinstance(message, AIMessage):
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        print("Agent: Planning to use tools...")
                        for tool_call in message.tool_calls:
                            print(f"  🔧 {tool_call['name']}: {tool_call['args']}")
                    else:
                        print(f"Agent: {message.content}")
                elif isinstance(message, ToolMessage):
                    print(f"Tool Result: {message.content}")

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
    
    print("🚀 ReAct Agent Demo (Latest LangGraph API)")
    print("This agent can reason about queries and use tools when needed.\n")
    
    for query in test_queries[:2]:  # Test first 2 queries
        try:
            print(f"\n{'='*70}")
            print("TESTING WITH PREBUILT AGENT:")
            response = run_react_agent(query, use_prebuilt=True)
            print(f"\n✅ Final Answer: {response}")
            
            print(f"\n{'='*70}")
            print("TESTING WITH CUSTOM AGENT:")
            response = run_react_agent(query, use_prebuilt=False)
            print(f"\n✅ Final Answer: {response}")
            
            print(f"\n{'='*70}")
            print("TESTING STREAMING VERSION:")
            run_react_agent_streaming(query, use_prebuilt=True)
            
            print("\n" + "="*70)
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Make sure to set your OPENAI_API_KEY environment variable")
            break

# Additional utility: Interactive mode
def interactive_mode():
    """Run the agent in interactive mode."""
    print("\n🎮 Interactive ReAct Agent")
    print("Type 'quit' to exit, 'custom' to switch to custom agent, 'prebuilt' for prebuilt agent")
    print("Current mode: prebuilt\n")
    
    use_prebuilt = True
    
    while True:
        query = input("👤 You: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        elif query.lower() == 'custom':
            use_prebuilt = False
            print("🔧 Switched to custom agent")
            continue
        elif query.lower() == 'prebuilt':
            use_prebuilt = True
            print("⚡ Switched to prebuilt agent")
            continue
        elif query.lower() == 'stream':
            query = input("Enter query for streaming: ").strip()
            if query:
                try:
                    run_react_agent_streaming(query, use_prebuilt)
                except Exception as e:
                    print(f"❌ Error: {e}")
            continue
        
        if query:
            try:
                response = run_react_agent(query, use_prebuilt)
                print(f"\n🤖 Agent: {response}\n")
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Make sure to set your OPENAI_API_KEY environment variable\n")

# Uncomment to run interactive mode:
# interactive_mode()