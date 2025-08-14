# requirements.txt

“””
fastmcp
langgraph
langchain
arxiv
python-dotenv
openai
anthropic
“””

# arxiv_server.py

import asyncio
import arxiv
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastmcp import FastMCP
from pydantic import BaseModel

class ArxivSearchParams(BaseModel):
query: str
max_results: int = 5
sort_by: str = “relevance”  # relevance, lastUpdatedDate, submittedDate
sort_order: str = “descending”  # ascending, descending

class ArxivPaper(BaseModel):
title: str
authors: List[str]
summary: str
published: str
updated: str
entry_id: str
pdf_url: str
categories: List[str]

class ArxivMCPServer:
def **init**(self):
self.mcp = FastMCP(“ArXiv Research Server”)
self.setup_tools()

```
def setup_tools(self):
    @self.mcp.tool()
    async def search_arxiv_papers(params: ArxivSearchParams) -> List[ArxivPaper]:
        """
        Search for academic papers on ArXiv.
        
        Args:
            params: Search parameters including query, max_results, sort_by, and sort_order
        
        Returns:
            List of ArXiv papers matching the search criteria
        """
        try:
            # Configure arxiv client
            client = arxiv.Client()
            
            # Map sort parameters
            sort_map = {
                "relevance": arxiv.SortCriterion.Relevance,
                "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
                "submittedDate": arxiv.SortCriterion.SubmittedDate
            }
            
            order_map = {
                "ascending": arxiv.SortOrder.Ascending,
                "descending": arxiv.SortOrder.Descending
            }
            
            search = arxiv.Search(
                query=params.query,
                max_results=params.max_results,
                sort_by=sort_map.get(params.sort_by, arxiv.SortCriterion.Relevance),
                sort_order=order_map.get(params.sort_order, arxiv.SortOrder.Descending)
            )
            
            papers = []
            for result in client.results(search):
                paper = ArxivPaper(
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    summary=result.summary,
                    published=result.published.isoformat(),
                    updated=result.updated.isoformat(),
                    entry_id=result.entry_id,
                    pdf_url=result.pdf_url,
                    categories=result.categories
                )
                papers.append(paper)
            
            return papers
            
        except Exception as e:
            raise Exception(f"Error searching ArXiv: {str(e)}")
    
    @self.mcp.tool()
    async def get_paper_details(entry_id: str) -> ArxivPaper:
        """
        Get detailed information about a specific ArXiv paper by its entry ID.
        
        Args:
            entry_id: The ArXiv entry ID (e.g., "2301.07041")
        
        Returns:
            Detailed information about the paper
        """
        try:
            client = arxiv.Client()
            search = arxiv.Search(id_list=[entry_id])
            
            for result in client.results(search):
                return ArxivPaper(
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    summary=result.summary,
                    published=result.published.isoformat(),
                    updated=result.updated.isoformat(),
                    entry_id=result.entry_id,
                    pdf_url=result.pdf_url,
                    categories=result.categories
                )
            
            raise Exception(f"Paper with ID {entry_id} not found")
            
        except Exception as e:
            raise Exception(f"Error fetching paper details: {str(e)}")
    
    @self.mcp.tool()
    async def get_papers_by_author(author_name: str, max_results: int = 10) -> List[ArxivPaper]:
        """
        Search for papers by a specific author.
        
        Args:
            author_name: Name of the author to search for
            max_results: Maximum number of papers to return
        
        Returns:
            List of papers by the specified author
        """
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=f"au:{author_name}",
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            papers = []
            for result in client.results(search):
                paper = ArxivPaper(
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    summary=result.summary,
                    published=result.published.isoformat(),
                    updated=result.updated.isoformat(),
                    entry_id=result.entry_id,
                    pdf_url=result.pdf_url,
                    categories=result.categories
                )
                papers.append(paper)
            
            return papers
            
        except Exception as e:
            raise Exception(f"Error searching papers by author: {str(e)}")
    
    @self.mcp.tool()
    async def get_recent_papers_by_category(category: str, days_back: int = 7, max_results: int = 10) -> List[ArxivPaper]:
        """
        Get recent papers from a specific ArXiv category.
        
        Args:
            category: ArXiv category (e.g., "cs.AI", "cs.LG", "physics.quant-ph")
            days_back: Number of days to look back
            max_results: Maximum number of papers to return
        
        Returns:
            List of recent papers from the specified category
        """
        try:
            client = arxiv.Client()
            
            # Calculate date range
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            search = arxiv.Search(
                query=f"cat:{category}",
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            papers = []
            for result in client.results(search):
                # Filter by date
                if result.published.replace(tzinfo=None) >= start_date:
                    paper = ArxivPaper(
                        title=result.title,
                        authors=[author.name for author in result.authors],
                        summary=result.summary,
                        published=result.published.isoformat(),
                        updated=result.updated.isoformat(),
                        entry_id=result.entry_id,
                        pdf_url=result.pdf_url,
                        categories=result.categories
                    )
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            raise Exception(f"Error fetching recent papers: {str(e)}")

async def run(self, transport: str = "stdio"):
    """Run the MCP server"""
    await self.mcp.run(transport)
```

# langgraph_client.py

import asyncio
import json
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
import subprocess
import os
from dotenv import load_dotenv

load_dotenv()

class ResearchState(BaseModel):
messages: List[BaseMessage] = []
research_query: str = “”
papers_found: List[Dict] = []
analysis_complete: bool = False

class ArxivResearchAgent:
def **init**(self, model_name: str = “gpt-4”):
self.llm = ChatOpenAI(
model=model_name,
temperature=0.1,
openai_api_key=os.getenv(“OPENAI_API_KEY”)
)
self.arxiv_tools = self._create_arxiv_tools()
self.tool_executor = ToolExecutor(self.arxiv_tools)
self.graph = self._create_graph()

```
def _create_arxiv_tools(self) -> List[Tool]:
    """Create tools that interface with the ArXiv MCP server"""
    
    async def search_papers(query: str, max_results: int = 5) -> str:
        """Search for papers on ArXiv"""
        try:
            # In a real implementation, this would call the MCP server
            # For demonstration, we'll simulate the call
            cmd = [
                "python", "-c", f"""
```

import asyncio
import json
from arxiv_server import ArxivMCPServer, ArxivSearchParams

async def search():
server = ArxivMCPServer()
params = ArxivSearchParams(query=”{query}”, max_results={max_results})
results = await server.mcp.tools[0].func(params)
print(json.dumps([paper.dict() for paper in results], indent=2))

asyncio.run(search())
“””
]

```
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            if result.returncode == 0:
                return result.stdout
            else:
                return f"Error: {result.stderr}"
                
        except Exception as e:
            return f"Error searching papers: {str(e)}"
    
    async def get_paper_by_id(entry_id: str) -> str:
        """Get detailed information about a specific paper"""
        try:
            cmd = [
                "python", "-c", f"""
```

import asyncio
import json
from arxiv_server import ArxivMCPServer

async def get_paper():
server = ArxivMCPServer()
result = await server.mcp.tools[1].func(”{entry_id}”)
print(json.dumps(result.dict(), indent=2))

asyncio.run(get_paper())
“””
]

```
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            if result.returncode == 0:
                return result.stdout
            else:
                return f"Error: {result.stderr}"
                
        except Exception as e:
            return f"Error getting paper details: {str(e)}"
    
    async def get_papers_by_author(author_name: str, max_results: int = 10) -> str:
        """Get papers by a specific author"""
        try:
            cmd = [
                "python", "-c", f"""
```

import asyncio
import json
from arxiv_server import ArxivMCPServer

async def get_author_papers():
server = ArxivMCPServer()
results = await server.mcp.tools[2].func(”{author_name}”, {max_results})
print(json.dumps([paper.dict() for paper in results], indent=2))

asyncio.run(get_author_papers())
“””
]

```
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            if result.returncode == 0:
                return result.stdout
            else:
                return f"Error: {result.stderr}"
                
        except Exception as e:
            return f"Error getting author papers: {str(e)}"
    
    return [
        Tool(
            name="search_arxiv_papers",
            description="Search for academic papers on ArXiv by query",
            func=search_papers
        ),
        Tool(
            name="get_paper_details",
            description="Get detailed information about a specific ArXiv paper by ID",
            func=get_paper_by_id
        ),
        Tool(
            name="get_papers_by_author", 
            description="Get papers by a specific author from ArXiv",
            func=get_papers_by_author
        )
    ]

def _create_graph(self) -> StateGraph:
    """Create the LangGraph workflow"""
    
    def researcher_node(state: ResearchState) -> Dict[str, Any]:
        """Main research node that processes queries and calls tools"""
        messages = state.messages
        last_message = messages[-1] if messages else None
        
        if isinstance(last_message, HumanMessage):
            # Extract research query
            research_query = last_message.content
            
            # Generate tool calls based on the query
            response = self.llm.bind_tools(self.arxiv_tools).invoke(messages)
            
            return {
                "messages": messages + [response],
                "research_query": research_query
            }
        
        return {"messages": messages}
    
    def tool_node(state: ResearchState) -> Dict[str, Any]:
        """Execute tools and return results"""
        messages = state.messages
        last_message = messages[-1]
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            tool_messages = []
            papers_found = []
            
            for tool_call in last_message.tool_calls:
                try:
                    # Execute the tool
                    result = asyncio.run(
                        self.tool_executor.ainvoke({
                            "tool": tool_call["name"],
                            "tool_input": tool_call["args"]
                        })
                    )
                    
                    # Parse result if it's JSON
                    try:
                        parsed_result = json.loads(result) if isinstance(result, str) else result
                        if isinstance(parsed_result, list):
                            papers_found.extend(parsed_result)
                        elif isinstance(parsed_result, dict):
                            papers_found.append(parsed_result)
                    except:
                        pass
                    
                    tool_message = ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"]
                    )
                    tool_messages.append(tool_message)
                    
                except Exception as e:
                    error_message = ToolMessage(
                        content=f"Error executing {tool_call['name']}: {str(e)}",
                        tool_call_id=tool_call["id"]
                    )
                    tool_messages.append(error_message)
            
            return {
                "messages": messages + tool_messages,
                "papers_found": state.papers_found + papers_found
            }
        
        return {"messages": messages}
    
    def analyzer_node(state: ResearchState) -> Dict[str, Any]:
        """Analyze the research results and provide insights"""
        messages = state.messages
        papers = state.papers_found
        
        if papers:
            analysis_prompt = f"""
            Based on the research conducted, please provide a comprehensive analysis of the {len(papers)} papers found.
            
            Please include:
            1. Key themes and trends across the papers
            2. Notable authors and institutions
            3. Recent developments in the field
            4. Potential research gaps or opportunities
            5. Recommendations for further reading
            
            Papers analyzed: {len(papers)} total
            """
            
            analysis_message = HumanMessage(content=analysis_prompt)
            response = self.llm.invoke(messages + [analysis_message])
            
            return {
                "messages": messages + [analysis_message, response],
                "analysis_complete": True
            }
        
        return {
            "messages": messages + [AIMessage(content="No papers were found to analyze.")],
            "analysis_complete": True
        }
    
    def should_continue(state: ResearchState) -> str:
        """Determine whether to continue with tools or move to analysis"""
        last_message = state.messages[-1] if state.messages else None
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        elif state.papers_found and not state.analysis_complete:
            return "analyze"
        else:
            return END
    
    # Build the graph
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("analyzer", analyzer_node)
    
    # Add edges
    workflow.set_entry_point("researcher")
    workflow.add_conditional_edges(
        "researcher",
        should_continue,
        {
            "tools": "tools",
            "analyze": "analyzer",
            END: END
        }
    )
    workflow.add_edge("tools", "researcher")
    workflow.add_edge("analyzer", END)
    
    return workflow.compile()

async def research(self, query: str) -> Dict[str, Any]:
    """Conduct research using the LangGraph workflow"""
    initial_state = ResearchState(
        messages=[HumanMessage(content=query)],
        research_query=query
    )
    
    result = await self.graph.ainvoke(initial_state)
    return result
```

# mcp_client.py

import asyncio
import json
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()

class ArxivMCPClient:
def **init**(self):
self.llm = ChatOpenAI(
model=“gpt-4”,
temperature=0.1,
openai_api_key=os.getenv(“OPENAI_API_KEY”)
)
self.agent = ArxivResearchAgent()

```
async def query_papers(self, user_query: str) -> str:
    """Query papers using natural language"""
    try:
        # Use LangGraph agent to process the query
        result = await self.agent.research(user_query)
        
        # Extract the final response
        final_message = result.messages[-1]
        if hasattr(final_message, 'content'):
            return final_message.content
        else:
            return str(final_message)
            
    except Exception as e:
        return f"Error processing query: {str(e)}"

async def interactive_session(self):
    """Run an interactive research session"""
    print("🔬 ArXiv Research Assistant")
    print("Ask me anything about academic papers!")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            user_input = input("\n📝 Your research question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("\n🔍 Researching...")
            response = await self.query_papers(user_input)
            print(f"\n📊 Results:\n{response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
```

# server_runner.py

async def run_arxiv_server():
“”“Run the ArXiv MCP server”””
server = ArxivMCPServer()
print(“🚀 Starting ArXiv MCP Server…”)
await server.run()

# example_usage.py

async def example_research_session():
“”“Example of how to use the ArXiv research system”””

```
print("🔬 ArXiv Research Example Session")
print("=" * 50)

# Initialize the client
client = ArxivMCPClient()

# Example queries
research_queries = [
    "Find recent papers about transformer architectures in machine learning",
    "What are the latest developments in quantum computing error correction?",
    "Show me papers by Geoffrey Hinton on deep learning",
    "Get recent papers in computer vision category from the last 5 days"
]

for i, query in enumerate(research_queries, 1):
    print(f"\n📝 Query {i}: {query}")
    print("-" * 50)
    
    try:
        response = await client.query_papers(query)
        print(f"📊 Results:\n{response}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print("\n" + "=" * 50)
```

# direct_mcp_example.py

async def direct_mcp_example():
“”“Example of directly using the MCP server tools”””

```
print("🔬 Direct MCP Server Example")
print("=" * 30)

# Initialize server
server = ArxivMCPServer()

# Example 1: Search papers
print("\n📝 Searching for 'machine learning' papers...")
search_params = ArxivSearchParams(
    query="machine learning",
    max_results=3,
    sort_by="relevance"
)

try:
    papers = await server.mcp.tools[0].func(search_params)
    print(f"Found {len(papers)} papers:")
    for paper in papers:
        print(f"- {paper.title}")
        print(f"  Authors: {', '.join(paper.authors[:3])}...")
        print(f"  Published: {paper.published}")
        print()
except Exception as e:
    print(f"Error: {e}")

# Example 2: Get papers by author
print("\n📝 Getting papers by 'Yoshua Bengio'...")
try:
    papers = await server.mcp.tools[2].func("Yoshua Bengio", 3)
    print(f"Found {len(papers)} papers:")
    for paper in papers:
        print(f"- {paper.title}")
        print(f"  Published: {paper.published}")
        print()
except Exception as e:
    print(f"Error: {e}")
```

# main.py

import asyncio
import sys
from arxiv_server import ArxivMCPServer
from mcp_client import ArxivMCPClient

async def main():
“”“Main entry point”””
if len(sys.argv) > 1:
mode = sys.argv[1]
else:
mode = “interactive”

```
if mode == "server":
    # Run the MCP server
    server = ArxivMCPServer()
    await server.run()
    
elif mode == "example":
    # Run example session
    await example_research_session()
    
elif mode == "direct":
    # Run direct MCP example
    await direct_mcp_example()
    
else:
    # Run interactive client
    client = ArxivMCPClient()
    await client.interactive_session()
```

if **name** == “**main**”:
asyncio.run(main())

# config.py

“”“Configuration settings for the ArXiv MCP system”””

# ArXiv API settings

ARXIV_BASE_URL = “http://export.arxiv.org/api/query”
DEFAULT_MAX_RESULTS = 10
DEFAULT_SORT_BY = “relevance”

# MCP Server settings

MCP_SERVER_NAME = “arxiv-research-server”
MCP_SERVER_VERSION = “1.0.0”

# LangGraph settings

DEFAULT_MODEL = “gpt-4”
RESEARCH_TEMPERATURE = 0.1

# Categories mapping for ArXiv

ARXIV_CATEGORIES = {
“cs.AI”: “Artificial Intelligence”,
“cs.LG”: “Machine Learning”,
“cs.CV”: “Computer Vision”,
“cs.CL”: “Computation and Language”,
“cs.RO”: “Robotics”,
“physics.quant-ph”: “Quantum Physics”,
“math.CO”: “Combinatorics”,
“stat.ML”: “Machine Learning (Statistics)”
}

# docker-compose.yml (for easy deployment)

“””
version: ‘3.8’

services:
arxiv-mcp-server:
build: .
environment:
- OPENAI_API_KEY=${OPENAI_API_KEY}
ports:
- “8000:8000”
volumes:
- .:/app
working_dir: /app
command: python main.py server

arxiv-client:
build: .
environment:
- OPENAI_API_KEY=${OPENAI_API_KEY}
volumes:
- .:/app
working_dir: /app
command: python main.py interactive
stdin_open: true
tty: true
depends_on:
- arxiv-mcp-server
“””

# Dockerfile

“””
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies

RUN apt-get update && apt-get install -y   
gcc   
&& rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies

COPY requirements.txt .
RUN pip install –no-cache-dir -r requirements.txt

# Copy application code

COPY . .

# Expose port for MCP server

EXPOSE 8000

# Default command

CMD [“python”, “main.py”, “server”]
“””

# .env.example

“””

# OpenAI API Key (required for LLM functionality)

OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Key (alternative to OpenAI)

ANTHROPIC_API_KEY=your_anthropic_api_key_here

# MCP Server Configuration

MCP_SERVER_PORT=8000
MCP_SERVER_HOST=localhost

# Logging Level

LOG_LEVEL=INFO
“””

# Usage instructions and examples

“””
USAGE INSTRUCTIONS:

1. Setup:
- Copy .env.example to .env and fill in your API keys
- Install dependencies: pip install -r requirements.txt
1. Running the system:
   
   a) Start MCP Server:
   python main.py server
   
   b) Run interactive research session:
   python main.py interactive
   
   c) Run example queries:
   python main.py example
   
   d) Test direct MCP functionality:
   python main.py direct
1. Example Research Queries:
- “Find papers about large language models published in the last month”
- “What are Andrej Karpathy’s recent papers?”
- “Show me quantum computing papers with high citation counts”
- “Find papers that cite ‘Attention is All You Need’”
- “Get recent computer vision papers about object detection”
1. Using with Docker:
   docker-compose up
1. Integration with other LLMs:
   The system can be easily adapted to work with other LLM providers
   by changing the model initialization in langgraph_client.py

FEATURES:

- FastMCP server with ArXiv integration
- LangGraph workflow for intelligent research
- Multiple search methods (query, author, category, ID)
- Real-time paper fetching and analysis
- Interactive CLI interface
- Docker deployment ready
- Extensible architecture for additional data sources
  “””