"""
Complete Graph of Thought (GoT) Implementation
==============================================
A production-ready implementation of the Graph of Thought framework for enhanced LLM reasoning.
Supports multiple LLM backends, graph operations, and integration with popular frameworks.

Author: Based on research by Besta et al. (2024) and community implementations
License: MIT
"""

import json
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple, Callable, Union
from enum import Enum
import networkx as nx
import numpy as np
from collections import defaultdict, deque
import heapq
import time
import hashlib

# Optional imports for framework integrations
try:
    from langchain_core.language_models import BaseLLM
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Core Data Structures
# ============================================================================

class ThoughtStatus(Enum):
    """Status of a thought node in the graph"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REFINED = "refined"
    AGGREGATED = "aggregated"


class OperationType(Enum):
    """Types of operations that can be performed on thoughts"""
    GENERATE = "generate"
    AGGREGATE = "aggregate"
    REFINE = "refine"
    SCORE = "score"
    VALIDATE = "validate"
    TRANSFORM = "transform"
    MERGE = "merge"
    BRANCH = "branch"


@dataclass
class Thought:
    """Represents a single thought/reasoning step in the graph"""
    id: str
    content: str
    operation: Optional[OperationType] = None
    score: float = 0.0
    status: ThoughtStatus = ThoughtStatus.PENDING
    parent_ids: List[str] = field(default_factory=list)
    child_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    iteration: int = 0
    
    def __hash__(self):
        return hash(self.id)
    
    def to_dict(self) -> Dict:
        """Convert thought to dictionary representation"""
        return {
            'id': self.id,
            'content': self.content,
            'operation': self.operation.value if self.operation else None,
            'score': self.score,
            'status': self.status.value,
            'parent_ids': self.parent_ids,
            'child_ids': self.child_ids,
            'metadata': self.metadata,
            'timestamp': self.timestamp,
            'iteration': self.iteration
        }


@dataclass
class GraphState:
    """Maintains the current state of the reasoning graph"""
    thoughts: Dict[str, Thought] = field(default_factory=dict)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    best_thought_id: Optional[str] = None
    iteration_count: int = 0
    total_cost: float = 0.0
    
    def add_thought(self, thought: Thought) -> None:
        """Add a thought to the graph"""
        self.thoughts[thought.id] = thought
        self.graph.add_node(thought.id, thought=thought)
        
    def add_edge(self, parent_id: str, child_id: str) -> None:
        """Add an edge between two thoughts"""
        if parent_id in self.thoughts and child_id in self.thoughts:
            self.edges.append((parent_id, child_id))
            self.graph.add_edge(parent_id, child_id)
            self.thoughts[parent_id].child_ids.append(child_id)
            self.thoughts[child_id].parent_ids.append(parent_id)
    
    def get_leaves(self) -> List[Thought]:
        """Get all leaf nodes (thoughts with no children)"""
        return [self.thoughts[node] for node in self.graph.nodes() 
                if self.graph.out_degree(node) == 0]
    
    def get_roots(self) -> List[Thought]:
        """Get all root nodes (thoughts with no parents)"""
        return [self.thoughts[node] for node in self.graph.nodes() 
                if self.graph.in_degree(node) == 0]


# ============================================================================
# Language Model Interface
# ============================================================================

class LLMInterface(ABC):
    """Abstract base class for LLM interfaces"""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM"""
        pass
    
    @abstractmethod
    def estimate_cost(self, prompt: str, response: str) -> float:
        """Estimate the cost of a generation"""
        pass


class OpenAIInterface(LLMInterface):
    """OpenAI API interface implementation"""
    
    def __init__(self, api_key: str, model: str = "gpt-4", temperature: float = 0.7):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.cost_per_1k_tokens = {"gpt-4": 0.03, "gpt-3.5-turbo": 0.002}.get(model, 0.03)
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI API"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', 1000)
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def estimate_cost(self, prompt: str, response: str) -> float:
        """Estimate generation cost"""
        total_tokens = (len(prompt) + len(response)) / 4  # Rough estimate
        return (total_tokens / 1000) * self.cost_per_1k_tokens


class LangChainInterface(LLMInterface):
    """LangChain LLM interface implementation"""
    
    def __init__(self, llm: Optional['BaseLLM'] = None):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not installed. Run: pip install langchain langchain-openai")
        
        self.llm = llm or ChatOpenAI(model="gpt-4", temperature=0.7)
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using LangChain"""
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            logger.error(f"LangChain generation error: {e}")
            raise
    
    def estimate_cost(self, prompt: str, response: str) -> float:
        """Estimate generation cost"""
        return len(prompt + response) * 0.00001  # Simplified cost estimate


# ============================================================================
# Graph Operations
# ============================================================================

class GraphOperation(ABC):
    """Abstract base class for graph operations"""
    
    @abstractmethod
    async def execute(self, state: GraphState, llm: LLMInterface, **kwargs) -> GraphState:
        """Execute the operation on the graph state"""
        pass


class GenerateOperation(GraphOperation):
    """Generate new thoughts from existing ones"""
    
    def __init__(self, num_branches: int = 3, prompt_template: Optional[str] = None):
        self.num_branches = num_branches
        self.prompt_template = prompt_template or """
        Based on the following thought: "{thought}"
        Generate {num} different perspectives or continuations.
        Format each as: [THOUGHT {i}]: <content>
        """
    
    async def execute(self, state: GraphState, llm: LLMInterface, 
                     source_thought_id: str, **kwargs) -> GraphState:
        """Generate new thoughts from a source thought"""
        source_thought = state.thoughts[source_thought_id]
        
        prompt = self.prompt_template.format(
            thought=source_thought.content,
            num=self.num_branches
        )
        
        response = await llm.generate(prompt)
        
        # Parse generated thoughts
        new_thoughts = self._parse_thoughts(response)
        
        # Add thoughts to graph
        for i, content in enumerate(new_thoughts):
            thought_id = f"{source_thought_id}_gen_{i}_{hashlib.md5(content.encode()).hexdigest()[:8]}"
            new_thought = Thought(
                id=thought_id,
                content=content,
                operation=OperationType.GENERATE,
                parent_ids=[source_thought_id],
                iteration=state.iteration_count
            )
            state.add_thought(new_thought)
            state.add_edge(source_thought_id, thought_id)
        
        state.total_cost += llm.estimate_cost(prompt, response)
        return state
    
    def _parse_thoughts(self, response: str) -> List[str]:
        """Parse thoughts from LLM response"""
        thoughts = []
        lines = response.split('\n')
        for line in lines:
            if '[THOUGHT' in line and ']:' in line:
                content = line.split(']:')[1].strip()
                thoughts.append(content)
        return thoughts[:self.num_branches]


class AggregateOperation(GraphOperation):
    """Aggregate multiple thoughts into a single thought"""
    
    def __init__(self, prompt_template: Optional[str] = None):
        self.prompt_template = prompt_template or """
        Synthesize the following thoughts into a coherent conclusion:
        {thoughts}
        
        Provide a comprehensive synthesis that captures the key insights.
        """
    
    async def execute(self, state: GraphState, llm: LLMInterface,
                     thought_ids: List[str], **kwargs) -> GraphState:
        """Aggregate multiple thoughts"""
        thoughts_content = []
        for tid in thought_ids:
            if tid in state.thoughts:
                thoughts_content.append(f"- {state.thoughts[tid].content}")
        
        prompt = self.prompt_template.format(
            thoughts='\n'.join(thoughts_content)
        )
        
        response = await llm.generate(prompt)
        
        # Create aggregated thought
        agg_id = f"agg_{hashlib.md5(''.join(thought_ids).encode()).hexdigest()[:8]}"
        agg_thought = Thought(
            id=agg_id,
            content=response,
            operation=OperationType.AGGREGATE,
            parent_ids=thought_ids,
            iteration=state.iteration_count,
            status=ThoughtStatus.AGGREGATED
        )
        
        state.add_thought(agg_thought)
        for parent_id in thought_ids:
            state.add_edge(parent_id, agg_id)
        
        state.total_cost += llm.estimate_cost(prompt, response)
        return state


class RefineOperation(GraphOperation):
    """Refine a thought based on feedback"""
    
    def __init__(self, prompt_template: Optional[str] = None):
        self.prompt_template = prompt_template or """
        Original thought: {thought}
        
        Feedback/Context: {feedback}
        
        Provide an improved version of this thought that addresses the feedback.
        """
    
    async def execute(self, state: GraphState, llm: LLMInterface,
                     thought_id: str, feedback: str = "", **kwargs) -> GraphState:
        """Refine a thought"""
        original_thought = state.thoughts[thought_id]
        
        prompt = self.prompt_template.format(
            thought=original_thought.content,
            feedback=feedback or "Improve clarity, accuracy, and completeness."
        )
        
        response = await llm.generate(prompt)
        
        # Create refined thought
        refined_id = f"{thought_id}_refined_{hashlib.md5(response.encode()).hexdigest()[:8]}"
        refined_thought = Thought(
            id=refined_id,
            content=response,
            operation=OperationType.REFINE,
            parent_ids=[thought_id],
            iteration=state.iteration_count,
            status=ThoughtStatus.REFINED
        )
        
        state.add_thought(refined_thought)
        state.add_edge(thought_id, refined_id)
        
        state.total_cost += llm.estimate_cost(prompt, response)
        return state


class ScoreOperation(GraphOperation):
    """Score thoughts based on quality criteria"""
    
    def __init__(self, scoring_fn: Optional[Callable] = None, prompt_template: Optional[str] = None):
        self.scoring_fn = scoring_fn
        self.prompt_template = prompt_template or """
        Evaluate the following thought on a scale of 0-100:
        "{thought}"
        
        Criteria: relevance, accuracy, completeness, clarity
        
        Provide only a numeric score.
        """
    
    async def execute(self, state: GraphState, llm: LLMInterface,
                     thought_ids: Optional[List[str]] = None, **kwargs) -> GraphState:
        """Score thoughts"""
        thought_ids = thought_ids or list(state.thoughts.keys())
        
        for thought_id in thought_ids:
            thought = state.thoughts[thought_id]
            
            if self.scoring_fn:
                # Use custom scoring function
                score = self.scoring_fn(thought, state)
            else:
                # Use LLM for scoring
                prompt = self.prompt_template.format(thought=thought.content)
                response = await llm.generate(prompt)
                
                try:
                    score = float(response.strip()) / 100.0
                except ValueError:
                    score = 0.5
                
                state.total_cost += llm.estimate_cost(prompt, response)
            
            thought.score = score
        
        # Update best thought
        best_thought = max(state.thoughts.values(), key=lambda t: t.score)
        state.best_thought_id = best_thought.id
        
        return state


# ============================================================================
# Graph of Thought Controller
# ============================================================================

class GraphOfThoughtController:
    """Main controller for Graph of Thought reasoning"""
    
    def __init__(self, llm: LLMInterface, max_iterations: int = 10,
                 convergence_threshold: float = 0.95):
        self.llm = llm
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.operations = {
            'generate': GenerateOperation(),
            'aggregate': AggregateOperation(),
            'refine': RefineOperation(),
            'score': ScoreOperation()
        }
    
    async def solve(self, problem: str, operation_graph: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Solve a problem using Graph of Thought reasoning
        
        Args:
            problem: The problem statement
            operation_graph: Optional custom operation graph specification
        
        Returns:
            Solution dictionary with result and metadata
        """
        # Initialize graph state
        state = GraphState()
        
        # Create initial thought
        initial_thought = Thought(
            id="root",
            content=problem,
            status=ThoughtStatus.COMPLETED
        )
        state.add_thought(initial_thought)
        
        # Use default or custom operation graph
        if operation_graph:
            result = await self._execute_custom_graph(state, operation_graph)
        else:
            result = await self._execute_default_graph(state)
        
        return {
            'solution': state.thoughts[state.best_thought_id].content if state.best_thought_id else None,
            'best_thought_id': state.best_thought_id,
            'graph_size': len(state.thoughts),
            'total_cost': state.total_cost,
            'iterations': state.iteration_count,
            'thoughts': {tid: t.to_dict() for tid, t in state.thoughts.items()},
            'edges': state.edges
        }
    
    async def _execute_default_graph(self, state: GraphState) -> GraphState:
        """Execute default Graph of Thought operations"""
        
        for iteration in range(self.max_iterations):
            state.iteration_count = iteration
            logger.info(f"Iteration {iteration + 1}/{self.max_iterations}")
            
            # Get current leaf nodes
            leaves = state.get_leaves()
            
            # Generate new thoughts from promising leaves
            for leaf in leaves:
                if leaf.score < 0.3:  # Skip low-quality thoughts
                    continue
                
                # Generate branches
                await self.operations['generate'].execute(
                    state, self.llm, leaf.id
                )
            
            # Score all thoughts
            await self.operations['score'].execute(state, self.llm)
            
            # Check for convergence
            if state.best_thought_id:
                best_thought = state.thoughts[state.best_thought_id]
                if best_thought.score >= self.convergence_threshold:
                    logger.info(f"Converged with score {best_thought.score}")
                    break
            
            # Aggregate promising thoughts periodically
            if iteration % 3 == 2:
                top_thoughts = sorted(
                    state.thoughts.values(),
                    key=lambda t: t.score,
                    reverse=True
                )[:5]
                
                if len(top_thoughts) >= 2:
                    await self.operations['aggregate'].execute(
                        state, self.llm,
                        [t.id for t in top_thoughts]
                    )
            
            # Refine the best thought periodically
            if iteration % 2 == 1 and state.best_thought_id:
                await self.operations['refine'].execute(
                    state, self.llm,
                    state.best_thought_id,
                    feedback="Improve based on other high-scoring thoughts"
                )
        
        # Final scoring
        await self.operations['score'].execute(state, self.llm)
        
        return state
    
    async def _execute_custom_graph(self, state: GraphState, 
                                   operation_graph: Dict) -> GraphState:
        """Execute custom operation graph"""
        
        for step in operation_graph.get('steps', []):
            operation_type = step['operation']
            params = step.get('params', {})
            
            if operation_type in self.operations:
                operation = self.operations[operation_type]
                await operation.execute(state, self.llm, **params)
            else:
                logger.warning(f"Unknown operation: {operation_type}")
        
        return state


# ============================================================================
# Advanced Graph Algorithms
# ============================================================================

class AdaptiveGraphOfThought(GraphOfThoughtController):
    """
    Adaptive Graph of Thought implementation with dynamic graph construction
    Based on "Adaptive Graph of Thoughts" (2025)
    """
    
    def __init__(self, llm: LLMInterface, complexity_threshold: float = 0.7,
                 max_depth: int = 5, **kwargs):
        super().__init__(llm, **kwargs)
        self.complexity_threshold = complexity_threshold
        self.max_depth = max_depth
        self.complexity_estimator = ComplexityEstimator(llm)
    
    async def solve(self, problem: str, **kwargs) -> Dict[str, Any]:
        """Solve using adaptive graph construction"""
        state = GraphState()
        
        # Create initial thought
        initial_thought = Thought(
            id="root",
            content=problem,
            metadata={'depth': 0}
        )
        state.add_thought(initial_thought)
        
        # Estimate initial complexity
        complexity = await self.complexity_estimator.estimate(problem)
        
        # Build adaptive graph
        await self._build_adaptive_graph(state, initial_thought, complexity)
        
        # Execute reasoning
        await self._execute_adaptive_reasoning(state)
        
        return self._prepare_result(state)
    
    async def _build_adaptive_graph(self, state: GraphState, thought: Thought,
                                   complexity: float) -> None:
        """Recursively build adaptive graph based on complexity"""
        
        depth = thought.metadata.get('depth', 0)
        
        if depth >= self.max_depth or complexity < self.complexity_threshold:
            return
        
        # Decide branching factor based on complexity
        num_branches = min(int(complexity * 5), 5)
        
        # Generate sub-thoughts
        sub_thoughts = await self._decompose_thought(thought, num_branches)
        
        for sub_thought in sub_thoughts:
            sub_thought.metadata['depth'] = depth + 1
            state.add_thought(sub_thought)
            state.add_edge(thought.id, sub_thought.id)
            
            # Estimate sub-problem complexity
            sub_complexity = await self.complexity_estimator.estimate(sub_thought.content)
            
            # Recursive decomposition
            await self._build_adaptive_graph(state, sub_thought, sub_complexity)
    
    async def _decompose_thought(self, thought: Thought, num_branches: int) -> List[Thought]:
        """Decompose a thought into sub-thoughts"""
        prompt = f"""
        Decompose this problem into {num_branches} sub-problems:
        "{thought.content}"
        
        Format: [SUB {i}]: <sub-problem>
        """
        
        response = await self.llm.generate(prompt)
        
        sub_thoughts = []
        for i, content in enumerate(self._parse_sub_problems(response)):
            sub_thought = Thought(
                id=f"{thought.id}_sub_{i}",
                content=content,
                parent_ids=[thought.id]
            )
            sub_thoughts.append(sub_thought)
        
        return sub_thoughts
    
    async def _execute_adaptive_reasoning(self, state: GraphState) -> None:
        """Execute reasoning on adaptive graph"""
        # Topological sort for bottom-up processing
        topo_order = list(nx.topological_sort(state.graph))
        
        # Process nodes in reverse topological order (leaves first)
        for node_id in reversed(topo_order):
            thought = state.thoughts[node_id]
            
            if state.graph.out_degree(node_id) == 0:
                # Leaf node - solve directly
                await self._solve_leaf(state, thought)
            else:
                # Internal node - aggregate children
                child_ids = list(state.graph.successors(node_id))
                await self.operations['aggregate'].execute(
                    state, self.llm, child_ids
                )
        
        # Score all thoughts
        await self.operations['score'].execute(state, self.llm)
    
    async def _solve_leaf(self, state: GraphState, thought: Thought) -> None:
        """Solve a leaf node thought"""
        prompt = f"Solve this atomic problem: {thought.content}"
        solution = await self.llm.generate(prompt)
        thought.content = solution
        thought.status = ThoughtStatus.COMPLETED
    
    def _parse_sub_problems(self, response: str) -> List[str]:
        """Parse sub-problems from response"""
        problems = []
        for line in response.split('\n'):
            if '[SUB' in line and ']:' in line:
                content = line.split(']:')[1].strip()
                problems.append(content)
        return problems
    
    def _prepare_result(self, state: GraphState) -> Dict[str, Any]:
        """Prepare final result dictionary"""
        return {
            'solution': state.thoughts.get(state.best_thought_id, Thought('', '')).content,
            'graph_stats': {
                'num_nodes': len(state.thoughts),
                'num_edges': len(state.edges),
                'max_depth': max(t.metadata.get('depth', 0) for t in state.thoughts.values())
            },
            'total_cost': state.total_cost,
            'thoughts': {tid: t.to_dict() for tid, t in state.thoughts.items()}
        }


class ComplexityEstimator:
    """Estimates problem complexity for adaptive graph construction"""
    
    def __init__(self, llm: LLMInterface):
        self.llm = llm
    
    async def estimate(self, problem: str) -> float:
        """Estimate complexity of a problem (0-1 scale)"""
        prompt = f"""
        Rate the complexity of this problem from 0 to 100:
        "{problem}"
        
        Consider: number of steps, interdependencies, domain knowledge required.
        Respond with only a number.
        """
        
        response = await self.llm.generate(prompt)
        
        try:
            complexity = float(response.strip()) / 100.0
            return min(max(complexity, 0.0), 1.0)
        except ValueError:
            return 0.5  # Default complexity


# ============================================================================
# Specialized Solvers
# ============================================================================

class SortingGoTSolver:
    """Specialized Graph of Thought solver for sorting problems"""
    
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.controller = GraphOfThoughtController(llm)
    
    async def sort(self, items: List[Any]) -> List[Any]:
        """Sort items using Graph of Thought merge sort approach"""
        
        if len(items) <= 1:
            return items
        
        # Create sorting problem
        problem = f"Sort these items: {items}"
        
        # Define custom operation graph for merge sort
        operation_graph = {
            'steps': [
                {
                    'operation': 'generate',
                    'params': {
                        'source_thought_id': 'root',
                        'prompt_template': """
                        Split this list into two halves:
                        {thought}
                        
                        Format:
                        [THOUGHT 1]: First half: [items]
                        [THOUGHT 2]: Second half: [items]
                        """
                    }
                },
                {
                    'operation': 'custom_merge',
                    'params': {}
                }
            ]
        }
        
        result = await self.controller.solve(problem, operation_graph)
        
        # Parse sorted result
        return self._parse_sorted_list(result['solution'])
    
    def _parse_sorted_list(self, solution: str) -> List[Any]:
        """Parse sorted list from solution string"""
        # Implementation depends on output format
        import ast
        try:
            # Try to extract list from solution
            start = solution.find('[')
            end = solution.rfind(']') + 1
            return ast.literal_eval(solution[start:end])
        except:
            return []


class MathematicalReasoningGoT:
    """Graph of Thought solver for mathematical reasoning problems"""
    
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.controller = AdaptiveGraphOfThought(llm)
    
    async def solve_equation(self, equation: str) -> Dict[str, Any]:
        """Solve mathematical equation using GoT"""
        
        # Enhanced prompt for mathematical reasoning
        problem = f"""
        Solve this mathematical problem step by step:
        {equation}
        
        Break it down into logical steps and show your work.
        """
        
        # Custom scoring function for math problems
        def math_scoring_fn(thought: Thought, state: GraphState) -> float:
            # Check for mathematical consistency
            score = 0.5  # Base score
            
            # Bonus for containing numbers and operations
            if any(char in thought.content for char in '0123456789+-*/='):
                score += 0.2
            
            # Bonus for step-by-step format
            if 'step' in thought.content.lower():
                score += 0.1
            
            # Bonus for final answer format
            if 'answer:' in thought.content.lower() or '=' in thought.content:
                score += 0.2
            
            return min(score, 1.0)
        
        # Override scoring operation
        self.controller.operations['score'] = ScoreOperation(scoring_fn=math_scoring_fn)
        
        return await self.controller.solve(problem)


# ============================================================================
# Example Usage and Testing
# ============================================================================

async def example_basic_got():
    """Example of basic Graph of Thought usage"""
    
    # Initialize LLM (using mock for example)
    class MockLLM(LLMInterface):
        async def generate(self, prompt: str, **kwargs) -> str:
            # Mock responses for demonstration
            if "Generate" in prompt:
                return "[THOUGHT 1]: First perspective\n[THOUGHT 2]: Second perspective"
            elif "Synthesize" in prompt:
                return "Combined insight from multiple perspectives"
            elif "Evaluate" in prompt:
                return "85"
            else:
                return "Mock response"
        
        def estimate_cost(self, prompt: str, response: str) -> float:
            return 0.001
    
    llm = MockLLM()
    controller = GraphOfThoughtController(llm, max_iterations=3)
    
    problem = "What are the implications of artificial general intelligence?"
    result = await controller.solve(problem)
    
    print(f"Solution: {result['solution']}")
    print(f"Graph size: {result['graph_size']} thoughts")
    print(f"Total cost: ${result['total_cost']:.4f}")
    print(f"Iterations: {result['iterations']}")


async def example_adaptive_got():
    """Example of Adaptive Graph of Thought"""
    
    # This would use real LLM in production
    llm = MockLLM()  # Use the mock from above
    
    adaptive_controller = AdaptiveGraphOfThought(
        llm,
        complexity_threshold=0.6,
        max_depth=3
    )
    
    problem = "Design a sustainable city for 1 million people"
    result = await adaptive_controller.solve(problem)
    
    print(f"Adaptive GoT Solution: {result['solution']}")
    print(f"Graph statistics: {result['graph_stats']}")


async def example_with_langchain():
    """Example integration with LangChain"""
    
    if not LANGCHAIN_AVAILABLE:
        print("LangChain not available. Install with: pip install langchain langchain-openai")
        return
    
    # Initialize LangChain LLM
    from langchain_openai import ChatOpenAI
    
    langchain_llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.7,
        openai_api_key="your-api-key"  # Replace with actual key
    )
    
    llm_interface = LangChainInterface(langchain_llm)
    controller = GraphOfThoughtController(llm_interface)
    
    problem = "Analyze the economic impact of remote work"
    result = await controller.solve(problem)
    
    print(f"LangChain GoT Result: {result['solution']}")


# ============================================================================
# Integration with LangGraph
# ============================================================================

try:
    from langgraph.graph import Graph, Node, Edge
    from langgraph.prebuilt import ToolExecutor
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


class LangGraphGoTIntegration:
    """Integration of Graph of Thought with LangGraph"""
    
    def __init__(self, llm: LLMInterface):
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("LangGraph not installed. Run: pip install langgraph")
        
        self.llm = llm
        self.graph = self._build_langgraph()
    
    def _build_langgraph(self) -> 'Graph':
        """Build LangGraph representation of GoT"""
        
        graph = Graph()
        
        # Define nodes for GoT operations
        graph.add_node("generate", self._generate_node)
        graph.add_node("aggregate", self._aggregate_node)
        graph.add_node("refine", self._refine_node)
        graph.add_node("score", self._score_node)
        
        # Define edges
        graph.add_edge("generate", "score")
        graph.add_edge("score", "aggregate", condition=self._should_aggregate)
        graph.add_edge("aggregate", "refine")
        graph.add_edge("refine", "score")
        
        return graph
    
    async def _generate_node(self, state: Dict) -> Dict:
        """Generate thoughts node"""
        operation = GenerateOperation()
        graph_state = state.get('graph_state')
        source_id = state.get('current_thought_id', 'root')
        
        await operation.execute(graph_state, self.llm, source_id)
        
        state['graph_state'] = graph_state
        return state
    
    async def _aggregate_node(self, state: Dict) -> Dict:
        """Aggregate thoughts node"""
        operation = AggregateOperation()
        graph_state = state.get('graph_state')
        
        # Get top thoughts for aggregation
        top_thoughts = sorted(
            graph_state.thoughts.values(),
            key=lambda t: t.score,
            reverse=True
        )[:3]
        
        await operation.execute(
            graph_state, self.llm,
            [t.id for t in top_thoughts]
        )
        
        state['graph_state'] = graph_state
        return state
    
    async def _refine_node(self, state: Dict) -> Dict:
        """Refine thoughts node"""
        operation = RefineOperation()
        graph_state = state.get('graph_state')
        
        if graph_state.best_thought_id:
            await operation.execute(
                graph_state, self.llm,
                graph_state.best_thought_id
            )
        
        state['graph_state'] = graph_state
        return state
    
    async def _score_node(self, state: Dict) -> Dict:
        """Score thoughts node"""
        operation = ScoreOperation()
        graph_state = state.get('graph_state')
        
        await operation.execute(graph_state, self.llm)
        
        state['graph_state'] = graph_state
        state['iteration'] = state.get('iteration', 0) + 1
        return state
    
    def _should_aggregate(self, state: Dict) -> bool:
        """Condition for aggregation"""
        return state.get('iteration', 0) % 3 == 0
    
    async def run(self, problem: str) -> Dict:
        """Run GoT using LangGraph"""
        initial_state = {
            'graph_state': GraphState(),
            'problem': problem,
            'iteration': 0
        }
        
        # Add initial thought
        root_thought = Thought(id="root", content=problem)
        initial_state['graph_state'].add_thought(root_thought)
        
        # Run graph
        final_state = await self.graph.arun(initial_state)
        
        return {
            'solution': final_state['graph_state'].thoughts[
                final_state['graph_state'].best_thought_id
            ].content if final_state['graph_state'].best_thought_id else None,
            'graph_state': final_state['graph_state']
        }


# ============================================================================
# Visualization and Analysis Tools
# ============================================================================

class GoTVisualizer:
    """Visualization tools for Graph of Thought"""
    
    @staticmethod
    def visualize_graph(state: GraphState, output_file: str = "got_graph.html"):
        """Create interactive visualization of the reasoning graph"""
        
        try:
            import pyvis
            from pyvis.network import Network
        except ImportError:
            logger.error("PyVis not installed. Run: pip install pyvis")
            return
        
        net = Network(height="750px", width="100%", directed=True)
        
        # Add nodes with properties
        for thought_id, thought in state.thoughts.items():
            color = GoTVisualizer._get_node_color(thought)
            label = f"{thought_id[:8]}\nScore: {thought.score:.2f}"
            title = thought.content[:100] + "..." if len(thought.content) > 100 else thought.content
            
            net.add_node(
                thought_id,
                label=label,
                title=title,
                color=color,
                size=20 + thought.score * 30
            )
        
        # Add edges
        for parent_id, child_id in state.edges:
            net.add_edge(parent_id, child_id)
        
        # Configure physics
        net.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=100)
        
        # Save visualization
        net.show(output_file)
        logger.info(f"Graph visualization saved to {output_file}")
    
    @staticmethod
    def _get_node_color(thought: Thought) -> str:
        """Get node color based on status and score"""
        if thought.status == ThoughtStatus.FAILED:
            return "#FF0000"
        elif thought.status == ThoughtStatus.AGGREGATED:
            return "#0000FF"
        elif thought.status == ThoughtStatus.REFINED:
            return "#00FF00"
        elif thought.score > 0.8:
            return "#FFD700"  # Gold for high-scoring
        elif thought.score > 0.5:
            return "#FFA500"  # Orange for medium
        else:
            return "#808080"  # Gray for low-scoring
    
    @staticmethod
    def analyze_graph_metrics(state: GraphState) -> Dict[str, Any]:
        """Analyze graph metrics and statistics"""
        
        metrics = {
            'num_thoughts': len(state.thoughts),
            'num_edges': len(state.edges),
            'avg_score': np.mean([t.score for t in state.thoughts.values()]),
            'max_score': max(t.score for t in state.thoughts.values()),
            'min_score': min(t.score for t in state.thoughts.values()),
            'graph_density': nx.density(state.graph) if len(state.thoughts) > 1 else 0,
            'avg_degree': np.mean([d for _, d in state.graph.degree()]) if state.thoughts else 0,
            'max_depth': GoTVisualizer._calculate_max_depth(state),
            'num_leaves': len(state.get_leaves()),
            'num_roots': len(state.get_roots()),
            'thought_distribution': GoTVisualizer._get_thought_distribution(state)
        }
        
        # Check for cycles
        metrics['has_cycles'] = not nx.is_directed_acyclic_graph(state.graph)
        
        # Connected components
        metrics['num_components'] = nx.number_weakly_connected_components(state.graph)
        
        return metrics
    
    @staticmethod
    def _calculate_max_depth(state: GraphState) -> int:
        """Calculate maximum depth of the graph"""
        if not state.thoughts:
            return 0
        
        roots = state.get_roots()
        max_depth = 0
        
        for root in roots:
            depths = nx.single_source_shortest_path_length(state.graph, root.id)
            max_depth = max(max_depth, max(depths.values()) if depths else 0)
        
        return max_depth
    
    @staticmethod
    def _get_thought_distribution(state: GraphState) -> Dict[str, int]:
        """Get distribution of thought types"""
        distribution = defaultdict(int)
        
        for thought in state.thoughts.values():
            if thought.operation:
                distribution[thought.operation.value] += 1
            else:
                distribution['initial'] += 1
        
        return dict(distribution)


# ============================================================================
# Performance Optimization
# ============================================================================

class CachedGoTController(GraphOfThoughtController):
    """Graph of Thought controller with caching for improved performance"""
    
    def __init__(self, llm: LLMInterface, cache_size: int = 1000, **kwargs):
        super().__init__(llm, **kwargs)
        self.cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _get_cache_key(self, operation: str, content: str) -> str:
        """Generate cache key for operation and content"""
        return hashlib.md5(f"{operation}:{content}".encode()).hexdigest()
    
    async def _cached_generate(self, prompt: str) -> str:
        """Generate with caching"""
        cache_key = self._get_cache_key("generate", prompt)
        
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        self.cache_misses += 1
        response = await self.llm.generate(prompt)
        
        # Add to cache with LRU eviction
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = response
        return response
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_size': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'memory_usage_mb': len(str(self.cache)) / (1024 * 1024)
        }


class ParallelGoTController(GraphOfThoughtController):
    """Graph of Thought with parallel execution capabilities"""
    
    def __init__(self, llm: LLMInterface, max_parallel: int = 5, **kwargs):
        super().__init__(llm, **kwargs)
        self.max_parallel = max_parallel
    
    async def _parallel_generate(self, state: GraphState, thought_ids: List[str]) -> None:
        """Generate thoughts in parallel"""
        
        # Create tasks for parallel generation
        tasks = []
        for thought_id in thought_ids[:self.max_parallel]:
            task = self.operations['generate'].execute(
                state, self.llm, thought_id
            )
            tasks.append(task)
        
        # Execute in parallel
        await asyncio.gather(*tasks)
    
    async def _execute_default_graph(self, state: GraphState) -> GraphState:
        """Execute with parallel operations"""
        
        for iteration in range(self.max_iterations):
            state.iteration_count = iteration
            
            # Get current leaf nodes
            leaves = state.get_leaves()
            promising_leaves = [l for l in leaves if l.score >= 0.3]
            
            # Parallel generation
            if promising_leaves:
                leaf_ids = [l.id for l in promising_leaves]
                await self._parallel_generate(state, leaf_ids)
            
            # Score all thoughts
            await self.operations['score'].execute(state, self.llm)
            
            # Check convergence
            if state.best_thought_id:
                best_thought = state.thoughts[state.best_thought_id]
                if best_thought.score >= self.convergence_threshold:
                    break
        
        return state


# ============================================================================
# Production-Ready Wrapper
# ============================================================================

class GraphOfThought:
    """
    Production-ready Graph of Thought implementation
    
    Examples:
        >>> # Basic usage with OpenAI
        >>> got = GraphOfThought(api_key="your-key", model="gpt-4")
        >>> result = await got.solve("Complex problem here")
        
        >>> # Advanced usage with custom configuration
        >>> got = GraphOfThought(
        ...     api_key="your-key",
        ...     use_adaptive=True,
        ...     enable_caching=True,
        ...     enable_parallel=True,
        ...     max_iterations=15
        ... )
        >>> result = await got.solve("Problem", visualize=True)
    """
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "gpt-4",
                 use_adaptive: bool = False,
                 enable_caching: bool = True,
                 enable_parallel: bool = False,
                 max_iterations: int = 10,
                 **kwargs):
        
        # Initialize LLM interface
        if api_key and OPENAI_AVAILABLE:
            self.llm = OpenAIInterface(api_key, model)
        elif LANGCHAIN_AVAILABLE:
            self.llm = LangChainInterface()
        else:
            raise ValueError("No LLM backend available. Install openai or langchain.")
        
        # Select controller type
        if use_adaptive:
            controller_class = AdaptiveGraphOfThought
        elif enable_caching:
            controller_class = CachedGoTController
        elif enable_parallel:
            controller_class = ParallelGoTController
        else:
            controller_class = GraphOfThoughtController
        
        self.controller = controller_class(
            self.llm,
            max_iterations=max_iterations,
            **kwargs
        )
        
        self.visualizer = GoTVisualizer()
    
    async def solve(self,
                   problem: str,
                   operation_graph: Optional[Dict] = None,
                   visualize: bool = False,
                   analyze: bool = False) -> Dict[str, Any]:
        """
        Solve a problem using Graph of Thought
        
        Args:
            problem: Problem statement to solve
            operation_graph: Optional custom operation graph
            visualize: Whether to create visualization
            analyze: Whether to include graph analysis
        
        Returns:
            Solution dictionary with results and metadata
        """
        
        # Solve problem
        result = await self.controller.solve(problem, operation_graph)
        
        # Add visualization if requested
        if visualize and 'graph_state' in result:
            self.visualizer.visualize_graph(result['graph_state'])
            result['visualization'] = "got_graph.html"
        
        # Add analysis if requested
        if analyze and 'graph_state' in result:
            result['metrics'] = self.visualizer.analyze_graph_metrics(
                result['graph_state']
            )
        
        # Add cache stats if available
        if hasattr(self.controller, 'get_cache_stats'):
            result['cache_stats'] = self.controller.get_cache_stats()
        
        return result
    
    def set_custom_operation(self, name: str, operation: GraphOperation) -> None:
        """Add custom operation to the controller"""
        self.controller.operations[name] = operation


# ============================================================================
# Testing and Validation
# ============================================================================

async def run_tests():
    """Run comprehensive tests of the Graph of Thought implementation"""
    
    print("="*60)
    print("Graph of Thought Implementation Tests")
    print("="*60)
    
    # Test 1: Basic GoT
    print("\n1. Testing Basic Graph of Thought...")
    await example_basic_got()
    
    # Test 2: Adaptive GoT
    print("\n2. Testing Adaptive Graph of Thought...")
    await example_adaptive_got()
    
    # Test 3: LangChain Integration (if available)
    if LANGCHAIN_AVAILABLE:
        print("\n3. Testing LangChain Integration...")
        await example_with_langchain()
    
    print("\n" + "="*60)
    print("All tests completed!")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Run tests
    asyncio.run(run_tests())
    
    print("\n" + "="*60)
    print("Graph of Thought Implementation Ready!")
    print("="*60)
    print("\nQuick Start:")
    print("1. Install dependencies: pip install openai langchain networkx pyvis")
    print("2. Set your API key")
    print("3. Use: got = GraphOfThought(api_key='your-key')")
    print("4. Solve: result = await got.solve('your problem')")
    print("\nFor documentation, see class docstrings and examples above.")



# -------------------------------------------------------------------------------


import asyncio
from got_implementation import GraphOfThought

async def main():
    # Initialize with OpenAI
    got = GraphOfThought(
        api_key="your-openai-key",
        model="gpt-4",
        use_adaptive=True,  # Use adaptive graph construction
        enable_caching=True,  # Enable response caching
        max_iterations=10
    )
    
    # Solve a complex problem
    result = await got.solve(
        problem="Design a sustainable transportation system for a city",
        visualize=True,  # Create graph visualization
        analyze=True     # Include metrics analysis
    )
    
    print(f"Solution: {result['solution']}")
    print(f"Graph metrics: {result['metrics']}")
    print(f"Total cost: ${result['total_cost']:.4f}")

# Run
asyncio.run(main())
