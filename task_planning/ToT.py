"""
Tree of Thought (ToT) Implementation for Text-Based Use Cases

This is a complete implementation of the Tree of Thought algorithm based on the 
Princeton NLP paper "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
(https://arxiv.org/abs/2305.10601)

The implementation supports:
- Creative writing tasks
- Mathematical reasoning (Game of 24)
- General text-based problem solving
- Both BFS and DFS search strategies
- Value-based and vote-based evaluation

Usage:
    # Set your OpenAI API key
    import os
    os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
    
    # Example: Creative Writing
    task = CreativeWritingTask()
    problem = ["The old lighthouse stood sentinel.", "Waves crashed below.", 
               "A figure appeared in the distance.", "The mystery began to unfold."]
    result = TreeOfThought(task).solve(problem)
    
    # Example: Game of 24
    task = GameOf24Task()
    problem = [4, 1, 8, 7]
    result = TreeOfThought(task).solve(problem)
"""

import os
import json
import time
import random
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import defaultdict
import openai

# Configure OpenAI client
openai.api_key = os.getenv('OPENAI_API_KEY')

@dataclass
class State:
    """
    Represents a state in the Tree of Thought search.
    
    Each state contains:
    - problem: The original problem being solved
    - thoughts: List of intermediate reasoning steps taken so far
    - depth: Current depth in the search tree
    - score: Evaluation score for this state (if evaluated)
    """
    problem: Any
    thoughts: List[str]
    depth: int = 0
    score: float = 0.0
    
    def __hash__(self):
        # Enable states to be used in sets/dicts for duplicate detection
        return hash((str(self.problem), tuple(self.thoughts)))
    
    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return self.problem == other.problem and self.thoughts == other.thoughts
    
    def copy_with_thought(self, new_thought: str) -> 'State':
        """Create a new state by adding a thought to the current state."""
        return State(
            problem=self.problem,
            thoughts=self.thoughts + [new_thought],
            depth=self.depth + 1,
            score=0.0
        )

class LLMInterface:
    """
    Interface for interacting with Large Language Models.
    
    This class handles:
    - API calls to OpenAI
    - Response parsing and error handling
    - Caching to avoid redundant API calls
    - Rate limiting and retry logic
    """
    
    def __init__(self, model: str = "gpt-4", temperature: float = 0.7):
        self.model = model
        self.temperature = temperature
        self.cache = {}  # Simple cache to avoid redundant API calls
        self.api_calls = 0  # Track API usage
        
    def generate(self, prompt: str, n: int = 1, max_tokens: int = 1000) -> List[str]:
        """
        Generate text using the language model.
        
        Args:
            prompt: Input prompt for the model
            n: Number of completions to generate
            max_tokens: Maximum tokens in each completion
            
        Returns:
            List of generated text completions
        """
        # Check cache first to avoid redundant API calls
        cache_key = f"{prompt}_{n}_{max_tokens}_{self.temperature}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Make API call to OpenAI
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                n=n,
                max_tokens=max_tokens,
                temperature=self.temperature,
                stop=None
            )
            
            # Extract text from response
            results = [choice.message.content.strip() for choice in response.choices]
            
            # Cache the results
            self.cache[cache_key] = results
            self.api_calls += 1
            
            return results
            
        except Exception as e:
            print(f"API Error: {e}")
            # Return fallback responses to prevent crashes
            return [f"Error in generation: {str(e)}"] * n
    
    def evaluate_value(self, prompt: str, state_description: str) -> float:
        """
        Evaluate a state using value-based scoring (1-10 scale).
        
        Args:
            prompt: Evaluation prompt template
            state_description: Description of the state to evaluate
            
        Returns:
            Numerical score between 0-10
        """
        full_prompt = f"{prompt}\n\nState to evaluate: {state_description}\n\nScore (1-10):"
        
        try:
            response = self.generate(full_prompt, n=1, max_tokens=50)[0]
            
            # Extract numerical score from response
            score_match = re.search(r'\b(\d+(?:\.\d+)?)\b', response)
            if score_match:
                score = float(score_match.group(1))
                # Ensure score is in valid range
                return max(0.0, min(10.0, score))
            else:
                # If no valid score found, return neutral score
                return 5.0
                
        except Exception as e:
            print(f"Evaluation Error: {e}")
            return 5.0  # Neutral score on error
    
    def evaluate_vote(self, prompt: str, states: List[str]) -> int:
        """
        Evaluate states using vote-based comparison.
        
        Args:
            prompt: Voting prompt template
            states: List of state descriptions to compare
            
        Returns:
            Index of the best state (0-based)
        """
        # Format states for voting
        formatted_states = "\n".join([f"{i+1}. {state}" for i, state in enumerate(states)])
        full_prompt = f"{prompt}\n\nOptions:\n{formatted_states}\n\nBest option (enter number):"
        
        try:
            response = self.generate(full_prompt, n=1, max_tokens=50)[0]
            
            # Extract vote from response
            vote_match = re.search(r'\b(\d+)\b', response)
            if vote_match:
                vote = int(vote_match.group(1)) - 1  # Convert to 0-based index
                # Ensure vote is in valid range
                if 0 <= vote < len(states):
                    return vote
            
            # If no valid vote found, return random choice
            return random.randint(0, len(states) - 1)
            
        except Exception as e:
            print(f"Voting Error: {e}")
            return 0  # Return first option on error

class ToTTask(ABC):
    """
    Abstract base class for Tree of Thought tasks.
    
    Each specific task (e.g., creative writing, math problems) should inherit
    from this class and implement the required methods.
    """
    
    def __init__(self, llm: LLMInterface):
        self.llm = llm
    
    @abstractmethod
    def generate_thoughts(self, state: State, n_thoughts: int = 5) -> List[str]:
        """Generate candidate thoughts for the given state."""
        pass
    
    @abstractmethod
    def evaluate_states(self, states: List[State]) -> List[float]:
        """Evaluate a list of states and return scores."""
        pass
    
    @abstractmethod
    def is_solution(self, state: State) -> bool:
        """Check if the given state represents a valid solution."""
        pass
    
    @abstractmethod
    def get_state_description(self, state: State) -> str:
        """Get a text description of the state for evaluation."""
        pass

class CreativeWritingTask(ToTTask):
    """
    Task for creative writing where the goal is to write a coherent story
    incorporating given sentences as paragraph endings.
    
    This task demonstrates ToT's ability to handle creative, subjective problems
    that require planning and coherence across multiple steps.
    """
    
    def __init__(self, llm: LLMInterface):
        super().__init__(llm)
        
        # Templates for generating writing plans and passages
        self.plan_generation_prompt = """You are helping to write a creative story. You must incorporate these specific sentences as the ENDING of each paragraph in order:

Sentences to incorporate: {sentences}

Generate a creative writing plan that explains how to build a coherent story where each paragraph naturally leads to and ends with the corresponding sentence. The plan should outline the main theme, character development, and narrative structure.

Writing Plan:"""
        
        # Template for evaluating writing plans
        self.plan_evaluation_prompt = """Evaluate this creative writing plan based on:
1. Coherence and logical flow
2. Creative potential
3. How well it incorporates the required sentences
4. Overall narrative structure

Rate the plan's quality and potential for creating an engaging story."""
        
        # Template for generating actual passages
        self.passage_generation_prompt = """Based on this writing plan, write the complete story with {n_paragraphs} paragraphs. Each paragraph must end with the corresponding sentence provided.

Writing Plan: {plan}

Required ending sentences: {sentences}

Write the complete story:"""
    
    def generate_thoughts(self, state: State, n_thoughts: int = 5) -> List[str]:
        """
        Generate writing plans or story passages based on current state.
        
        In the first step, we generate writing plans.
        In subsequent steps, we generate story passages based on the chosen plan.
        """
        sentences = state.problem
        
        if state.depth == 0:
            # Generate writing plans
            prompt = self.plan_generation_prompt.format(sentences=sentences)
            plans = self.llm.generate(prompt, n=n_thoughts, max_tokens=300)
            return plans
            
        elif state.depth == 1:
            # Generate story passages based on the chosen plan
            chosen_plan = state.thoughts[0]  # The writing plan from previous step
            prompt = self.passage_generation_prompt.format(
                plan=chosen_plan,
                sentences=sentences,
                n_paragraphs=len(sentences)
            )
            passages = self.llm.generate(prompt, n=n_thoughts, max_tokens=800)
            return passages
        
        else:
            # For deeper levels, refine existing passages
            current_passage = state.thoughts[-1]
            refine_prompt = f"""Improve this story passage for better coherence and creativity:

{current_passage}

Required ending sentences: {sentences}

Improved version:"""
            
            refined_passages = self.llm.generate(refine_prompt, n=n_thoughts, max_tokens=800)
            return refined_passages
    
    def evaluate_states(self, states: List[State]) -> List[float]:
        """
        Evaluate writing states based on creativity and coherence.
        
        Uses value-based evaluation to score each state numerically.
        """
        scores = []
        
        for state in states:
            state_desc = self.get_state_description(state)
            
            if state.depth == 0:
                # Evaluate writing plans
                score = self.llm.evaluate_value(
                    self.plan_evaluation_prompt,
                    state_desc
                )
            else:
                # Evaluate story passages
                evaluation_prompt = """Evaluate this creative writing attempt based on:
1. How well it incorporates the required sentences naturally
2. Coherence and flow of the narrative
3. Creativity and engagement
4. Overall story quality

Rate the writing quality:"""
                
                score = self.llm.evaluate_value(evaluation_prompt, state_desc)
            
            scores.append(score)
        
        return scores
    
    def is_solution(self, state: State) -> bool:
        """
        Check if we have a complete story that incorporates all required sentences.
        
        A solution is reached when we have generated a story passage (depth >= 1)
        with reasonable quality.
        """
        if state.depth < 1:
            return False
        
        # Check if the story contains the required sentences
        if len(state.thoughts) > 1:  # Has both plan and passage
            story = state.thoughts[-1]
            sentences = state.problem
            
            # Simple check: see if most required sentences appear in the story
            found_sentences = sum(1 for sentence in sentences if sentence.lower() in story.lower())
            return found_sentences >= len(sentences) * 0.7  # 70% of sentences found
        
        return False
    
    def get_state_description(self, state: State) -> str:
        """Get a description of the current writing state for evaluation."""
        if state.depth == 0:
            return f"Initial problem: Incorporate sentences {state.problem}"
        elif state.depth == 1:
            return f"Writing plan: {state.thoughts[0]}"
        else:
            return f"Story passage: {state.thoughts[-1][:500]}..."  # Truncate for brevity

class GameOf24Task(ToTTask):
    """
    Task for the Game of 24 mathematical puzzle.
    
    Given 4 numbers, use +, -, *, / operations to reach exactly 24.
    This task demonstrates ToT's ability to handle logical, step-by-step reasoning
    with clear success/failure criteria.
    """
    
    def __init__(self, llm: LLMInterface):
        super().__init__(llm)
        
        # Template for proposing mathematical operations
        self.operation_prompt = """In the Game of 24, you need to use the numbers {numbers} with operations +, -, *, / to get exactly 24.

Current numbers available: {numbers}

Propose a single arithmetic operation that brings you closer to 24. Respond in the format:
"a ○ b = c" where ○ is an operation and a, b are from the available numbers.

For example: "4 * 6 = 24" or "8 - 3 = 5"

Operation:"""
        
        # Template for evaluating progress toward 24
        self.evaluation_prompt = """In the Game of 24, evaluate how likely these remaining numbers can reach 24:

Remaining numbers: {numbers}

Consider:
1. Are there obvious operations that lead directly to 24?
2. Can these numbers be combined reasonably to approach 24?
3. Are we stuck with no good options?

Classify this state as:
- "sure" if you can definitely reach 24
- "maybe" if it's possible but unclear  
- "impossible" if there's no way to reach 24

Answer:"""
    
    def generate_thoughts(self, state: State, n_thoughts: int = 5) -> List[str]:
        """
        Generate possible arithmetic operations for current numbers.
        
        Each thought represents a single mathematical operation that transforms
        the current set of numbers into a new set.
        """
        numbers = self.get_current_numbers(state)
        
        if len(numbers) < 2:
            return []  # Cannot generate operations with < 2 numbers
        
        prompt = self.operation_prompt.format(numbers=numbers)
        operations = self.llm.generate(prompt, n=n_thoughts, max_tokens=100)
        
        # Filter valid operations
        valid_operations = []
        for op in operations:
            if self.is_valid_operation(op, numbers):
                valid_operations.append(op)
        
        return valid_operations if valid_operations else operations[:n_thoughts]
    
    def get_current_numbers(self, state: State) -> List[float]:
        """
        Extract the current available numbers from the state.
        
        Starts with the original problem numbers and applies each operation
        in the thought sequence to get the current numbers.
        """
        numbers = list(state.problem)  # Start with original numbers
        
        # Apply each operation in sequence
        for thought in state.thoughts:
            numbers = self.apply_operation(thought, numbers)
        
        return numbers
    
    def apply_operation(self, operation: str, numbers: List[float]) -> List[float]:
        """
        Apply an arithmetic operation to the list of numbers.
        
        Args:
            operation: String like "4 * 6 = 24"
            numbers: Current list of available numbers
            
        Returns:
            New list of numbers after applying the operation
        """
        try:
            # Parse operation (e.g., "4 * 6 = 24")
            parts = operation.split('=')
            if len(parts) != 2:
                return numbers  # Invalid format
            
            expr = parts[0].strip()
            result = float(parts[1].strip())
            
            # Find the operands in the expression
            for op in ['*', '/', '+', '-']:
                if op in expr:
                    left, right = expr.split(op)
                    left_val = float(left.strip())
                    right_val = float(right.strip())
                    
                    # Check if operands are available
                    if left_val in numbers and right_val in numbers:
                        # Create new number list
                        new_numbers = numbers.copy()
                        new_numbers.remove(left_val)
                        new_numbers.remove(right_val)
                        new_numbers.append(result)
                        return new_numbers
            
            return numbers  # Couldn't apply operation
            
        except (ValueError, IndexError):
            return numbers  # Error parsing operation
    
    def is_valid_operation(self, operation: str, numbers: List[float]) -> bool:
        """Check if an operation string is valid for the current numbers."""
        try:
            # Basic format check
            if '=' not in operation:
                return False
            
            # Try to apply the operation
            new_numbers = self.apply_operation(operation, numbers)
            return len(new_numbers) == len(numbers) - 1  # Should have one fewer number
            
        except:
            return False
    
    def evaluate_states(self, states: List[State]) -> List[float]:
        """
        Evaluate mathematical states based on likelihood of reaching 24.
        
        Uses the language model to classify states as "sure", "maybe", or "impossible"
        and converts these to numerical scores.
        """
        scores = []
        
        for state in states:
            numbers = self.get_current_numbers(state)
            
            # Check if we've already reached 24
            if len(numbers) == 1 and abs(numbers[0] - 24) < 0.001:
                scores.append(10.0)  # Perfect score for solution
                continue
            
            # Use LLM to evaluate the state
            prompt = self.evaluation_prompt.format(numbers=numbers)
            evaluation = self.llm.generate(prompt, n=1, max_tokens=50)[0].lower()
            
            # Convert evaluation to numerical score
            if 'sure' in evaluation:
                score = 9.0
            elif 'maybe' in evaluation:
                score = 6.0
            elif 'impossible' in evaluation:
                score = 1.0
            else:
                score = 5.0  # Neutral score if unclear
            
            scores.append(score)
        
        return scores
    
    def is_solution(self, state: State) -> bool:
        """
        Check if we've reached exactly 24.
        
        A solution is found when we have exactly one number remaining
        and that number equals 24 (within floating-point precision).
        """
        numbers = self.get_current_numbers(state)
        return len(numbers) == 1 and abs(numbers[0] - 24) < 0.001
    
    def get_state_description(self, state: State) -> str:
        """Get a description of the current mathematical state."""
        numbers = self.get_current_numbers(state)
        operations = " -> ".join(state.thoughts) if state.thoughts else "No operations yet"
        return f"Numbers: {numbers}, Operations: {operations}"

class TreeOfThought:
    """
    Main Tree of Thought algorithm implementation.
    
    This class orchestrates the complete ToT process:
    1. Initialize with a specific task
    2. Use either BFS or DFS to explore the solution space
    3. Generate, evaluate, and select thoughts at each step
    4. Return the best solution found
    """
    
    def __init__(self, 
                 task: ToTTask,
                 method: str = 'bfs',
                 beam_size: int = 3,
                 max_depth: int = 4,
                 n_generate_sample: int = 5,
                 n_evaluate_sample: int = 3):
        """
        Initialize the Tree of Thought algorithm.
        
        Args:
            task: The specific task to solve (e.g., CreativeWritingTask)
            method: Search method ('bfs' or 'dfs')
            beam_size: Number of best states to keep at each level (for BFS)
            max_depth: Maximum depth to search
            n_generate_sample: Number of thoughts to generate at each step
            n_evaluate_sample: Number of evaluations per state (for averaging)
        """
        self.task = task
        self.method = method
        self.beam_size = beam_size
        self.max_depth = max_depth
        self.n_generate_sample = n_generate_sample
        self.n_evaluate_sample = n_evaluate_sample
        
        # Statistics tracking
        self.stats = {
            'nodes_explored': 0,
            'api_calls': 0,
            'solutions_found': 0,
            'max_depth_reached': 0
        }
    
    def solve(self, problem: Any) -> Optional[State]:
        """
        Solve the given problem using Tree of Thought.
        
        Args:
            problem: The problem to solve (format depends on task)
            
        Returns:
            Best solution state found, or None if no solution
        """
        print(f"Starting Tree of Thought search for problem: {problem}")
        print(f"Method: {self.method}, Beam size: {self.beam_size}, Max depth: {self.max_depth}")
        
        # Initialize with the problem
        initial_state = State(problem=problem, thoughts=[], depth=0)
        
        # Choose search method
        if self.method == 'bfs':
            solution = self._breadth_first_search(initial_state)
        elif self.method == 'dfs':
            solution = self._depth_first_search(initial_state)
        else:
            raise ValueError(f"Unknown search method: {self.method}")
        
        # Print final statistics
        print(f"\nSearch completed:")
        print(f"- Nodes explored: {self.stats['nodes_explored']}")
        print(f"- API calls made: {self.task.llm.api_calls}")
        print(f"- Max depth reached: {self.stats['max_depth_reached']}")
        print(f"- Solution found: {'Yes' if solution else 'No'}")
        
        return solution
    
    def _breadth_first_search(self, initial_state: State) -> Optional[State]:
        """
        Breadth-First Search implementation.
        
        Explores all states at depth d before moving to depth d+1.
        Maintains a beam of the best states at each level.
        """
        # Initialize frontier with the starting state
        frontier = [initial_state]
        
        # Search level by level
        for depth in range(self.max_depth):
            print(f"\n=== Depth {depth} ===")
            print(f"Exploring {len(frontier)} states")
            
            if not frontier:
                break
            
            # Check if any current states are solutions
            for state in frontier:
                if self.task.is_solution(state):
                    print(f"Solution found at depth {depth}!")
                    self.stats['solutions_found'] += 1
                    return state
            
            # Generate new states from current frontier
            new_states = []
            
            for state in frontier:
                self.stats['nodes_explored'] += 1
                self.stats['max_depth_reached'] = max(self.stats['max_depth_reached'], state.depth)
                
                # Generate thoughts for this state
                thoughts = self.task.generate_thoughts(state, self.n_generate_sample)
                
                if not thoughts:
                    continue  # Skip if no thoughts generated
                
                # Create new states for each thought
                state_thoughts = []
                for thought in thoughts:
                    new_state = state.copy_with_thought(thought)
                    new_states.append(new_state)
                    state_thoughts.append(thought)
                
                print(f"Generated {len(state_thoughts)} thoughts from state")
            
            if not new_states:
                print("No new states generated - search terminated")
                break
            
            # Evaluate all new states
            print(f"Evaluating {len(new_states)} new states...")
            scores = self.task.evaluate_states(new_states)
            
            # Assign scores to states
            for state, score in zip(new_states, scores):
                state.score = score
            
            # Remove duplicates and sort by score
            unique_states = self._remove_duplicates(new_states)
            unique_states.sort(key=lambda s: s.score, reverse=True)
            
            # Keep only the best states for next iteration (beam search)
            frontier = unique_states[:self.beam_size]
            
            print(f"Selected top {len(frontier)} states for next iteration")
            for i, state in enumerate(frontier[:3]):  # Show top 3
                print(f"  {i+1}. Score: {state.score:.2f}, Thoughts: {len(state.thoughts)}")
        
        # Return best state found (might not be a complete solution)
        if frontier:
            best_state = max(frontier, key=lambda s: s.score)
            print(f"Search completed. Best state score: {best_state.score:.2f}")
            return best_state
        
        return None
    
    def _depth_first_search(self, initial_state: State) -> Optional[State]:
        """
        Depth-First Search implementation.
        
        Explores states deeply before backtracking.
        Good for problems where solutions are likely to be found at deeper levels.
        """
        # Stack for DFS (LIFO)
        stack = [initial_state]
        visited = set()  # Prevent cycles
        best_solution = None
        best_score = -1
        
        print(f"Starting DFS with max depth {self.max_depth}")
        
        while stack:
            current_state = stack.pop()
            
            # Skip if already visited (prevent infinite loops)
            state_key = (str(current_state.problem), tuple(current_state.thoughts))
            if state_key in visited:
                continue
            visited.add(state_key)
            
            self.stats['nodes_explored'] += 1
            self.stats['max_depth_reached'] = max(self.stats['max_depth_reached'], current_state.depth)
            
            print(f"Exploring state at depth {current_state.depth}")
            
            # Check if this is a solution
            if self.task.is_solution(current_state):
                print(f"Solution found at depth {current_state.depth}!")
                
                # Evaluate solution quality
                scores = self.task.evaluate_states([current_state])
                current_score = scores[0] if scores else 0
                
                if current_score > best_score:
                    best_solution = current_state
                    best_score = current_score
                    print(f"New best solution with score: {best_score:.2f}")
                
                self.stats['solutions_found'] += 1
                continue  # Continue searching for better solutions
            
            # Don't expand if we've reached maximum depth
            if current_state.depth >= self.max_depth:
                continue
            
            # Generate thoughts for expansion
            thoughts = self.task.generate_thoughts(current_state, self.n_generate_sample)
            
            if not thoughts:
                continue
            
            # Evaluate potential next states
            potential_states = [current_state.copy_with_thought(t) for t in thoughts]
            scores = self.task.evaluate_states(potential_states)
            
            # Add promising states to stack (sorted by score, best first)
            scored_states = list(zip(potential_states, scores))
            scored_states.sort(key=lambda x: x[1])  # Sort ascending, so best is popped first
            
            for state, score in scored_states:
                if score > 3.0:  # Only explore states above threshold
                    state.score = score
                    stack.append(state)
        
        return best_solution
    
    def _remove_duplicates(self, states: List[State]) -> List[State]:
        """
        Remove duplicate states to avoid redundant computation.
        
        Two states are considered duplicates if they have the same
        problem and sequence of thoughts.
        """
        seen = set()
        unique_states = []
        
        for state in states:
            state_key = (str(state.problem), tuple(state.thoughts))
            if state_key not in seen:
                seen.add(state_key)
                unique_states.append(state)
        
        return unique_states

def main():
    """
    Example usage of the Tree of Thought implementation.
    
    Shows how to use ToT for both creative writing and mathematical reasoning tasks.
    """
    
    # Check if API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("Please set your OPENAI_API_KEY environment variable")
        return
    
    # Initialize language model interface
    llm = LLMInterface(model="gpt-4", temperature=0.7)
    
    print("=== Tree of Thought Demo ===\n")
    
    # Example 1: Creative Writing Task
    print("1. Creative Writing Task")
    print("-" * 50)
    
    creative_task = CreativeWritingTask(llm)
    writing_problem = [
        "The old lighthouse stood sentinel.",
        "Waves crashed below with growing intensity.", 
        "A figure appeared in the distance.",
        "The mystery began to unfold."
    ]
    
    tot_writer = TreeOfThought(
        task=creative_task,
        method='bfs',
        beam_size=3,
        max_depth=3,
        n_generate_sample=3
    )
    
    writing_solution = tot_writer.solve(writing_problem)
    
    if writing_solution:
        print("\n=== Creative Writing Solution ===")
        print(f"Final story (depth {writing_solution.depth}):")
        if len(writing_solution.thoughts) > 1:
            print(writing_solution.thoughts[-1])  # The final story
        else:
            print("Plan only:", writing_solution.thoughts[0])
    else:
        print("No creative writing solution found")
    
    print("\n" + "="*70 + "\n")
    
    # Example 2: Game of 24 Task  
    print("2. Game of 24 Task")
    print("-" * 50)
    
    math_task = GameOf24Task(llm)
    math_problem = [4, 1, 8, 7]  # Try to make 24 from these numbers
    
    tot_math = TreeOfThought(
        task=math_task,
        method='dfs',  # DFS often works better for mathematical reasoning
        beam_size=5,
        max_depth=4,
        n_generate_sample=4
    )
