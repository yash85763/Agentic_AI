# Chain of Thought Implementations

# This file demonstrates Chain of Thought reasoning using different frameworks

# =============================================================================

# 1. PYDANTIC VERSION

# =============================================================================

from pydantic import BaseModel, Field
from typing import List, Optional
import json

class ThoughtStep(BaseModel):
step_number: int
description: str
reasoning: str
intermediate_result: Optional[str] = None

class ChainOfThought(BaseModel):
question: str
steps: List[ThoughtStep] = Field(default_factory=list)
final_answer: Optional[str] = None

```
def add_step(self, description: str, reasoning: str, result: Optional[str] = None):
    step = ThoughtStep(
        step_number=len(self.steps) + 1,
        description=description,
        reasoning=reasoning,
        intermediate_result=result
    )
    self.steps.append(step)

def solve_math_problem(self, problem: str):
    self.question = problem
    
    # Example: "Sarah has 15 apples. She gives away 1/3 to her friends and eats 2. How many apples does she have left?"
    self.add_step(
        description="Identify initial quantity",
        reasoning="Sarah starts with 15 apples",
        result="15 apples"
    )
    
    self.add_step(
        description="Calculate apples given away",
        reasoning="1/3 of 15 = 15 ÷ 3 = 5",
        result="5 apples given away"
    )
    
    self.add_step(
        description="Calculate apples after giving some away",
        reasoning="15 - 5 = 10",
        result="10 apples remaining"
    )
    
    self.add_step(
        description="Calculate apples after eating some",
        reasoning="She eats 2 apples: 10 - 2 = 8",
        result="8 apples remaining"
    )
    
    self.final_answer = "8 apples"
    return self
```

# Usage example for Pydantic version

def demo_pydantic_cot():
print(”=== PYDANTIC CHAIN OF THOUGHT ===”)
cot = ChainOfThought()
result = cot.solve_math_problem(“Sarah has 15 apples. She gives away 1/3 to her friends and eats 2. How many apples does she have left?”)

```
print(f"Question: {result.question}")
for step in result.steps:
    print(f"Step {step.step_number}: {step.description}")
    print(f"  Reasoning: {step.reasoning}")
    print(f"  Result: {step.intermediate_result}")
print(f"Final Answer: {result.final_answer}\n")
```

# =============================================================================

# 2. DSPY VERSION

# =============================================================================

try:
import dspy
from dspy import Signature, InputField, OutputField

```
class ChainOfThoughtSignature(Signature):
    """Break down the problem into logical steps and solve systematically."""
    question: str = InputField(desc="The problem to solve")
    reasoning_steps: str = OutputField(desc="Step-by-step reasoning process")
    final_answer: str = OutputField(desc="The final answer")

class DSPyChainOfThought:
    def __init__(self):
        # Configure DSPy (you'd normally set up your LM here)
        # dspy.configure(lm=your_language_model)
        self.cot_module = dspy.ChainOfThought(ChainOfThoughtSignature)
    
    def solve(self, question: str):
        # This would normally call the LM, but for demo purposes:
        steps = [
            "1. Identify what we know: Sarah has 15 apples initially",
            "2. Calculate 1/3 of 15: 15 ÷ 3 = 5 apples given away", 
            "3. Subtract given apples: 15 - 5 = 10 apples left",
            "4. Subtract eaten apples: 10 - 2 = 8 apples remaining"
        ]
        
        return {
            'question': question,
            'reasoning_steps': '\n'.join(steps),
            'final_answer': '8 apples'
        }

def demo_dspy_cot():
    print("=== DSPY CHAIN OF THOUGHT ===")
    cot = DSPyChainOfThought()
    result = cot.solve("Sarah has 15 apples. She gives away 1/3 to her friends and eats 2. How many apples does she have left?")
    
    print(f"Question: {result['question']}")
    print(f"Reasoning Steps:\n{result['reasoning_steps']}")
    print(f"Final Answer: {result['final_answer']}\n")
```

except ImportError:
def demo_dspy_cot():
print(”=== DSPY CHAIN OF THOUGHT ===”)
print(“DSPy not installed. Run: pip install dspy-ai”)
print(“Here’s what the DSPy version would look like:”)
print(”- Uses Signature to define input/output structure”)
print(”- Leverages dspy.ChainOfThought module”)
print(”- Automatically optimizes prompting strategies\n”)

# =============================================================================

# 3. LANGCHAIN VERSION

# =============================================================================

try:
from langchain.schema import BaseOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.base import LLM
from typing import Any

```
class ChainOfThoughtParser(BaseOutputParser):
    def parse(self, text: str) -> dict:
        # Simple parser for demo
        lines = text.strip().split('\n')
        steps = []
        final_answer = None
        
        for line in lines:
            if line.startswith('Step'):
                steps.append(line)
            elif 'Final Answer:' in line:
                final_answer = line.split('Final Answer:')[1].strip()
        
        return {
            'steps': steps,
            'final_answer': final_answer
        }

class MockLLM(LLM):
    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs: Any) -> str:
        # Mock response for demo
        return """Step 1: Identify initial quantity - Sarah has 15 apples
```

Step 2: Calculate 1/3 of 15 - 15 ÷ 3 = 5 apples to give away  
Step 3: Subtract given apples - 15 - 5 = 10 apples remaining
Step 4: Subtract eaten apples - 10 - 2 = 8 apples left
Final Answer: 8 apples”””

```
    @property
    def _llm_type(self) -> str:
        return "mock"

class LangChainCoT:
    def __init__(self):
        self.llm = MockLLM()
        self.parser = ChainOfThoughtParser()
        
        template = """
        Solve this problem step by step using chain of thought reasoning:
        
        Problem: {question}
        
        Break it down into clear steps:
        """
        
        self.prompt = PromptTemplate(
            input_variables=["question"],
            template=template
        )
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            output_parser=self.parser
        )
    
    def solve(self, question: str):
        result = self.chain.run(question=question)
        return result

def demo_langchain_cot():
    print("=== LANGCHAIN CHAIN OF THOUGHT ===")
    cot = LangChainCoT()
    result = cot.solve("Sarah has 15 apples. She gives away 1/3 to her friends and eats 2. How many apples does she have left?")
    
    print("Reasoning Steps:")
    for step in result['steps']:
        print(f"  {step}")
    print(f"Final Answer: {result['final_answer']}\n")
```

except ImportError:
def demo_langchain_cot():
print(”=== LANGCHAIN CHAIN OF THOUGHT ===”)
print(“LangChain not installed. Run: pip install langchain”)
print(“Here’s what the LangChain version would include:”)
print(”- Custom output parser for structured reasoning”)
print(”- Prompt templates for consistent formatting”)
print(”- LLM chain integration\n”)

# =============================================================================

# 4. LANGGRAPH VERSION

# =============================================================================

try:
from langgraph.graph import Graph, END
from typing import TypedDict

```
class ReasoningState(TypedDict):
    question: str
    steps: List[str]
    current_step: int
    final_answer: str

class LangGraphCoT:
    def __init__(self):
        self.graph = self._build_graph()
    
    def _build_graph(self):
        graph = Graph()
        
        # Define the workflow nodes
        graph.add_node("identify_problem", self.identify_problem)
        graph.add_node("break_down", self.break_down_problem)
        graph.add_node("solve_step", self.solve_current_step)
        graph.add_node("check_complete", self.check_if_complete)
        graph.add_node("finalize", self.finalize_answer)
        
        # Define the edges
        graph.set_entry_point("identify_problem")
        graph.add_edge("identify_problem", "break_down")
        graph.add_edge("break_down", "solve_step")
        graph.add_conditional_edges(
            "solve_step",
            self.should_continue,
            {
                "continue": "solve_step",
                "finish": "finalize"
            }
        )
        graph.add_edge("finalize", END)
        
        return graph.compile()
    
    def identify_problem(self, state: ReasoningState) -> ReasoningState:
        state["steps"] = ["Identified problem: " + state["question"]]
        state["current_step"] = 0
        return state
    
    def break_down_problem(self, state: ReasoningState) -> ReasoningState:
        # For demo, we'll use predefined steps
        reasoning_steps = [
            "Calculate 1/3 of 15 apples given away",
            "Subtract given apples from initial amount", 
            "Subtract eaten apples from remaining amount"
        ]
        state["steps"].extend(reasoning_steps)
        return state
    
    def solve_current_step(self, state: ReasoningState) -> ReasoningState:
        step = state["current_step"]
        solutions = [
            "1/3 of 15 = 5 apples given away",
            "15 - 5 = 10 apples remaining", 
            "10 - 2 = 8 apples left"
        ]
        
        if step < len(solutions):
            state["steps"].append(f"Solution: {solutions[step]}")
            state["current_step"] += 1
        
        return state
    
    def should_continue(self, state: ReasoningState) -> str:
        if state["current_step"] >= 3:  # We have 3 solution steps
            return "finish"
        return "continue"
    
    def finalize_answer(self, state: ReasoningState) -> ReasoningState:
        state["final_answer"] = "8 apples"
        return state
    
    def solve(self, question: str) -> ReasoningState:
        initial_state = ReasoningState(
            question=question,
            steps=[],
            current_step=0,
            final_answer=""
        )
        
        result = self.graph.invoke(initial_state)
        return result

def demo_langgraph_cot():
    print("=== LANGGRAPH CHAIN OF THOUGHT ===")
    cot = LangGraphCoT()
    result = cot.solve("Sarah has 15 apples. She gives away 1/3 to her friends and eats 2. How many apples does she have left?")
    
    print(f"Question: {result['question']}")
    print("Reasoning Process:")
    for i, step in enumerate(result['steps'], 1):
        print(f"  {i}. {step}")
    print(f"Final Answer: {result['final_answer']}\n")
```

except ImportError:
def demo_langgraph_cot():
print(”=== LANGGRAPH CHAIN OF THOUGHT ===”)
print(“LangGraph not installed. Run: pip install langgraph”)
print(“Here’s what the LangGraph version would feature:”)
print(”- State-based graph workflow”)
print(”- Conditional routing between reasoning steps”)
print(”- Structured multi-step processing\n”)

# =============================================================================

# 5. VANILLA PYTHON VERSION

# =============================================================================

class VanillaChainOfThought:
def **init**(self):
self.steps = []
self.question = “”
self.final_answer = “”

```
def think(self, description: str, reasoning: str, calculation: str = None):
    """Add a reasoning step"""
    step = {
        'step_number': len(self.steps) + 1,
        'description': description,
        'reasoning': reasoning,
        'calculation': calculation,
        'result': None
    }
    
    # If there's a calculation, try to evaluate it
    if calculation:
        try:
            # Simple eval for basic math (in real app, use safer parsing)
            if any(op in calculation for op in ['+', '-', '*', '/', '//']):
                # Replace division symbols for Python eval
                calc = calculation.replace('÷', '//')
                step['result'] = eval(calc)
            else:
                step['result'] = calculation
        except:
            step['result'] = calculation
    
    self.steps.append(step)
    return step['result']

def solve_math_word_problem(self, problem: str):
    """Solve a math word problem using chain of thought"""
    self.question = problem
    self.steps = []
    
    # Step 1: Identify what we know
    self.think(
        "Identify initial conditions",
        "Sarah starts with some number of apples",
        "15"
    )
    
    # Step 2: Identify what happens first
    apples_given = self.think(
        "Calculate apples given to friends",
        "She gives away 1/3 of her apples. 1/3 of 15",
        "15 // 3"
    )
    
    # Step 3: Calculate remaining after giving away
    after_giving = self.think(
        "Calculate apples after giving some away", 
        f"Subtract the {apples_given} apples given away from original 15",
        f"15 - {apples_given}"
    )
    
    # Step 4: Calculate final amount
    final_amount = self.think(
        "Calculate apples after eating",
        f"She eats 2 apples from the remaining {after_giving}",
        f"{after_giving} - 2"
    )
    
    self.final_answer = f"{final_amount} apples"
    return self

def display_reasoning(self):
    """Display the complete chain of thought"""
    print(f"Problem: {self.question}")
    print("\nReasoning Process:")
    
    for step in self.steps:
        print(f"\nStep {step['step_number']}: {step['description']}")
        print(f"  Thought: {step['reasoning']}")
        if step['calculation']:
            print(f"  Calculation: {step['calculation']}")
        if step['result'] is not None:
            print(f"  Result: {step['result']}")
    
    print(f"\nFinal Answer: {self.final_answer}")
```

def demo_vanilla_cot():
print(”=== VANILLA PYTHON CHAIN OF THOUGHT ===”)
cot = VanillaChainOfThought()
cot.solve_math_word_problem(“Sarah has 15 apples. She gives away 1/3 to her friends and eats 2. How many apples does she have left?”)
cot.display_reasoning()
print()

# =============================================================================

# DEMONSTRATION RUNNER

# =============================================================================

def run_all_demos():
“”“Run all Chain of Thought implementations”””
print(“CHAIN OF THOUGHT REASONING IMPLEMENTATIONS\n”)
print(”=”*60)

```
demo_pydantic_cot()
demo_dspy_cot() 
demo_langchain_cot()
demo_langgraph_cot()
demo_vanilla_cot()

print("="*60)
print("COMPARISON SUMMARY:")
print("- Pydantic: Type-safe, structured data validation")
print("- DSPy: Optimized prompting and LM integration") 
print("- LangChain: Flexible chains with parsing")
print("- LangGraph: State-based workflow graphs")
print("- Vanilla: Simple, no dependencies, full control")
```

if **name** == “**main**”:
run_all_demos()