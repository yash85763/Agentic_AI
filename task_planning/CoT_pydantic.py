from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, validator
from enum import Enum
import json
import asyncio
from datetime import datetime

class ReasoningStepType(str, Enum):
“”“Types of reasoning steps in the chain of thought”””
ANALYSIS = “analysis”
HYPOTHESIS = “hypothesis”
CALCULATION = “calculation”
DEDUCTION = “deduction”
CONCLUSION = “conclusion”
VERIFICATION = “verification”

class ReasoningStep(BaseModel):
“”“Individual step in the chain of thought reasoning”””
step_number: int = Field(…, ge=1, description=“Sequential step number”)
step_type: ReasoningStepType = Field(…, description=“Type of reasoning step”)
description: str = Field(…, min_length=10, description=“Description of what this step does”)
input_data: Optional[Dict[str, Any]] = Field(default=None, description=“Input data for this step”)
reasoning: str = Field(…, min_length=20, description=“Detailed reasoning for this step”)
output: Any = Field(…, description=“Output or result of this step”)
confidence: float = Field(…, ge=0.0, le=1.0, description=“Confidence level (0-1)”)
dependencies: List[int] = Field(default=[], description=“Step numbers this step depends on”)

```
@validator('dependencies')
def validate_dependencies(cls, v, values):
    """Ensure dependencies reference earlier steps"""
    if 'step_number' in values:
        current_step = values['step_number']
        for dep in v:
            if dep >= current_step:
                raise ValueError(f"Dependency {dep} must be earlier than current step {current_step}")
    return v
```

class ChainOfThought(BaseModel):
“”“Complete chain of thought reasoning structure”””
question: str = Field(…, min_length=5, description=“Original question or problem”)
context: Optional[str] = Field(default=None, description=“Additional context for the problem”)
reasoning_steps: List[ReasoningStep] = Field(…, min_items=1, description=“Sequential reasoning steps”)
final_answer: str = Field(…, min_length=1, description=“Final answer to the question”)
overall_confidence: float = Field(…, ge=0.0, le=1.0, description=“Overall confidence in the answer”)
timestamp: datetime = Field(default_factory=datetime.now, description=“When this reasoning was generated”)
metadata: Dict[str, Any] = Field(default_factory=dict, description=“Additional metadata”)

```
@validator('reasoning_steps')
def validate_step_sequence(cls, v):
    """Ensure steps are properly numbered and sequential"""
    expected_step = 1
    for step in v:
        if step.step_number != expected_step:
            raise ValueError(f"Expected step {expected_step}, got {step.step_number}")
        expected_step += 1
    return v

@validator('overall_confidence')
def calculate_overall_confidence(cls, v, values):
    """Calculate overall confidence based on individual step confidences"""
    if 'reasoning_steps' in values and values['reasoning_steps']:
        # Use the minimum confidence as a conservative estimate
        step_confidences = [step.confidence for step in values['reasoning_steps']]
        calculated_confidence = min(step_confidences)
        # Allow some tolerance for manually set confidence
        if abs(v - calculated_confidence) > 0.2:
            raise ValueError(f"Overall confidence {v} differs too much from calculated {calculated_confidence}")
    return v
```

class LLMInterface(BaseModel):
“”“Interface for LLM interactions”””
model_name: str = Field(default=“gpt-3.5-turbo”, description=“LLM model to use”)
api_key: Optional[str] = Field(default=None, description=“API key for the LLM service”)
base_url: Optional[str] = Field(default=None, description=“Base URL for API calls”)

```
class Config:
    # Don't include sensitive fields in serialization
    fields = {'api_key': {'write_only': True}}
```

class ChainOfThoughtEngine:
“”“Main engine for generating chain of thought reasoning using an LLM”””

```
def __init__(self, llm_interface: LLMInterface):
    self.llm = llm_interface
    
async def generate_reasoning_step(
    self, 
    question: str, 
    context: Optional[str],
    previous_steps: List[ReasoningStep],
    step_number: int,
    step_type: ReasoningStepType
) -> ReasoningStep:
    """Generate a single reasoning step using the LLM"""
    
    # Prepare context for the LLM
    prompt = self._build_step_prompt(question, context, previous_steps, step_number, step_type)
    
    # Simulate LLM call (replace with actual API call)
    llm_response = await self._call_llm(prompt)
    
    # Parse LLM response into structured format
    step_data = self._parse_llm_response(llm_response, step_number, step_type)
    
    return ReasoningStep(**step_data)

def _build_step_prompt(
    self,
    question: str,
    context: Optional[str],
    previous_steps: List[ReasoningStep],
    step_number: int,
    step_type: ReasoningStepType
) -> str:
    """Build a prompt for generating a reasoning step"""
    
    prompt_parts = [
        f"Question: {question}",
        f"Context: {context or 'No additional context provided'}",
        "",
        "Previous reasoning steps:"
    ]
    
    for step in previous_steps:
        prompt_parts.append(f"Step {step.step_number} ({step.step_type}): {step.reasoning}")
        prompt_parts.append(f"Output: {step.output}")
        prompt_parts.append("")
    
    prompt_parts.extend([
        f"Generate step {step_number} of type '{step_type}' for the chain of thought reasoning.",
        "Provide:",
        "1. A clear description of what this step accomplishes",
        "2. Detailed reasoning for this step",
        "3. The output or result of this step",
        "4. Your confidence level (0.0 to 1.0)",
        "5. Any dependencies on previous steps (list of step numbers)",
        "",
        "Format your response as JSON with keys: description, reasoning, output, confidence, dependencies"
    ])
    
    return "\n".join(prompt_parts)

async def _call_llm(self, prompt: str) -> str:
    """Make an API call to the LLM (mock implementation)"""
    # This is a mock implementation - replace with actual LLM API call
    await asyncio.sleep(0.1)  # Simulate API latency
    
    # Mock response based on prompt analysis
    if "calculate" in prompt.lower() or "computation" in prompt.lower():
        return json.dumps({
            "description": "Perform mathematical calculation based on given values",
            "reasoning": "I need to apply the appropriate mathematical formula to the given inputs to get the numerical result",
            "output": "42",
            "confidence": 0.9,
            "dependencies": [1] if "step 2" in prompt.lower() else []
        })
    else:
        return json.dumps({
            "description": "Analyze the problem and identify key components",
            "reasoning": "Breaking down the question into its fundamental components to understand what needs to be solved",
            "output": "Identified key variables and relationships",
            "confidence": 0.85,
            "dependencies": []
        })

def _parse_llm_response(self, response: str, step_number: int, step_type: ReasoningStepType) -> Dict[str, Any]:
    """Parse LLM response into structured data"""
    try:
        data = json.loads(response)
        data['step_number'] = step_number
        data['step_type'] = step_type
        return data
    except json.JSONDecodeError:
        # Fallback for malformed JSON
        return {
            'step_number': step_number,
            'step_type': step_type,
            'description': f"Generated step {step_number}",
            'reasoning': response[:200] + "..." if len(response) > 200 else response,
            'output': "Parsed from unstructured response",
            'confidence': 0.5,
            'dependencies': []
        }

async def generate_chain_of_thought(
    self, 
    question: str, 
    context: Optional[str] = None,
    max_steps: int = 5
) -> ChainOfThought:
    """Generate a complete chain of thought reasoning"""
    
    reasoning_steps = []
    step_types = [
        ReasoningStepType.ANALYSIS,
        ReasoningStepType.HYPOTHESIS,
        ReasoningStepType.CALCULATION,
        ReasoningStepType.DEDUCTION,
        ReasoningStepType.CONCLUSION
    ]
    
    # Generate reasoning steps
    for i in range(min(max_steps, len(step_types))):
        step = await self.generate_reasoning_step(
            question=question,
            context=context,
            previous_steps=reasoning_steps,
            step_number=i + 1,
            step_type=step_types[i]
        )
        reasoning_steps.append(step)
    
    # Generate final answer based on all steps
    final_answer = await self._generate_final_answer(question, reasoning_steps)
    
    # Calculate overall confidence
    overall_confidence = min([step.confidence for step in reasoning_steps])
    
    return ChainOfThought(
        question=question,
        context=context,
        reasoning_steps=reasoning_steps,
        final_answer=final_answer,
        overall_confidence=overall_confidence,
        metadata={
            'model_used': self.llm.model_name,
            'steps_generated': len(reasoning_steps)
        }
    )

async def _generate_final_answer(self, question: str, steps: List[ReasoningStep]) -> str:
    """Generate final answer based on reasoning steps"""
    # Mock implementation - would use LLM to synthesize final answer
    last_step_output = steps[-1].output if steps else "No reasoning steps provided"
    return f"Based on the chain of reasoning, the answer is: {last_step_output}"
```

# Usage example

async def example_usage():
“”“Example of how to use the Chain of Thought system”””

```
# Initialize the LLM interface
llm_interface = LLMInterface(
    model_name="gpt-3.5-turbo",
    api_key="your-api-key-here"
)

# Create the reasoning engine
engine = ChainOfThoughtEngine(llm_interface)

# Generate chain of thought for a question
question = "What is the area of a circle with radius 5 meters?"
context = "Use π ≈ 3.14159 for calculations"

try:
    cot = await engine.generate_chain_of_thought(
        question=question,
        context=context,
        max_steps=4
    )
    
    print("Chain of Thought Analysis:")
    print(f"Question: {cot.question}")
    print(f"Overall Confidence: {cot.overall_confidence:.2f}")
    print("\nReasoning Steps:")
    
    for step in cot.reasoning_steps:
        print(f"\nStep {step.step_number} ({step.step_type.value}):")
        print(f"Description: {step.description}")
        print(f"Reasoning: {step.reasoning}")
        print(f"Output: {step.output}")
        print(f"Confidence: {step.confidence:.2f}")
        if step.dependencies:
            print(f"Dependencies: {step.dependencies}")
    
    print(f"\nFinal Answer: {cot.final_answer}")
    
    # Validate the structure
    print(f"\nValidation: Chain of thought is valid - {len(cot.reasoning_steps)} steps generated")
    
    # Export to JSON
    json_output = cot.json(indent=2)
    print(f"\nJSON representation length: {len(json_output)} characters")
    
except Exception as e:
    print(f"Error generating chain of thought: {e}")
```

# Helper function to run the example

def run_example():
“”“Run the example usage”””
asyncio.run(example_usage())

if **name** == “**main**”:
run_example()