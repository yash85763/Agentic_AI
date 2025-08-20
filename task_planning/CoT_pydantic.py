from typing import List, Optional, Any, Dict, Self
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_core import ValidationError
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
step_number: int = Field(ge=1, description=“Sequential step number”)
step_type: ReasoningStepType = Field(description=“Type of reasoning step”)
description: str = Field(min_length=10, description=“Description of what this step does”)
input_data: Optional[Dict[str, Any]] = Field(default=None, description=“Input data for this step”)
reasoning: str = Field(min_length=20, description=“Detailed reasoning for this step”)
output: Any = Field(description=“Output or result of this step”)
confidence: float = Field(ge=0.0, le=1.0, description=“Confidence level (0-1)”)
dependencies: List[int] = Field(default_factory=list, description=“Step numbers this step depends on”)

```
@model_validator(mode='after')
def validate_dependencies(self) -> Self:
    """Ensure dependencies reference earlier steps"""
    for dep in self.dependencies:
        if dep >= self.step_number:
            raise ValueError(f"Dependency {dep} must be earlier than current step {self.step_number}")
    return self
```

class ChainOfThought(BaseModel):
“”“Complete chain of thought reasoning structure”””
question: str = Field(min_length=5, description=“Original question or problem”)
context: Optional[str] = Field(default=None, description=“Additional context for the problem”)
reasoning_steps: List[ReasoningStep] = Field(min_length=1, description=“Sequential reasoning steps”)
final_answer: str = Field(min_length=1, description=“Final answer to the question”)
overall_confidence: float = Field(ge=0.0, le=1.0, description=“Overall confidence in the answer”)
timestamp: datetime = Field(default_factory=datetime.now, description=“When this reasoning was generated”)
metadata: Dict[str, Any] = Field(default_factory=dict, description=“Additional metadata”)

```
@field_validator('reasoning_steps')
@classmethod
def validate_step_sequence(cls, v: List[ReasoningStep]) -> List[ReasoningStep]:
    """Ensure steps are properly numbered and sequential"""
    for i, step in enumerate(v, 1):
        if step.step_number != i:
            raise ValueError(f"Expected step {i}, got step {step.step_number}")
    return v

@model_validator(mode='after')
def validate_overall_confidence(self) -> Self:
    """Validate overall confidence based on individual step confidences"""
    if self.reasoning_steps:
        step_confidences = [step.confidence for step in self.reasoning_steps]
        calculated_confidence = min(step_confidences)
        # Allow some tolerance for manually set confidence
        if abs(self.overall_confidence - calculated_confidence) > 0.2:
            raise ValueError(
                f"Overall confidence {self.overall_confidence:.2f} differs too much "
                f"from calculated minimum {calculated_confidence:.2f}"
            )
    return self
```

class LLMInterface(BaseModel):
“”“Interface for LLM interactions”””
model_name: str = Field(default=“gpt-3.5-turbo”, description=“LLM model to use”)
api_key: Optional[str] = Field(default=None, description=“API key for the LLM service”)
base_url: Optional[str] = Field(default=None, description=“Base URL for API calls”)
max_tokens: int = Field(default=1000, ge=1, description=“Maximum tokens for response”)
temperature: float = Field(default=0.7, ge=0.0, le=2.0, description=“Temperature for response generation”)

```
model_config = {
    "json_schema_extra": {
        "properties": {
            "api_key": {"writeOnly": True}
        }
    }
}
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
    
    # Build prompt for the LLM
    prompt = self._build_step_prompt(question, context, previous_steps, step_number, step_type)
    
    # Call LLM to generate step
    llm_response = await self._call_llm(prompt)
    
    # Parse response into structured format
    step_data = self._parse_llm_response(llm_response, step_number, step_type)
    
    # Create and validate ReasoningStep
    try:
        return ReasoningStep(**step_data)
    except ValidationError as e:
        raise ValueError(f"Failed to create valid reasoning step: {e}")

def _build_step_prompt(
    self,
    question: str,
    context: Optional[str],
    previous_steps: List[ReasoningStep],
    step_number: int,
    step_type: ReasoningStepType
) -> str:
    """Build a comprehensive prompt for generating a reasoning step"""
    
    prompt_sections = [
        "# Chain of Thought Reasoning Task",
        f"**Question:** {question}",
        f"**Context:** {context or 'No additional context provided'}",
        ""
    ]
    
    if previous_steps:
        prompt_sections.append("## Previous Reasoning Steps:")
        for step in previous_steps:
            prompt_sections.extend([
                f"### Step {step.step_number} ({step.step_type.value.title()})",
                f"**Description:** {step.description}",
                f"**Reasoning:** {step.reasoning}",
                f"**Output:** {step.output}",
                f"**Confidence:** {step.confidence:.2f}",
                ""
            ])
    
    prompt_sections.extend([
        f"## Generate Step {step_number}: {step_type.value.title()}",
        "",
        "Please provide the following for this reasoning step:",
        "1. **Description**: A clear description of what this step accomplishes (minimum 10 characters)",
        "2. **Reasoning**: Detailed reasoning for this step (minimum 20 characters)",
        "3. **Output**: The concrete output or result of this step",
        "4. **Confidence**: Your confidence level as a decimal between 0.0 and 1.0",
        "5. **Dependencies**: List of previous step numbers this step depends on (empty array if none)",
        "",
        "**Response Format (JSON):**",
        "```json",
        "{",
        '  "description": "Clear description of step purpose",',
        '  "reasoning": "Detailed explanation of the reasoning process",',
        '  "output": "Concrete result or conclusion",',
        '  "confidence": 0.85,',
        '  "dependencies": [1, 2]',
        "}",
        "```"
    ])
    
    return "\n".join(prompt_sections)

async def _call_llm(self, prompt: str) -> str:
    """Make an API call to the LLM"""
    # This is a sophisticated mock implementation
    # Replace with actual LLM API call (OpenAI, Anthropic, etc.)
    
    await asyncio.sleep(0.1)  # Simulate API latency
    
    # Generate contextually appropriate mock responses
    if "analysis" in prompt.lower():
        return json.dumps({
            "description": "Analyze the core components and requirements of the problem",
            "reasoning": "I need to break down the question into its fundamental elements to understand what information is given, what needs to be found, and what approach would be most appropriate for solving this problem systematically.",
            "output": "Problem decomposed into: given parameters, target outcome, and solution methodology",
            "confidence": 0.85,
            "dependencies": []
        })
    elif "hypothesis" in prompt.lower():
        return json.dumps({
            "description": "Formulate initial hypothesis based on analysis",
            "reasoning": "Based on the analysis, I can form a preliminary hypothesis about the expected outcome. This hypothesis will guide the subsequent calculations and help validate the final result.",
            "output": "Initial hypothesis formed with expected result range",
            "confidence": 0.75,
            "dependencies": [1]
        })
    elif "calculation" in prompt.lower():
        return json.dumps({
            "description": "Perform necessary mathematical computations",
            "reasoning": "Using the identified parameters and methodology, I'll execute the required calculations step by step, ensuring accuracy and showing intermediate results for verification.",
            "output": "Mathematical computation completed with intermediate steps shown",
            "confidence": 0.90,
            "dependencies": [1, 2]
        })
    elif "deduction" in prompt.lower():
        return json.dumps({
            "description": "Draw logical conclusions from calculations and analysis",
            "reasoning": "Based on the computational results and initial hypothesis, I can now deduce the implications and draw logical conclusions that directly address the original question.",
            "output": "Logical conclusions drawn connecting calculations to question requirements",
            "confidence": 0.80,
            "dependencies": [1, 2, 3]
        })
    else:  # conclusion
        return json.dumps({
            "description": "Synthesize all previous steps into final conclusion",
            "reasoning": "Integrating all the analysis, hypothesis, calculations, and deductions to provide a comprehensive answer that directly addresses the original question with supporting evidence.",
            "output": "Final synthesized answer with supporting reasoning chain",
            "confidence": 0.85,
            "dependencies": [1, 2, 3, 4]
        })

def _parse_llm_response(
    self, 
    response: str, 
    step_number: int, 
    step_type: ReasoningStepType
) -> Dict[str, Any]:
    """Parse LLM response into structured data"""
    try:
        # Try to parse JSON response
        data = json.loads(response)
        
        # Add required fields
        data['step_number'] = step_number
        data['step_type'] = step_type
        
        # Ensure all required fields are present with defaults if needed
        required_fields = {
            'description': f"Generated {step_type.value} step",
            'reasoning': "Reasoning provided by LLM",
            'output': "Output generated",
            'confidence': 0.5,
            'dependencies': []
        }
        
        for field, default_value in required_fields.items():
            if field not in data or not data[field]:
                data[field] = default_value
        
        return data
        
    except json.JSONDecodeError:
        # Fallback for malformed JSON
        return {
            'step_number': step_number,
            'step_type': step_type,
            'description': f"Generated {step_type.value} step from unstructured response",
            'reasoning': response[:200] + "..." if len(response) > 200 else response,
            'output': "Extracted from unstructured LLM response",
            'confidence': 0.5,
            'dependencies': []
        }

async def generate_chain_of_thought(
    self, 
    question: str, 
    context: Optional[str] = None,
    max_steps: int = 5,
    custom_step_types: Optional[List[ReasoningStepType]] = None
) -> ChainOfThought:
    """Generate a complete chain of thought reasoning"""
    
    # Define default step sequence
    default_step_types = [
        ReasoningStepType.ANALYSIS,
        ReasoningStepType.HYPOTHESIS,
        ReasoningStepType.CALCULATION,
        ReasoningStepType.DEDUCTION,
        ReasoningStepType.CONCLUSION
    ]
    
    step_types = custom_step_types or default_step_types
    num_steps = min(max_steps, len(step_types))
    
    reasoning_steps = []
    
    # Generate each reasoning step
    for i in range(num_steps):
        try:
            step = await self.generate_reasoning_step(
                question=question,
                context=context,
                previous_steps=reasoning_steps,
                step_number=i + 1,
                step_type=step_types[i]
            )
            reasoning_steps.append(step)
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate step {i + 1}: {e}")
    
    # Generate final answer
    final_answer = await self._generate_final_answer(question, reasoning_steps)
    
    # Calculate overall confidence (minimum of all step confidences)
    overall_confidence = min(step.confidence for step in reasoning_steps) if reasoning_steps else 0.0
    
    # Create and return ChainOfThought
    try:
        return ChainOfThought(
            question=question,
            context=context,
            reasoning_steps=reasoning_steps,
            final_answer=final_answer,
            overall_confidence=overall_confidence,
            metadata={
                'model_used': self.llm.model_name,
                'steps_generated': len(reasoning_steps),
                'step_types': [step.step_type.value for step in reasoning_steps],
                'generation_timestamp': datetime.now().isoformat()
            }
        )
    except ValidationError as e:
        raise ValueError(f"Failed to create valid ChainOfThought: {e}")

async def _generate_final_answer(self, question: str, steps: List[ReasoningStep]) -> str:
    """Generate final answer based on reasoning steps"""
    if not steps:
        return "No reasoning steps provided to generate answer"
    
    # Build final answer synthesis prompt
    prompt_parts = [
        f"Question: {question}",
        "",
        "Reasoning chain completed:",
    ]
    
    for step in steps:
        prompt_parts.append(f"Step {step.step_number}: {step.output}")
    
    prompt_parts.extend([
        "",
        "Provide a concise final answer that synthesizes all reasoning steps:",
    ])
    
    # Simulate LLM call for final answer
    await asyncio.sleep(0.1)
    
    # Generate contextual final answer
    last_output = steps[-1].output
    return f"Based on the systematic reasoning chain, the final answer is: {last_output}. This conclusion is supported by the {len(steps)}-step analysis that progressed from initial problem breakdown through logical deduction."
```

# Enhanced example usage with error handling

async def example_usage():
“”“Comprehensive example of how to use the Chain of Thought system”””

```
# Initialize LLM interface
llm_config = LLMInterface(
    model_name="gpt-4",
    api_key="your-api-key-here",  # In practice, use environment variables
    max_tokens=1500,
    temperature=0.7
)

# Create reasoning engine
engine = ChainOfThoughtEngine(llm_config)

# Example questions with different complexities
examples = [
    {
        "question": "What is the area of a circle with radius 5 meters?",
        "context": "Use π ≈ 3.14159 for calculations. Round to 2 decimal places.",
        "max_steps": 4
    },
    {
        "question": "If I invest $1000 at 5% annual interest compounded monthly for 2 years, how much will I have?",
        "context": "Use the compound interest formula A = P(1 + r/n)^(nt)",
        "max_steps": 5
    }
]

for i, example in enumerate(examples, 1):
    print(f"\n{'='*60}")
    print(f"EXAMPLE {i}")
    print(f"{'='*60}")
    
    try:
        # Generate chain of thought
        cot = await engine.generate_chain_of_thought(**example)
        
        # Display results
        print(f"Question: {cot.question}")
        print(f"Context: {cot.context}")
        print(f"Overall Confidence: {cot.overall_confidence:.2f}")
        print(f"Generated: {cot.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\nReasoning Chain ({len(cot.reasoning_steps)} steps):")
        print("-" * 40)
        
        for step in cot.reasoning_steps:
            print(f"\n📍 Step {step.step_number}: {step.step_type.value.upper()}")
            print(f"   Description: {step.description}")
            print(f"   Reasoning: {step.reasoning}")
            print(f"   Output: {step.output}")
            print(f"   Confidence: {step.confidence:.2f}")
            if step.dependencies:
                print(f"   Depends on steps: {step.dependencies}")
        
        print(f"\n🎯 FINAL ANSWER:")
        print(f"   {cot.final_answer}")
        
        # Validation confirmation
        print(f"\n✅ Validation: Structure is valid")
        print(f"   - {len(cot.reasoning_steps)} sequential steps")
        print(f"   - All dependencies properly referenced")
        print(f"   - Confidence levels within bounds")
        
        # JSON export size
        json_size = len(cot.model_dump_json(indent=2))
        print(f"   - JSON export: {json_size:,} characters")
        
    except Exception as e:
        print(f"❌ Error processing example {i}: {e}")
        continue
```

# Utility functions

def validate_cot_structure(cot: ChainOfThought) -> Dict[str, bool]:
“”“Validate chain of thought structure comprehensively”””
validations = {
“sequential_steps”: True,
“valid_dependencies”: True,
“confidence_bounds”: True,
“non_empty_content”: True
}

```
try:
    # Check sequential numbering
    for i, step in enumerate(cot.reasoning_steps, 1):
        if step.step_number != i:
            validations["sequential_steps"] = False
            break
    
    # Check dependencies
    for step in cot.reasoning_steps:
        for dep in step.dependencies:
            if dep >= step.step_number:
                validations["valid_dependencies"] = False
                break
    
    # Check confidence bounds
    all_confidences = [step.confidence for step in cot.reasoning_steps] + [cot.overall_confidence]
    if not all(0.0 <= conf <= 1.0 for conf in all_confidences):
        validations["confidence_bounds"] = False
    
    # Check non-empty content
    if not all([
        cot.question.strip(),
        cot.final_answer.strip(),
        all(step.description.strip() and step.reasoning.strip() for step in cot.reasoning_steps)
    ]):
        validations["non_empty_content"] = False
        
except Exception:
    return {key: False for key in validations}

return validations
```

def run_example():
“”“Run the comprehensive example”””
print(“🧠 Chain of Thought Reasoning System”)
print(“Using Pydantic v2 with modern validation”)
asyncio.run(example_usage())

if **name** == “**main**”:
run_example()