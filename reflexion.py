from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
import json
from datetime import datetime

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI  # or your preferred LLM


class ReflexionStep(str, Enum):
    THINKING = "thinking"
    ACTING = "acting"
    REFLECTING = "reflecting"
    MEMORY_UPDATE = "memory_update"


class ActionType(str, Enum):
    SEARCH = "search"
    CALCULATE = "calculate"
    REASON = "reason"
    ANSWER = "answer"
    ASK_CLARIFICATION = "ask_clarification"


class Action(BaseModel):
    """Represents an action the agent can take"""
    type: ActionType
    content: str
    reasoning: str = Field(description="Why this action was chosen")


class ActionResult(BaseModel):
    """Result of executing an action"""
    action: Action
    result: str
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class Reflection(BaseModel):
    """Reflection on a failed attempt"""
    attempt_summary: str = Field(description="Summary of what was attempted")
    failure_analysis: str = Field(description="Analysis of why it failed")
    lessons_learned: str = Field(description="Key lessons from this failure")
    improvement_strategy: str = Field(description="How to improve next time")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in the reflection")


class Memory(BaseModel):
    """Long-term memory for storing experiences"""
    successful_strategies: List[str] = Field(default_factory=list)
    common_pitfalls: List[str] = Field(default_factory=list)
    domain_knowledge: Dict[str, Any] = Field(default_factory=dict)
    reflection_history: List[Reflection] = Field(default_factory=list)


class ReflexionState(BaseModel):
    """Current state of the Reflexion agent"""
    task: str
    current_step: ReflexionStep = ReflexionStep.THINKING
    attempt_count: int = 0
    max_attempts: int = 3
    
    # Current attempt state
    current_reasoning: str = ""
    planned_actions: List[Action] = Field(default_factory=list)
    executed_actions: List[ActionResult] = Field(default_factory=list)
    
    # Results and reflections
    final_answer: Optional[str] = None
    current_reflection: Optional[Reflection] = None
    is_complete: bool = False
    
    # Memory
    memory: Memory = Field(default_factory=Memory)


class ReflexionAgent:
    """
    Reflexion reasoning agent that can reflect on failures and improve
    """
    
    def __init__(self, llm: BaseLanguageModel, max_attempts: int = 3):
        self.llm = llm
        self.max_attempts = max_attempts
        
        # Initialize parsers
        self.action_parser = PydanticOutputParser(pydantic_object=Action)
        self.reflection_parser = PydanticOutputParser(pydantic_object=Reflection)
        
        # Initialize prompts
        self._setup_prompts()
    
    def _setup_prompts(self):
        """Setup prompt templates for different phases"""
        
        # Thinking phase prompt
        self.thinking_prompt = PromptTemplate(
            input_variables=["task", "memory", "attempt_count", "previous_attempts"],
            template="""
You are a reasoning agent using Reflexion methodology. Your task is to think through the problem and plan your approach.

TASK: {task}

ATTEMPT: {attempt_count}/{max_attempts}

MEMORY FROM PREVIOUS EXPERIENCES:
Successful strategies: {memory.successful_strategies}
Common pitfalls: {memory.common_pitfalls}
Previous reflections: {memory.reflection_history}

PREVIOUS ATTEMPTS (if any):
{previous_attempts}

Think step by step about how to approach this task. Consider:
1. What is the core problem?
2. What information do you need?
3. What are potential approaches?
4. What have you learned from previous attempts?
5. What strategy will you use?

Provide your reasoning:
""",
            partial_variables={"max_attempts": self.max_attempts}
        )
        
        # Action planning prompt
        self.action_prompt = PromptTemplate(
            input_variables=["task", "reasoning", "memory"],
            template="""
Based on your reasoning, plan the next action to take.

TASK: {task}
REASONING: {reasoning}
MEMORY: {memory}

Plan your next action. Consider what type of action would be most effective.

{format_instructions}
""",
            partial_variables={"format_instructions": self.action_parser.get_format_instructions()}
        )
        
        # Reflection prompt
        self.reflection_prompt = PromptTemplate(
            input_variables=["task", "attempt_summary", "actions_taken", "final_result"],
            template="""
You need to reflect on a failed attempt at solving a task.

TASK: {task}

ATTEMPT SUMMARY: {attempt_summary}

ACTIONS TAKEN:
{actions_taken}

FINAL RESULT: {final_result}

Reflect deeply on what went wrong and how to improve. Be specific and actionable.

{format_instructions}
""",
            partial_variables={"format_instructions": self.reflection_parser.get_format_instructions()}
        )
    
    def solve(self, task: str, context: Dict[str, Any] = None) -> ReflexionState:
        """
        Main method to solve a task using Reflexion reasoning
        """
        state = ReflexionState(task=task, max_attempts=self.max_attempts)
        
        while not state.is_complete and state.attempt_count < state.max_attempts:
            state.attempt_count += 1
            print(f"\n--- ATTEMPT {state.attempt_count} ---")
            
            # Thinking phase
            state.current_step = ReflexionStep.THINKING
            self._thinking_phase(state)
            
            # Acting phase
            state.current_step = ReflexionStep.ACTING
            self._acting_phase(state, context)
            
            # Check if we have a satisfactory answer
            if self._evaluate_attempt(state):
                state.is_complete = True
                print("✅ Task completed successfully!")
                break
            
            # Reflecting phase (if not the last attempt)
            if state.attempt_count < state.max_attempts:
                state.current_step = ReflexionStep.REFLECTING
                self._reflecting_phase(state)
                
                # Memory update phase
                state.current_step = ReflexionStep.MEMORY_UPDATE
                self._update_memory(state)
        
        if not state.is_complete:
            print(f"❌ Task not completed after {state.max_attempts} attempts")
        
        return state
    
    def _thinking_phase(self, state: ReflexionState):
        """Think through the problem and develop reasoning"""
        print("🤔 THINKING...")
        
        # Prepare previous attempts summary
        previous_attempts = ""
        if state.memory.reflection_history:
            previous_attempts = "\n".join([
                f"Attempt {i+1}: {reflection.attempt_summary} -> {reflection.failure_analysis}"
                for i, reflection in enumerate(state.memory.reflection_history)
            ])
        
        prompt = self.thinking_prompt.format(
            task=state.task,
            memory=state.memory,
            attempt_count=state.attempt_count,
            previous_attempts=previous_attempts or "None"
        )
        
        response = self.llm.invoke(prompt)
        state.current_reasoning = response.content
        print(f"Reasoning: {state.current_reasoning[:200]}...")
    
    def _acting_phase(self, state: ReflexionState, context: Dict[str, Any] = None):
        """Execute actions based on reasoning"""
        print("🎯 ACTING...")
        
        # Plan and execute multiple actions
        for step in range(3):  # Allow up to 3 actions per attempt
            # Plan next action
            action_prompt = self.action_prompt.format(
                task=state.task,
                reasoning=state.current_reasoning,
                memory=json.dumps(state.memory.dict(), indent=2)
            )
            
            try:
                action_response = self.llm.invoke(action_prompt)
                action = self.action_parser.parse(action_response.content)
                state.planned_actions.append(action)
                
                # Execute action
                result = self._execute_action(action, context)
                state.executed_actions.append(result)
                
                print(f"Action: {action.type} -> {result.result[:100]}...")
                
                # Check if this is a final answer
                if action.type == ActionType.ANSWER:
                    state.final_answer = result.result
                    break
                    
            except Exception as e:
                print(f"Error in action phase: {e}")
                break
    
    def _execute_action(self, action: Action, context: Dict[str, Any] = None) -> ActionResult:
        """Execute a specific action"""
        try:
            if action.type == ActionType.SEARCH:
                # Simulate search (in real implementation, this would call actual search)
                result = f"Search results for: {action.content}"
                
            elif action.type == ActionType.CALCULATE:
                # Simple calculation execution (extend as needed)
                try:
                    result = str(eval(action.content))  # Warning: unsafe for production
                except:
                    result = "Calculation error"
                    
            elif action.type == ActionType.REASON:
                # Use LLM for reasoning
                reasoning_prompt = f"Reason about: {action.content}"
                response = self.llm.invoke(reasoning_prompt)
                result = response.content
                
            elif action.type == ActionType.ANSWER:
                result = action.content
                
            else:
                result = f"Executed {action.type}: {action.content}"
            
            return ActionResult(
                action=action,
                result=result,
                success=True
            )
            
        except Exception as e:
            return ActionResult(
                action=action,
                result="",
                success=False,
                error_message=str(e)
            )
    
    def _evaluate_attempt(self, state: ReflexionState) -> bool:
        """Evaluate if the current attempt is satisfactory"""
        # Simple evaluation - in practice, this could be more sophisticated
        if state.final_answer:
            # You could add more sophisticated evaluation here
            # For example, checking against ground truth, using another LLM to evaluate, etc.
            return len(state.final_answer.strip()) > 10  # Simple length check
        return False
    
    def _reflecting_phase(self, state: ReflexionState):
        """Reflect on the failed attempt"""
        print("🔍 REFLECTING...")
        
        attempt_summary = f"Attempt {state.attempt_count}: {state.current_reasoning[:100]}..."
        actions_taken = "\n".join([
            f"{i+1}. {result.action.type}: {result.action.content} -> {result.result[:50]}..."
            for i, result in enumerate(state.executed_actions)
        ])
        
        final_result = state.final_answer or "No final answer provided"
        
        reflection_prompt = self.reflection_prompt.format(
            task=state.task,
            attempt_summary=attempt_summary,
            actions_taken=actions_taken,
            final_result=final_result
        )
        
        try:
            reflection_response = self.llm.invoke(reflection_prompt)
            reflection = self.reflection_parser.parse(reflection_response.content)
            state.current_reflection = reflection
            print(f"Reflection: {reflection.lessons_learned[:100]}...")
        except Exception as e:
            print(f"Error in reflection: {e}")
    
    def _update_memory(self, state: ReflexionState):
        """Update long-term memory with learnings"""
        print("💾 UPDATING MEMORY...")
        
        if state.current_reflection:
            # Add reflection to history
            state.memory.reflection_history.append(state.current_reflection)
            
            # Extract learnings for future use
            if state.current_reflection.improvement_strategy:
                state.memory.successful_strategies.append(
                    state.current_reflection.improvement_strategy
                )
            
            if state.current_reflection.failure_analysis:
                state.memory.common_pitfalls.append(
                    state.current_reflection.failure_analysis
                )
        
        # Reset current attempt state for next iteration
        state.planned_actions = []
        state.executed_actions = []
        state.current_reflection = None
        state.final_answer = None


# Example usage
def main():
    """Example of how to use the Reflexion agent"""
    
    # Initialize LLM (replace with your preferred model)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    
    # Create Reflexion agent
    agent = ReflexionAgent(llm, max_attempts=3)
    
    # Example task
    task = "Solve this math word problem: If a train travels 120 miles in 2 hours, and then travels 180 miles in 3 hours, what is the average speed for the entire journey?"
    
    # Solve the task
    result = agent.solve(task)
    
    # Print results
    print(f"\n=== FINAL RESULTS ===")
    print(f"Task: {result.task}")
    print(f"Attempts made: {result.attempt_count}")
    print(f"Completed: {result.is_complete}")
    print(f"Final answer: {result.final_answer}")
    
    if result.memory.reflection_history:
        print(f"\nReflections made: {len(result.memory.reflection_history)}")
        for i, reflection in enumerate(result.memory.reflection_history):
            print(f"Reflection {i+1}: {reflection.lessons_learned}")


if __name__ == "__main__":
    main()