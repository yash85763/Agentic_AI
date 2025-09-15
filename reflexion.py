from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, ValidationError
from enum import Enum
import json
import re
from datetime import datetime

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI  # or your preferred LLM


class ReflexionStep(str, Enum):
    THINKING = "thinking"
    ACTING = "acting"
    EVALUATING = "evaluating"
    REFLECTING = "reflecting"
    MEMORY_UPDATE = "memory_update"


class ActionType(str, Enum):
    SEARCH = "search"
    CALCULATE = "calculate"
    REASON = "reason"
    ANSWER = "answer"
    ASK_CLARIFICATION = "ask_clarification"


class Action(BaseModel):
    """Represents an action the actor can take"""
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


class EvaluationResult(BaseModel):
    """Result from the evaluator LLM"""
    is_correct: bool = Field(description="Whether the answer is correct")
    score: float = Field(ge=0.0, le=1.0, description="Quality score of the answer")
    feedback: str = Field(description="Detailed feedback on the answer")
    reasoning: str = Field(description="Evaluator's reasoning process")
    areas_for_improvement: List[str] = Field(default_factory=list)


class Reflection(BaseModel):
    """Reflection from the self-reflection LLM"""
    attempt_summary: str = Field(description="Summary of what was attempted")
    failure_analysis: str = Field(description="Analysis of why it failed")
    lessons_learned: str = Field(description="Key lessons from this failure")
    improvement_strategy: str = Field(description="How to improve next time")
    specific_mistakes: List[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in the reflection")


class Memory(BaseModel):
    """Long-term memory for storing experiences"""
    successful_strategies: List[str] = Field(default_factory=list)
    common_pitfalls: List[str] = Field(default_factory=list)
    domain_knowledge: Dict[str, Any] = Field(default_factory=dict)
    reflection_history: List[Reflection] = Field(default_factory=list)
    evaluation_history: List[EvaluationResult] = Field(default_factory=list)


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
    
    # Results and evaluations
    final_answer: Optional[str] = None
    current_evaluation: Optional[EvaluationResult] = None
    current_reflection: Optional[Reflection] = None
    is_complete: bool = False
    
    # Memory
    memory: Memory = Field(default_factory=Memory)


class BaseLLM:
    """Base class for LLM components with robust parsing"""
    
    def __init__(self, llm: BaseLanguageModel, temperature: float = 0.1):
        self.llm = llm
        self.temperature = temperature
    
    def _extract_json_from_response(self, response_text: str) -> Optional[Dict]:
        """Extract JSON from LLM response, handling various formats"""
        try:
            # First, try to parse the entire response as JSON
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Look for JSON blocks in the response
        json_patterns = [
            r'```json\s*(\{.*?\})\s*```',  # JSON in code blocks
            r'```\s*(\{.*?\})\s*```',      # JSON in generic code blocks
            r'(\{[^{}]*"[^"]*"[^{}]*\})',  # Simple JSON objects
            r'(\{.*\})'                    # Any JSON-like structure
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _safe_parse_with_fallback(self, response_text: str, parser: PydanticOutputParser, fallback_creator):
        """Safely parse LLM response with fallback options"""
        try:
            # First, try normal parsing
            return parser.parse(response_text)
        except Exception as e:
            print(f"Warning: Normal parsing failed: {e}")
            
            # Try to extract JSON from the response
            json_data = self._extract_json_from_response(response_text)
            if json_data:
                try:
                    # Try to create the Pydantic model from the JSON
                    return parser.pydantic_object(**json_data)
                except Exception as e2:
                    print(f"Warning: JSON parsing failed: {e2}")
            
            # Create fallback object
            print("Using fallback parsing...")
            return fallback_creator(response_text)
    
    def _create_action_fallback(self, response_text: str) -> Action:
        """Create fallback Action from raw response"""
        # Try to extract action type from text
        action_type = ActionType.REASON  # Default
        if "search" in response_text.lower():
            action_type = ActionType.SEARCH
        elif "calculate" in response_text.lower() or any(op in response_text for op in ['+', '-', '*', '/', '=']):
            action_type = ActionType.CALCULATE
        elif "answer" in response_text.lower() or "final" in response_text.lower():
            action_type = ActionType.ANSWER
        
        return Action(
            type=action_type,
            content=response_text[:500],  # Truncate to reasonable length
            reasoning="Fallback parsing - extracted from raw response"
        )
    
    def _create_evaluation_fallback(self, response_text: str) -> EvaluationResult:
        """Create fallback EvaluationResult from raw response"""
        # Simple heuristics to determine correctness
        positive_indicators = ['correct', 'good', 'right', 'accurate', 'well', 'excellent']
        negative_indicators = ['wrong', 'incorrect', 'bad', 'error', 'mistake', 'fail']
        
        text_lower = response_text.lower()
        positive_count = sum(1 for word in positive_indicators if word in text_lower)
        negative_count = sum(1 for word in negative_indicators if word in text_lower)
        
        is_correct = positive_count > negative_count
        score = 0.7 if is_correct else 0.3
        
        return EvaluationResult(
            is_correct=is_correct,
            score=score,
            feedback=response_text[:300],  # Use response as feedback
            reasoning="Fallback evaluation based on text analysis",
            areas_for_improvement=["Unable to parse detailed evaluation"]
        )
    
    def _create_reflection_fallback(self, response_text: str) -> Reflection:
        """Create fallback Reflection from raw response"""
        return Reflection(
            attempt_summary="Parsing failed - using raw response",
            failure_analysis=response_text[:200],
            lessons_learned="Need to improve response parsing and formatting",
            improvement_strategy="Ensure responses follow the expected JSON format",
            specific_mistakes=["Response format not parseable"],
            confidence_score=0.3
        )


class ActorLLM(BaseLLM):
    """Actor LLM - responsible for reasoning and taking actions"""
    
    def __init__(self, llm: BaseLanguageModel, temperature: float = 0.7):
        super().__init__(llm, temperature)
        self.action_parser = PydanticOutputParser(pydantic_object=Action)
        self._setup_prompts()
    
    def _setup_prompts(self):
        self.thinking_prompt = PromptTemplate(
            input_variables=["task", "memory", "attempt_count", "previous_attempts"],
            template="""
You are an Actor agent in a Reflexion system. Your job is to think through problems and take actions to solve them.

TASK: {task}

ATTEMPT: {attempt_count}/3

MEMORY FROM PREVIOUS EXPERIENCES:
Successful strategies: {successful_strategies}
Common pitfalls to avoid: {common_pitfalls}

PREVIOUS ATTEMPTS AND REFLECTIONS:
{previous_attempts}

Think step by step about how to approach this task. Consider:
1. What is the core problem?
2. What information do you need?
3. What approaches have worked before?
4. What mistakes should you avoid based on previous attempts?
5. What is your strategy for this attempt?

Provide your detailed reasoning and strategy:
""")
        
        self.action_prompt = PromptTemplate(
            input_variables=["task", "reasoning", "previous_actions", "context"],
            template="""
You are an Actor agent. Based on your reasoning, determine the next action to take.

TASK: {task}
YOUR REASONING: {reasoning}

PREVIOUS ACTIONS IN THIS ATTEMPT:
{previous_actions}

CURRENT CONTEXT: {context}

Determine the most appropriate next action. Be specific and purposeful.

{format_instructions}
""",
            partial_variables={"format_instructions": self.action_parser.get_format_instructions()}
        )
    
    def think(self, task: str, memory: Memory, attempt_count: int) -> str:
        """Generate reasoning for the current attempt"""
        # Prepare previous attempts summary
        previous_attempts = ""
        if memory.reflection_history:
            previous_attempts = "\n".join([
                f"Attempt {i+1}: {reflection.attempt_summary}\n"
                f"  Failed because: {reflection.failure_analysis}\n"
                f"  Lesson learned: {reflection.lessons_learned}\n"
                f"  Improvement strategy: {reflection.improvement_strategy}\n"
                for i, reflection in enumerate(memory.reflection_history[-2:])  # Last 2 attempts
            ])
        
        prompt = self.thinking_prompt.format(
            task=task,
            memory=memory,
            attempt_count=attempt_count,
            successful_strategies="; ".join(memory.successful_strategies) or "None yet",
            common_pitfalls="; ".join(memory.common_pitfalls) or "None identified",
            previous_attempts=previous_attempts or "This is the first attempt"
        )
        
        response = self.llm.invoke(prompt)
        return response.content
    
    def plan_action(self, task: str, reasoning: str, previous_actions: List[ActionResult], context: str = "") -> Action:
        """Plan the next action to take"""
        previous_actions_text = "\n".join([
            f"{i+1}. {result.action.type}: {result.action.content} -> Result: {result.result[:100]}..."
            for i, result in enumerate(previous_actions)
        ]) if previous_actions else "No previous actions in this attempt"
        
        prompt = self.action_prompt.format(
            task=task,
            reasoning=reasoning,
            previous_actions=previous_actions_text,
            context=context
        )
        
        response = self.llm.invoke(prompt)
        
        # Use safe parsing with fallback
        return self._safe_parse_with_fallback(
            response.content, 
            self.action_parser, 
            self._create_action_fallback
        )


class EvaluatorLLM(BaseLLM):
    """Evaluator LLM - responsible for evaluating the quality of answers"""
    
    def __init__(self, llm: BaseLanguageModel, temperature: float = 0.1):
        super().__init__(llm, temperature)
        self.evaluation_parser = PydanticOutputParser(pydantic_object=EvaluationResult)
        self._setup_prompts()
    
    def _setup_prompts(self):
        self.evaluation_prompt = PromptTemplate(
            input_variables=["task", "answer", "actions_taken", "ground_truth"],
            template="""
You are an Evaluator agent in a Reflexion system. Your job is to critically evaluate the quality and correctness of answers.

ORIGINAL TASK: {task}

PROPOSED ANSWER: {answer}

ACTIONS TAKEN TO REACH THIS ANSWER:
{actions_taken}

GROUND TRUTH (if available): {ground_truth}

Evaluate this answer comprehensively. Consider:
1. Correctness: Is the answer factually correct?
2. Completeness: Does it fully address the task?
3. Reasoning quality: Is the logic sound?
4. Clarity: Is it well-explained?
5. Efficiency: Could it have been reached more directly?

Be strict but fair in your evaluation. Identify specific areas for improvement.

{format_instructions}
""",
            partial_variables={"format_instructions": self.evaluation_parser.get_format_instructions()}
        )
    
    def evaluate(self, task: str, answer: str, actions_taken: List[ActionResult], ground_truth: str = None) -> EvaluationResult:
        """Evaluate the quality of an answer"""
        actions_text = "\n".join([
            f"{i+1}. {result.action.type}: {result.action.content}\n   Result: {result.result[:150]}..."
            for i, result in enumerate(actions_taken)
        ]) if actions_taken else "No actions taken"
        
        prompt = self.evaluation_prompt.format(
            task=task,
            answer=answer or "No answer provided",
            actions_taken=actions_text,
            ground_truth=ground_truth or "Not provided"
        )
        
        response = self.llm.invoke(prompt)
        
        # Use safe parsing with fallback
        return self._safe_parse_with_fallback(
            response.content,
            self.evaluation_parser,
            self._create_evaluation_fallback
        )


class SelfReflectionLLM(BaseLLM):
    """Self-Reflection LLM - responsible for analyzing failures and generating insights"""
    
    def __init__(self, llm: BaseLanguageModel, temperature: float = 0.3):
        super().__init__(llm, temperature)
        self.reflection_parser = PydanticOutputParser(pydantic_object=Reflection)
        self._setup_prompts()
    
    def _setup_prompts(self):
        self.reflection_prompt = PromptTemplate(
            input_variables=["task", "reasoning", "actions_taken", "answer", "evaluation"],
            template="""
You are a Self-Reflection agent in a Reflexion system. Your job is to deeply analyze failed attempts and extract actionable insights.

ORIGINAL TASK: {task}

REASONING USED: {reasoning}

ACTIONS TAKEN:
{actions_taken}

FINAL ANSWER: {answer}

EVALUATOR'S ASSESSMENT:
{evaluation}

Conduct a thorough post-mortem analysis. Focus on:
1. What exactly went wrong and why?
2. What were the key decision points where things could have gone differently?
3. What specific mistakes were made in reasoning or execution?
4. What patterns of error should be avoided in future attempts?
5. What concrete strategies would improve performance next time?

Be brutally honest and specific. The goal is maximum learning from this failure.

{format_instructions}
""",
            partial_variables={"format_instructions": self.reflection_parser.get_format_instructions()}
        )
    
    def reflect(self, task: str, reasoning: str, actions_taken: List[ActionResult], 
                answer: str, evaluation: EvaluationResult) -> Reflection:
        """Generate reflection on a failed attempt"""
        actions_text = "\n".join([
            f"{i+1}. {result.action.type}: {result.action.content}\n"
            f"   Reasoning: {result.action.reasoning}\n"
            f"   Result: {result.result[:150]}...\n"
            f"   Success: {result.success}\n"
            for i, result in enumerate(actions_taken)
        ]) if actions_taken else "No actions were taken"
        
        evaluation_text = (
            f"Correct: {evaluation.is_correct}\n"
            f"Score: {evaluation.score}\n"
            f"Feedback: {evaluation.feedback}\n"
            f"Areas for improvement: {', '.join(evaluation.areas_for_improvement)}"
        )
        
        prompt = self.reflection_prompt.format(
            task=task,
            reasoning=reasoning,
            actions_taken=actions_text,
            answer=answer or "No final answer provided",
            evaluation=evaluation_text
        )
        
        response = self.llm.invoke(prompt)
        
        # Use safe parsing with fallback
        return self._safe_parse_with_fallback(
            response.content,
            self.reflection_parser,
            self._create_reflection_fallback
        )


class ReflexionAgent:
    """
    Multi-LLM Reflexion reasoning agent with separate Actor, Evaluator, and Self-Reflection LLMs
    """
    
    def __init__(self, 
                 actor_llm: BaseLanguageModel,
                 evaluator_llm: BaseLanguageModel, 
                 reflection_llm: BaseLanguageModel,
                 max_attempts: int = 3):
        
        # Initialize the three specialized LLMs
        self.actor = ActorLLM(actor_llm, temperature=0.7)  # More creative for problem-solving
        self.evaluator = EvaluatorLLM(evaluator_llm, temperature=0.1)  # More deterministic for evaluation
        self.reflector = SelfReflectionLLM(reflection_llm, temperature=0.3)  # Balanced for analysis
        
        self.max_attempts = max_attempts
    
    def solve(self, task: str, ground_truth: str = None, context: Dict[str, Any] = None) -> ReflexionState:
        """
        Solve a task using the three-LLM Reflexion approach
        """
        state = ReflexionState(task=task, max_attempts=self.max_attempts)
        
        while not state.is_complete and state.attempt_count < state.max_attempts:
            state.attempt_count += 1
            print(f"\n{'='*50}")
            print(f"ATTEMPT {state.attempt_count}/{state.max_attempts}")
            print(f"{'='*50}")
            
            # ACTOR: Thinking phase
            print("🤔 ACTOR: Thinking and reasoning...")
            state.current_step = ReflexionStep.THINKING
            state.current_reasoning = self.actor.think(
                task=state.task, 
                memory=state.memory, 
                attempt_count=state.attempt_count
            )
            print(f"Reasoning: {state.current_reasoning[:200]}...")
            
            # ACTOR: Acting phase
            print("\n🎯 ACTOR: Taking actions...")
            state.current_step = ReflexionStep.ACTING
            self._acting_phase(state, context)
            
            # EVALUATOR: Evaluation phase
            print("\n📊 EVALUATOR: Evaluating the attempt...")
            state.current_step = ReflexionStep.EVALUATING
            if state.final_answer:
                state.current_evaluation = self.evaluator.evaluate(
                    task=state.task,
                    answer=state.final_answer,
                    actions_taken=state.executed_actions,
                    ground_truth=ground_truth
                )
                
                # Safe access to evaluation result
                if state.current_evaluation:
                    print(f"Evaluation: {'✅ CORRECT' if state.current_evaluation.is_correct else '❌ INCORRECT'} "
                          f"(Score: {state.current_evaluation.score:.2f})")
                    print(f"Feedback: {state.current_evaluation.feedback[:150]}...")
                    
                    # Check if satisfactory
                    if state.current_evaluation.is_correct or state.current_evaluation.score >= 0.8:
                        state.is_complete = True
                        print("🎉 Task completed successfully!")
                        break
                else:
                    print("❌ Evaluation failed - will continue to next attempt")
            else:
                print("⚠️ No final answer provided - creating default evaluation")
                state.current_evaluation = EvaluationResult(
                    is_correct=False,
                    score=0.0,
                    feedback="No final answer was provided",
                    reasoning="Missing final answer",
                    areas_for_improvement=["Provide a final answer"]
                )
            
            # SELF-REFLECTION: Reflect on failure (if not the last attempt)
            if state.attempt_count < state.max_attempts and not state.is_complete:
                print("\n🔍 REFLECTOR: Analyzing the failure...")
                state.current_step = ReflexionStep.REFLECTING
                
                # Ensure we have an evaluation to reflect on
                if not state.current_evaluation:
                    state.current_evaluation = EvaluationResult(
                        is_correct=False,
                        score=0.0,
                        feedback="No evaluation available",
                        reasoning="Missing evaluation",
                        areas_for_improvement=["Generate proper evaluation"]
                    )
                
                state.current_reflection = self.reflector.reflect(
                    task=state.task,
                    reasoning=state.current_reasoning,
                    actions_taken=state.executed_actions,
                    answer=state.final_answer or "",
                    evaluation=state.current_evaluation
                )
                
                if state.current_reflection:
                    print(f"Key insight: {state.current_reflection.lessons_learned[:150]}...")
                else:
                    print("⚠️ Reflection generation failed")
                
                # Update memory
                state.current_step = ReflexionStep.MEMORY_UPDATE
                self._update_memory(state)
                
                # Reset for next attempt
                self._reset_attempt_state(state)
        
        if not state.is_complete:
            print(f"\n❌ Task not completed successfully after {state.max_attempts} attempts")
        
        return state
    
    def _acting_phase(self, state: ReflexionState, context: Dict[str, Any] = None):
        """Execute actions using the Actor LLM"""
        context_str = json.dumps(context or {}, indent=2)
        
        # Allow up to 5 actions per attempt
        for step in range(5):
            try:
                # Actor plans the next action
                action = self.actor.plan_action(
                    task=state.task,
                    reasoning=state.current_reasoning,
                    previous_actions=state.executed_actions,
                    context=context_str
                )
                state.planned_actions.append(action)
                
                # Execute the action
                result = self._execute_action(action, context)
                state.executed_actions.append(result)
                
                print(f"  Action {step+1}: {action.type} -> {result.result[:100]}...")
                
                # Check if this is a final answer
                if action.type == ActionType.ANSWER:
                    state.final_answer = result.result
                    break
                    
            except Exception as e:
                print(f"  Error in action {step+1}: {e}")
                break
    
    def _execute_action(self, action: Action, context: Dict[str, Any] = None) -> ActionResult:
        """Execute a specific action (same as before but with better error handling)"""
        try:
            if action.type == ActionType.SEARCH:
                # In a real implementation, this would call actual search APIs
                result = f"Search results for '{action.content}': [Simulated search results would appear here]"
                
            elif action.type == ActionType.CALCULATE:
                try:
                    # Safe evaluation for simple math
                    import re
                    if re.match(r'^[0-9+\-*/().\s]+$', action.content):
                        result = str(eval(action.content))
                    else:
                        result = f"Cannot safely calculate: {action.content}"
                except:
                    result = f"Calculation error for: {action.content}"
                    
            elif action.type == ActionType.REASON:
                # Use actor LLM for additional reasoning
                reasoning_prompt = f"Think step by step about: {action.content}"
                response = self.actor.llm.invoke(reasoning_prompt)
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
    
    def _update_memory(self, state: ReflexionState):
        """Update long-term memory with learnings"""
        if state.current_reflection:
            # Add reflection to history
            state.memory.reflection_history.append(state.current_reflection)
            
            # Extract learnings for future use
            if state.current_reflection.improvement_strategy:
                state.memory.successful_strategies.append(
                    state.current_reflection.improvement_strategy
                )
            
            # Add specific mistakes to common pitfalls
            state.memory.common_pitfalls.extend(state.current_reflection.specific_mistakes)
        
        if state.current_evaluation:
            state.memory.evaluation_history.append(state.current_evaluation)
    
    def _reset_attempt_state(self, state: ReflexionState):
        """Reset state for next attempt"""
        state.planned_actions = []
        state.executed_actions = []
        state.current_reflection = None
        state.current_evaluation = None
        state.final_answer = None


# Example usage demonstrating the three-LLM architecture
def main():
    """Example usage with separate LLMs for each component"""
    
    # Initialize three separate LLMs (you could use different models for each)
    actor_llm = ChatOpenAI(model="gpt-4", temperature=0.7)       # Creative problem solver
    evaluator_llm = ChatOpenAI(model="gpt-4", temperature=0.1)   # Strict evaluator
    reflection_llm = ChatOpenAI(model="gpt-4", temperature=0.3)  # Analytical reflector
    
    # Create Reflexion agent with three specialized LLMs
    agent = ReflexionAgent(
        actor_llm=actor_llm,
        evaluator_llm=evaluator_llm, 
        reflection_llm=reflection_llm,
        max_attempts=3
    )
    
    # Example task with ground truth for evaluation
    task = """
    Solve this step by step: 
    A train leaves Station A at 2:00 PM traveling at 60 mph toward Station B. 
    Another train leaves Station B at 2:30 PM traveling at 80 mph toward Station A. 
    If the stations are 350 miles apart, at what time will the trains meet?
    """
    
    ground_truth = "The trains will meet at 4:30 PM"
    
    # Solve the task
    result = agent.solve(task, ground_truth=ground_truth)
    
    # Print comprehensive results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Task: {result.task}")
    print(f"Attempts made: {result.attempt_count}")
    print(f"Completed successfully: {result.is_complete}")
    print(f"Final answer: {result.final_answer}")
    
    if result.memory.reflection_history:
        print(f"\nReflections generated: {len(result.memory.reflection_history)}")
        for i, reflection in enumerate(result.memory.reflection_history):
            print(f"\nReflection {i+1}:")
            print(f"  Failure analysis: {reflection.failure_analysis}")
            print(f"  Lesson learned: {reflection.lessons_learned}")
            print(f"  Improvement strategy: {reflection.improvement_strategy}")
    
    if result.memory.evaluation_history:
        print(f"\nEvaluations performed: {len(result.memory.evaluation_history)}")
        for i, eval_result in enumerate(result.memory.evaluation_history):
            print(f"  Evaluation {i+1}: {'Correct' if eval_result.is_correct else 'Incorrect'} "
                  f"(Score: {eval_result.score:.2f})")


if __name__ == "__main__":
    main()