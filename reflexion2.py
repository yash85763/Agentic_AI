import json
import time
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from abc import ABC, abstractmethod
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemoryItem:
    """Single memory item storing trajectory and reflection"""
    task: str
    trajectory: List[str]  # Sequence of actions/thoughts
    reward: float  # Success/failure score
    reflection: str  # Verbal reflection on the trajectory
    timestamp: str
    task_hash: str
    
    def to_dict(self):
        return asdict(self)

class LLMClient:
    """Robust LLM client with retry logic and rate limiting"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", 
                 model: str = "gpt-3.5-turbo", rate_limit_delay: float = 1.0):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.rate_limit_delay = rate_limit_delay
        self.last_call_time = 0
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def _enforce_rate_limit(self):
        """Ensure minimum delay between API calls"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_call
            logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Generate text using LLM with proper error handling"""
        self._enforce_rate_limit()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            logger.info(f"Making API call to {self.model}")
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            if 'choices' not in result or not result['choices']:
                raise ValueError("Invalid response format from API")
            
            generated_text = result['choices'][0]['message']['content'].strip()
            logger.info(f"Successfully generated {len(generated_text)} characters")
            return generated_text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise Exception(f"LLM API call failed: {e}")
        except (KeyError, ValueError) as e:
            logger.error(f"Response parsing failed: {e}")
            raise Exception(f"Failed to parse LLM response: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise Exception(f"Unexpected LLM error: {e}")

class TaskEnvironment(ABC):
    """Abstract base class for task environments"""
    
    @abstractmethod
    def reset(self, task: str) -> str:
        """Reset environment with new task, return initial observation"""
        pass
    
    @abstractmethod
    def step(self, action: str) -> Tuple[str, float, bool]:
        """Execute action, return (observation, reward, done)"""
        pass
    
    @abstractmethod
    def get_task_description(self) -> str:
        """Get description of the current task"""
        pass

class MathEnvironment(TaskEnvironment):
    """Environment for mathematical reasoning tasks"""
    
    def __init__(self):
        self.task = ""
        self.steps = []
        self.max_steps = 5
        self.correct_answer = None
    
    def reset(self, task: str) -> str:
        self.task = task
        self.steps = []
        self.correct_answer = self._get_correct_answer(task)
        return f"Solve: {task}\nYou can reason step by step. Start with your first step."
    
    def step(self, action: str) -> Tuple[str, float, bool]:
        self.steps.append(action)
        
        if len(self.steps) >= self.max_steps:
            reward = self._evaluate_solution()
            return "Max steps reached. Provide final answer.", reward, True
        
        if "final answer:" in action.lower():
            reward = self._evaluate_solution()
            return "Solution complete.", reward, True
        
        return "Continue with next step.", 0.0, False
    
    def get_task_description(self) -> str:
        return self.task
    
    def _get_correct_answer(self, task: str) -> Any:
        """Get correct answer for evaluation (simplified)"""
        if "2 + 2" in task:
            return "4"
        elif "factorial of 5" in task:
            return "120"
        elif "fibonacci" in task and "10" in task:
            return "55"
        return None
    
    def _evaluate_solution(self) -> float:
        """Evaluate the solution quality"""
        if not self.steps:
            return 0.0
        
        final_step = self.steps[-1].lower()
        
        if self.correct_answer and str(self.correct_answer) in final_step:
            return 1.0
        
        # Partial credit for reasoning quality
        if any(word in final_step for word in ["step", "calculate", "multiply", "add"]):
            return 0.3
        
        return 0.0

class CodeEnvironment(TaskEnvironment):
    """Environment for code generation tasks"""
    
    def __init__(self):
        self.task = ""
        self.code_attempts = []
        self.max_attempts = 3
    
    def reset(self, task: str) -> str:
        self.task = task
        self.code_attempts = []
        return f"Task: {task}\nWrite code to solve this problem."
    
    def step(self, action: str) -> Tuple[str, float, bool]:
        self.code_attempts.append(action)
        
        if len(self.code_attempts) >= self.max_attempts:
            reward = self._evaluate_code()
            return "Max attempts reached.", reward, True
        
        if "def " in action or "class " in action:
            reward = self._evaluate_code()
            if reward > 0.5:
                return "Code looks good!", reward, True
            else:
                return "Code has issues. Try again.", reward, False
        
        return "Please provide complete code.", 0.0, False
    
    def get_task_description(self) -> str:
        return self.task
    
    def _evaluate_code(self) -> float:
        """Evaluate code quality (simplified)"""
        if not self.code_attempts:
            return 0.0
        
        latest_code = self.code_attempts[-1].lower()
        
        # Check for basic code structure
        if "def " not in latest_code:
            return 0.1
        
        # Task-specific evaluation
        if "palindrome" in self.task.lower():
            if "def is_palindrome" in latest_code and "return" in latest_code:
                return 1.0 if "[::-1]" in latest_code else 0.7
        
        elif "fibonacci" in self.task.lower():
            if "def fibonacci" in latest_code or "def fib" in latest_code:
                return 1.0 if ("memo" in latest_code or "cache" in latest_code) else 0.7
        
        return 0.5  # Basic code structure present

class ReflexionAgent:
    """
    Reflexion agent implementation based on the paper:
    "Reflexion: Language Agents with Verbal Reinforcement Learning"
    """
    
    def __init__(self, llm_client: LLMClient, max_trials: int = 3):
        self.llm = llm_client
        self.max_trials = max_trials
        self.memory: List[MemoryItem] = []
        self.episodic_memory_size = 100  # Limit memory size
    
    def solve_task(self, task: str, environment: TaskEnvironment) -> Tuple[List[str], float, str]:
        """
        Main Reflexion loop: Act -> Evaluate -> Reflect
        Returns: (trajectory, final_reward, final_reflection)
        """
        logger.info(f"Starting Reflexion for task: {task[:50]}...")
        
        task_hash = hashlib.md5(task.encode()).hexdigest()[:10]
        best_trajectory = []
        best_reward = 0.0
        final_reflection = ""
        
        for trial in range(self.max_trials):
            logger.info(f"Trial {trial + 1}/{self.max_trials}")
            
            try:
                # Get relevant memories
                relevant_memories = self._retrieve_memories(task, k=3)
                
                # Generate trajectory using Actor
                trajectory, reward = self._act(task, environment, relevant_memories, trial)
                
                if reward > best_reward:
                    best_trajectory = trajectory
                    best_reward = reward
                
                # Generate reflection using Self-Reflection
                reflection = self._reflect(task, trajectory, reward)
                final_reflection = reflection
                
                # Store in episodic memory
                memory_item = MemoryItem(
                    task=task,
                    trajectory=trajectory,
                    reward=reward,
                    reflection=reflection,
                    timestamp=datetime.now().isoformat(),
                    task_hash=task_hash
                )
                self._store_memory(memory_item)
                
                # Early stopping if perfect score
                if reward >= 1.0:
                    logger.info(f"Perfect score achieved on trial {trial + 1}")
                    break
                    
            except Exception as e:
                logger.error(f"Error in trial {trial + 1}: {e}")
                continue
        
        logger.info(f"Reflexion complete. Best reward: {best_reward}")
        return best_trajectory, best_reward, final_reflection
    
    def _act(self, task: str, environment: TaskEnvironment, 
             relevant_memories: List[MemoryItem], trial: int) -> Tuple[List[str], float]:
        """Actor: Generate action trajectory"""
        
        # Reset environment
        observation = environment.reset(task)
        trajectory = []
        done = False
        
        # Build context with relevant memories
        context = self._build_context(task, relevant_memories, trial)
        
        step_count = 0
        max_steps = 10
        
        while not done and step_count < max_steps:
            try:
                # Generate action using LLM
                action_prompt = f"""{context}

Current situation: {observation}
Previous actions: {' -> '.join(trajectory[-3:]) if trajectory else 'None'}

Generate the next action or reasoning step to solve this task.
Action:"""

                action = self.llm.generate(action_prompt, max_tokens=300, temperature=0.7)
                trajectory.append(action)
                
                # Execute action in environment
                observation, reward, done = environment.step(action)
                step_count += 1
                
                logger.debug(f"Step {step_count}: {action[:50]}... -> Reward: {reward}")
                
            except Exception as e:
                logger.error(f"Error generating action: {e}")
                action = "I need to think more carefully about this problem."
                trajectory.append(action)
                observation, reward, done = environment.step(action)
                step_count += 1
        
        return trajectory, reward
    
    def _reflect(self, task: str, trajectory: List[str], reward: float) -> str:
        """Self-Reflection: Generate verbal reflection on the trajectory"""
        
        trajectory_str = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(trajectory)])
        
        reflection_prompt = f"""You are an AI agent that just attempted to solve a task. Reflect on your performance.

Task: {task}

Your trajectory:
{trajectory_str}

Final reward: {reward}/1.0

Provide a detailed reflection on:
1. What you did well
2. What mistakes you made
3. What you would do differently next time
4. Key insights for similar tasks

Reflection:"""

        try:
            reflection = self.llm.generate(reflection_prompt, max_tokens=400, temperature=0.3)
            logger.debug(f"Generated reflection: {reflection[:100]}...")
            return reflection
        except Exception as e:
            logger.error(f"Error generating reflection: {e}")
            return f"Failed to generate reflection due to error: {e}"
    
    def _build_context(self, task: str, relevant_memories: List[MemoryItem], trial: int) -> str:
        """Build context for the Actor using relevant memories"""
        
        context = f"Task: {task}\n\nYou are an AI agent solving this task."
        
        if trial > 0:
            context += f" This is attempt {trial + 1}."
        
        if relevant_memories:
            context += "\n\nLearnings from similar past tasks:"
            for i, memory in enumerate(relevant_memories[:2]):  # Limit to avoid token overflow
                context += f"\n\nPast task: {memory.task[:100]}..."
                context += f"\nReward achieved: {memory.reward}"
                context += f"\nKey reflection: {memory.reflection[:200]}..."
        
        context += "\n\nNow solve the current task step by step."
        return context
    
    def _retrieve_memories(self, task: str, k: int = 3) -> List[MemoryItem]:
        """Retrieve k most relevant memories using simple similarity"""
        
        if not self.memory:
            return []
        
        task_words = set(task.lower().split())
        scored_memories = []
        
        for memory in self.memory:
            memory_words = set(memory.task.lower().split())
            similarity = len(task_words & memory_words) / len(task_words | memory_words)
            scored_memories.append((similarity, memory))
        
        # Sort by similarity and return top k
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored_memories[:k]]
    
    def _store_memory(self, memory_item: MemoryItem):
        """Store memory item with size limit"""
        self.memory.append(memory_item)
        
        # Maintain memory size limit
        if len(self.memory) > self.episodic_memory_size:
            # Remove oldest memories
            self.memory = self.memory[-self.episodic_memory_size:]
    
    def get_memory_stats(self) -> Dict:
        """Get statistics about stored memories"""
        if not self.memory:
            return {"total_memories": 0}
        
        rewards = [m.reward for m in self.memory]
        return {
            "total_memories": len(self.memory),
            "avg_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "recent_performance": sum(rewards[-5:]) / min(5, len(rewards))
        }

def demo_reflexion_system():
    """Demonstrate the Reflexion system"""
    
    # Note: Replace with your actual API key
    API_KEY = "your-openai-api-key-here"
    
    try:
        # Initialize components
        llm_client = LLMClient(api_key=API_KEY, rate_limit_delay=1.5)
        agent = ReflexionAgent(llm_client, max_trials=3)
        
        print("=== Reflexion Agent Demo ===\n")
        
        # Test 1: Math problem
        print("1. Testing Math Problem:")
        math_env = MathEnvironment()
        math_task = "What is 2 + 2?"
        
        trajectory, reward, reflection = agent.solve_task(math_task, math_env)
        
        print(f"Task: {math_task}")
        print(f"Final Reward: {reward}")
        print(f"Trajectory Length: {len(trajectory)}")
        print(f"Reflection: {reflection[:200]}...")
        
        print("\n" + "="*60 + "\n")
        
        # Test 2: Code generation
        print("2. Testing Code Generation:")
        code_env = CodeEnvironment()
        code_task = "Write a function to check if a string is a palindrome"
        
        trajectory, reward, reflection = agent.solve_task(code_task, code_env)
        
        print(f"Task: {code_task}")
        print(f"Final Reward: {reward}")
        print(f"Trajectory Length: {len(trajectory)}")
        print(f"Reflection: {reflection[:200]}...")
        
        print("\n" + "="*60 + "\n")
        
        # Test 3: Similar task to see memory usage
        print("3. Testing Similar Task (should use memory):")
        similar_task = "Create a palindrome checker function"
        
        trajectory, reward, reflection = agent.solve_task(similar_task, code_env)
        
        print(f"Task: {similar_task}")
        print(f"Final Reward: {reward}")
        print(f"Used Memory: {len(agent._retrieve_memories(similar_task))} relevant memories")
        
        print("\n" + "="*60 + "\n")
        
        # Show memory statistics
        print("4. Memory Statistics:")
        stats = agent.get_memory_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"Demo failed: {e}")
        print("\nNote: Make sure to set a valid OpenAI API key!")

if __name__ == "__main__":
    demo_reflexion_system()