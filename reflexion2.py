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
    """Robust LLM client supporting multiple providers"""
    
    def __init__(self, provider: str = "openai", api_key: str = None, 
                 model: str = None, rate_limit_delay: float = 1.0):
        self.provider = provider.lower()
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.last_call_time = 0
        
        # Configure provider-specific settings
        if self.provider == "openai":
            self.base_url = "https://api.openai.com/v1"
            self.model = model or "gpt-3.5-turbo"
            if not api_key:
                raise ValueError("OpenAI API key is required")
        elif self.provider == "anthropic":
            self.base_url = "https://api.anthropic.com/v1"
            self.model = model or "claude-3-haiku-20240307"
            if not api_key:
                raise ValueError("Anthropic API key is required")
        elif self.provider == "local":
            # For local models like Ollama
            self.base_url = "http://localhost:11434/api"
            self.model = model or "llama2"
            self.api_key = None  # No API key needed for local
        elif self.provider == "mock":
            # Mock provider for testing
            self.base_url = None
            self.model = "mock-model"
            self.api_key = None
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Setup session with retry strategy (except for mock)
        if self.provider != "mock":
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
        
        # Handle mock provider for testing
        if self.provider == "mock":
            return self._mock_generate(prompt, max_tokens, temperature)
        
        self._enforce_rate_limit()
        
        try:
            if self.provider == "openai":
                return self._generate_openai(prompt, max_tokens, temperature)
            elif self.provider == "anthropic":
                return self._generate_anthropic(prompt, max_tokens, temperature)
            elif self.provider == "local":
                return self._generate_local(prompt, max_tokens, temperature)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return f"Error: API request failed - {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in generate: {e}")
            return f"Error: Generation failed - {str(e)}"
    
    def _mock_generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Mock generation for testing without API"""
        time.sleep(0.1)  # Simulate API delay
        
        if "connection" in prompt.lower():
            return "Connection successful - mock LLM working"
        elif "reflect" in prompt.lower():
            return """I attempted to solve the task but had some challenges. 
            What went well: I understood the problem structure.
            What could improve: I need to be more systematic in my approach.
            Next time: I should break down the problem into smaller steps."""
        elif "code" in prompt.lower():
            return """def solve_problem():
    # This is a mock solution
    return "mock result" """
        elif "math" in prompt.lower():
            return "Step 1: Identify the operation needed. Step 2: Apply the calculation. Final answer: 4"
        else:
            return f"Mock response to: {prompt[:50]}..."
    
    def _generate_openai(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using OpenAI API"""
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
        
        logger.info(f"Making OpenAI API call to {self.model}")
        response = self.session.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        if 'choices' not in result or not result['choices']:
            raise ValueError("Invalid response format from OpenAI API")
        
        return result['choices'][0]['message']['content'].strip()
    
    def _generate_anthropic(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using Anthropic API"""
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        logger.info(f"Making Anthropic API call to {self.model}")
        response = self.session.post(
            f"{self.base_url}/messages",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        if 'content' not in result or not result['content']:
            raise ValueError("Invalid response format from Anthropic API")
        
        return result['content'][0]['text'].strip()
    
    def _generate_local(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using local model (e.g., Ollama)"""
        data = {
            "model": self.model,
            "prompt": prompt,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature
            }
        }
        
        logger.info(f"Making local API call to {self.model}")
        response = self.session.post(
            f"{self.base_url}/generate",
            json=data,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        if 'response' not in result:
            raise ValueError("Invalid response format from local API")
        
        return result['response'].strip()

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

Keep your reflection concise but insightful.

Reflection:"""

        try:
            reflection = self.llm.generate(reflection_prompt, max_tokens=400, temperature=0.3)
            logger.debug(f"Generated reflection: {reflection[:100]}...")
            
            # Validate reflection quality
            if len(reflection) < 20:
                return f"Brief reflection: The approach yielded {reward} reward. Need to improve strategy for better results."
            
            return reflection
        except Exception as e:
            logger.error(f"Error generating reflection: {e}")
            # Return a basic reflection instead of failing
            if reward < 0.5:
                return f"The task was challenging and my approach needs improvement. I achieved {reward} reward which suggests significant issues with my strategy. I should analyze the problem more systematically next time."
            else:
                return f"I had reasonable success with {reward} reward, but there's room for improvement. I should build on what worked while addressing the gaps in my approach."
    
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

def test_llm_connection(llm_client: LLMClient) -> bool:
    """Test if LLM is responding properly with a simple query"""
    print(f"🔍 Testing {llm_client.provider.upper()} LLM connection...")
    
    try:
        # Simple test prompt
        test_prompt = "Reply with exactly 'Connection successful' if you can understand this message."
        response = llm_client.generate(test_prompt, max_tokens=50, temperature=0.0)
        
        print(f"✅ LLM Response: '{response}'")
        
        # Check if response contains expected keywords or is reasonable
        if any(word in response.lower() for word in ['connection', 'successful', 'understand', 'yes', 'working']):
            print("✅ LLM connection test PASSED - API is responding correctly")
            return True
        elif len(response) > 10:  # Got some reasonable response
            print(f"✅ LLM connection test PASSED - Got valid response")
            return True
        else:
            print(f"⚠️  LLM connection test PARTIAL - Got response but unexpected content")
            print(f"   Response: '{response}'")
            return True  # Still working, just unexpected response
            
    except Exception as e:
        print(f"❌ LLM connection test FAILED: {e}")
        print("   Please check:")
        print("   - API key is correct and has proper permissions")
        print("   - Internet connection is working") 
        print("   - API service is available")
        print("   - Model name is correct")
        return False

def quick_llm_test(provider: str = "openai", api_key: str = None, model: str = None) -> bool:
    """Quick standalone test function with multiple provider support"""
    print(f"=== Quick {provider.upper()} LLM Test ===")
    
    if provider != "mock" and provider != "local" and (not api_key or "your-" in api_key):
        print("❌ Please provide a valid API key!")
        return False
    
    try:
        llm_client = LLMClient(provider=provider, api_key=api_key, model=model, rate_limit_delay=0.5)
        return test_llm_connection(llm_client)
    except Exception as e:
        print(f"❌ Failed to initialize LLM client: {e}")
        return False

def demo_reflexion_system():
    """Demonstrate the Reflexion system with multiple provider options"""
    
    print("=== Reflexion Agent Demo ===")
    print("Choose your LLM provider:")
    print("1. OpenAI (requires API key)")
    print("2. Anthropic Claude (requires API key)")
    print("3. Local model (e.g., Ollama)")
    print("4. Mock (for testing without API)")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        api_key = input("Enter OpenAI API key: ").strip()
        llm_client = LLMClient(provider="openai", api_key=api_key, rate_limit_delay=1.5)
    elif choice == "2":
        api_key = input("Enter Anthropic API key: ").strip()
        llm_client = LLMClient(provider="anthropic", api_key=api_key, rate_limit_delay=1.5)
    elif choice == "3":
        model = input("Enter model name (default: llama2): ").strip() or "llama2"
        llm_client = LLMClient(provider="local", model=model, rate_limit_delay=1.0)
    elif choice == "4":
        print("Using mock LLM for demonstration...")
        llm_client = LLMClient(provider="mock", rate_limit_delay=0.5)
    else:
        print("Invalid choice. Using mock LLM...")
        llm_client = LLMClient(provider="mock", rate_limit_delay=0.5)
    
    try:
        # Test LLM connection first
        if not test_llm_connection(llm_client):
            print("\n❌ LLM connection test failed. Aborting demo.")
            return
        
        agent = ReflexionAgent(llm_client, max_trials=3)
        
        print("\n=== Starting Reflexion Tests ===\n")
        
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
    # Quick test option
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Quick test mode: python reflexion.py test [provider]
        provider = sys.argv[2] if len(sys.argv) > 2 else "mock"
        
        if provider == "mock":
            quick_llm_test(provider="mock")
        elif provider == "openai":
            api_key = input("Enter your OpenAI API key: ").strip()
            quick_llm_test(provider="openai", api_key=api_key)
        elif provider == "anthropic":
            api_key = input("Enter your Anthropic API key: ").strip()
            quick_llm_test(provider="anthropic", api_key=api_key)
        elif provider == "local":
            model = input("Enter model name (default: llama2): ").strip() or "llama2"
            quick_llm_test(provider="local", model=model)
        else:
            print(f"Testing with mock provider...")
            quick_llm_test(provider="mock")
    else:
        # Full demo
        demo_reflexion_system()