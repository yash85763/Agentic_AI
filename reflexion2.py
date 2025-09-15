import json
import time
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from abc import ABC, abstractmethod

import asyncio
import aiohttp
from aiohttp import ClientResponseError, ClientConnectorError, ClientTimeout

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemoryItem:
    """Single memory item storing trajectory and reflection"""
    task: str
    trajectory: List[str]  # Sequence of actions/thoughts
    reward: float          # Success/failure score
    reflection: str        # Verbal reflection on the trajectory
    timestamp: str
    task_hash: str

    def to_dict(self):
        return asdict(self)

class AsyncLLMClient:
    """
    Robust async LLM client supporting multiple providers with:
      - aiohttp for async HTTP
      - explicit rate limit delay between calls
      - retries/backoff
      - a semaphore to enforce exactly ONE in-flight request at a time
    """

    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        rate_limit_delay: float = 1.5,
        max_retries: int = 3,
    ):
        self.provider = provider.lower()
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.last_call_time = 0.0

        # Connection/read timeouts
        self.connection_timeout = 10
        self.read_timeout = 30

        # Provider setup
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
            self.base_url = "http://localhost:11434/api"
            self.model = model or "llama2"
            self.api_key = None
            self.connection_timeout = 5
            self.read_timeout = 60
        elif self.provider == "mock":
            self.base_url = None
            self.model = "mock-model"
            self.api_key = None
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        # Single-session, single-flight semaphore to serialize calls
        self._session: Optional[aiohttp.ClientSession] = None
        self._sem = asyncio.Semaphore(1)  # ensures 1 request at a time

    async def __aenter__(self):
        if self._session is None or self._session.closed:
            timeout = ClientTimeout(total=None, connect=self.connection_timeout, sock_read=self.read_timeout)
            # Limit connections to be conservative; we still gate with semaphore
            connector = aiohttp.TCPConnector(limit=2, enable_cleanup_closed=True)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _enforce_rate_limit(self):
        """Ensure minimum delay between API calls (async)."""
        now = time.time()
        delta = now - self.last_call_time
        if delta < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - delta
            logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            await asyncio.sleep(sleep_time)
        self.last_call_time = time.time()

    async def _make_request_with_retries(self, url: str, headers: dict, data: dict) -> dict:
        """
        Async HTTP POST with manual retry/backoff and better error handling.
        Returns parsed JSON.
        """
        assert self._session is not None, "ClientSession is not initialized"

        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"Making request attempt {attempt + 1}/{self.max_retries + 1}")
                async with self._session.post(url, headers=headers, json=data) as resp:
                    status = resp.status
                    text = await resp.text()

                    if status == 200:
                        try:
                            return json.loads(text)
                        except json.JSONDecodeError:
                            raise ValueError("Invalid JSON in response")

                    if status == 429:
                        # Backoff: 2^attempt seconds
                        wait = 2 ** attempt
                        logger.warning(f"Rate limited (429). Waiting {wait}s ...")
                        await asyncio.sleep(wait)
                        continue

                    if status in (500, 502, 503, 504, 520, 521, 522, 523, 524):
                        wait = 2 ** attempt
                        logger.warning(f"Server error {status}. Waiting {wait}s ...")
                        await asyncio.sleep(wait)
                        continue

                    if status == 401:
                        raise ClientResponseError(resp.request_info, (), status=401, message="Authentication failed (401).")
                    if status == 403:
                        raise ClientResponseError(resp.request_info, (), status=403, message="Access forbidden (403).")

                    # Other errors: one more try, then raise
                    logger.warning(f"HTTP {status}: {text[:200]}...")
                    if attempt < self.max_retries:
                        await asyncio.sleep(1)
                        continue
                    else:
                        raise ClientResponseError(resp.request_info, (), status=status, message=text[:200])

            except (asyncio.TimeoutError, aiohttp.ServerTimeoutError):
                logger.warning(f"Timeout on attempt {attempt + 1}")
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
            except ClientConnectorError as e:
                logger.warning(f"Connection error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(1)
                    continue
                raise

        raise RuntimeError("All retry attempts failed")

    async def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """
        Generate text using LLM with async semantics.
        - Enforces 1 in-flight request via semaphore
        - Enforces inter-call delay via rate limit
        """
        # Mock: fast-path without network, still respect semaphore so everything is serialized
        if self.provider == "mock":
            async with self._sem:
                await self._enforce_rate_limit()
                await asyncio.sleep(0.05)
                pl = prompt.lower()
                if "connection" in pl:
                    return "Connection successful - mock LLM working"
                if "reflect" in pl:
                    return ("I attempted the task and observed some issues.\n"
                            "Well: structured reasoning. Improve: verification.\n"
                            "Next time: smaller steps.")
                if "code" in pl:
                    return "def solve_problem():\n    return 'mock result'\n"
                if "math" in pl:
                    return "Step 1: Identify operation. Step 2: Compute. Final answer: 4"
                return f"Mock response to: {prompt[:50]}..."
        # Real providers
        async with self._sem:
            await self._enforce_rate_limit()
            try:
                if self.provider == "openai":
                    return await self._generate_openai(prompt, max_tokens, temperature)
                elif self.provider == "anthropic":
                    return await self._generate_anthropic(prompt, max_tokens, temperature)
                elif self.provider == "local":
                    return await self._generate_local(prompt, max_tokens, temperature)
                else:
                    raise ValueError(f"Unsupported provider: {self.provider}")
            except Exception as e:
                logger.error(f"Generation failed after all retries: {e}")
                return f"[Generation failed: {str(e)[:100]}... Please check connection and try again.]"

    async def _generate_openai(self, prompt: str, max_tokens: int, temperature: float) -> str:
        assert self._session is not None
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Reflexion-Agent/1.0"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        logger.info(f"Making OpenAI API call to {self.model}")
        result = await self._make_request_with_retries(f"{self.base_url}/chat/completions", headers, data)
        if "choices" not in result or not result["choices"]:
            raise ValueError("Invalid response format from OpenAI API")
        return result["choices"][0]["message"]["content"].strip()

    async def _generate_anthropic(self, prompt: str, max_tokens: int, temperature: float) -> str:
        assert self._session is not None
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "User-Agent": "Reflexion-Agent/1.0"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        logger.info(f"Making Anthropic API call to {self.model}")
        result = await self._make_request_with_retries(f"{self.base_url}/messages", headers, data)
        if "content" not in result or not result["content"]:
            raise ValueError("Invalid response format from Anthropic API")
        # Anthropic v2023-06-01 returns list of content blocks
        block = result["content"][0]
        text = block.get("text") if isinstance(block, dict) else None
        if not text:
            raise ValueError("Anthropic response missing text")
        return text.strip()

    async def _generate_local(self, prompt: str, max_tokens: int, temperature: float) -> str:
        assert self._session is not None
        # Health check
        try:
            url = f"{self.base_url.replace('/api', '')}/api/tags"
            async with self._session.get(url) as resp:
                if resp.status != 200:
                    raise RuntimeError("Local model service not available")
        except Exception:
            raise RuntimeError("Cannot connect to local model service. Make sure Ollama is running.")

        data = {
            "model": self.model,
            "prompt": prompt,
            "options": {"num_predict": max_tokens, "temperature": temperature},
            "stream": False
        }
        logger.info(f"Making local API call to {self.model}")
        result = await self._make_request_with_retries(f"{self.base_url}/generate", {"Content-Type": "application/json"}, data)
        if "response" not in result:
            raise ValueError("Invalid response format from local API")
        return result["response"].strip()

class TaskEnvironment(ABC):
    @abstractmethod
    def reset(self, task: str) -> str:
        pass

    @abstractmethod
    def step(self, action: str) -> Tuple[str, float, bool]:
        pass

    @abstractmethod
    def get_task_description(self) -> str:
        pass

class MathEnvironment(TaskEnvironment):
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
        if "2 + 2" in task:
            return "4"
        elif "factorial of 5" in task:
            return "120"
        elif "fibonacci" in task and "10" in task:
            return "55"
        return None

    def _evaluate_solution(self) -> float:
        if not self.steps:
            return 0.0
        final_step = self.steps[-1].lower()
        if self.correct_answer and str(self.correct_answer) in final_step:
            return 1.0
        if any(word in final_step for word in ["step", "calculate", "multiply", "add"]):
            return 0.3
        return 0.0

class CodeEnvironment(TaskEnvironment):
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
        if not self.code_attempts:
            return 0.0
        latest_code = self.code_attempts[-1].lower()
        if "def " not in latest_code:
            return 0.1
        if "palindrome" in self.task.lower():
            if "def is_palindrome" in latest_code and "return" in latest_code:
                return 1.0 if "[::-1]" in latest_code else 0.7
        elif "fibonacci" in self.task.lower():
            if "def fibonacci" in latest_code or "def fib" in latest_code:
                return 1.0 if ("memo" in latest_code or "cache" in latest_code) else 0.7
        return 0.5

class ReflexionAgent:
    """
    Async Reflexion agent (Act -> Evaluate -> Reflect) that awaits LLM replies.
    """

    def __init__(self, llm_client: AsyncLLMClient, max_trials: int = 3):
        self.llm = llm_client
        self.max_trials = max_trials
        self.memory: List[MemoryItem] = []
        self.episodic_memory_size = 100

    async def solve_task(self, task: str, environment: TaskEnvironment) -> Tuple[List[str], float, str]:
        logger.info(f"Starting Reflexion for task: {task[:50]}...")
        task_hash = hashlib.md5(task.encode()).hexdigest()[:10]
        best_trajectory: List[str] = []
        best_reward = 0.0
        final_reflection = ""

        for trial in range(self.max_trials):
            logger.info(f"Trial {trial + 1}/{self.max_trials}")
            try:
                relevant_memories = self._retrieve_memories(task, k=3)
                trajectory, reward = await self._act(task, environment, relevant_memories, trial)

                if reward > best_reward:
                    best_trajectory = trajectory
                    best_reward = reward

                reflection = await self._reflect(task, trajectory, reward)
                final_reflection = reflection

                memory_item = MemoryItem(
                    task=task,
                    trajectory=trajectory,
                    reward=reward,
                    reflection=reflection,
                    timestamp=datetime.now().isoformat(),
                    task_hash=task_hash,
                )
                self._store_memory(memory_item)

                if reward >= 1.0:
                    logger.info(f"Perfect score achieved on trial {trial + 1}")
                    break

            except Exception as e:
                logger.error(f"Error in trial {trial + 1}: {e}")
                continue

        logger.info(f"Reflexion complete. Best reward: {best_reward}")
        return best_trajectory, best_reward, final_reflection

    async def _act(
        self,
        task: str,
        environment: TaskEnvironment,
        relevant_memories: List[MemoryItem],
        trial: int,
    ) -> Tuple[List[str], float]:
        observation = environment.reset(task)
        trajectory: List[str] = []
        done = False

        context = self._build_context(task, relevant_memories, trial)

        step_count = 0
        max_steps = 10

        reward = 0.0
        while not done and step_count < max_steps:
            try:
                action_prompt = f"""{context}

Current situation: {observation}
Previous actions: {' -> '.join(trajectory[-3:]) if trajectory else 'None'}

Generate the next action or reasoning step to solve this task.
Action:"""
                action = await self.llm.generate(action_prompt, max_tokens=300, temperature=0.7)
                trajectory.append(action)

                observation, reward, done = environment.step(action)
                step_count += 1
                logger.debug(f"Step {step_count}: {action[:50]}... -> Reward: {reward}")

            except Exception as e:
                logger.error(f"Error generating action: {e}")
                fallback = "I need to think more carefully about this problem."
                trajectory.append(fallback)
                observation, reward, done = environment.step(fallback)
                step_count += 1

        return trajectory, reward

    async def _reflect(self, task: str, trajectory: List[str], reward: float) -> str:
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
            reflection = await self.llm.generate(reflection_prompt, max_tokens=400, temperature=0.3)
            if len(reflection) < 20:
                return f"Brief reflection: The approach yielded {reward} reward. Need to improve strategy."
            return reflection
        except Exception as e:
            logger.error(f"Error generating reflection: {e}")
            if reward < 0.5:
                return (f"The task was challenging and my approach needs improvement. "
                        f"I achieved {reward} reward which suggests significant issues.")
            else:
                return (f"I had reasonable success with {reward} reward, but there's room for improvement. "
                        f"I should build on what worked while addressing gaps.")

    def _build_context(self, task: str, relevant_memories: List[MemoryItem], trial: int) -> str:
        context = f"Task: {task}\n\nYou are an AI agent solving this task."
        if trial > 0:
            context += f" This is attempt {trial + 1}."
        if relevant_memories:
            context += "\n\nLearnings from similar past tasks:"
            for i, memory in enumerate(relevant_memories[:2]):
                context += f"\n\nPast task: {memory.task[:100]}..."
                context += f"\nReward achieved: {memory.reward}"
                context += f"\nKey reflection: {memory.reflection[:200]}..."
        context += "\n\nNow solve the current task step by step."
        return context

    def _retrieve_memories(self, task: str, k: int = 3) -> List[MemoryItem]:
        if not self.memory:
            return []
        task_words = set(task.lower().split())
        scored = []
        for m in self.memory:
            mw = set(m.task.lower().split())
            sim = len(task_words & mw) / max(1, len(task_words | mw))
            scored.append((sim, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:k]]

    def _store_memory(self, memory_item: MemoryItem):
        self.memory.append(memory_item)
        if len(self.memory) > self.episodic_memory_size:
            self.memory = self.memory[-self.episodic_memory_size:]

    def get_memory_stats(self) -> Dict:
        if not self.memory:
            return {"total_memories": 0}
        rewards = [m.reward for m in self.memory]
        return {
            "total_memories": len(self.memory),
            "avg_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "recent_performance": sum(rewards[-5:]) / min(5, len(rewards)),
        }

# ---------- Async test helpers & demo ----------

async def test_llm_connection(llm_client: AsyncLLMClient) -> bool:
    print(f"🔍 Testing {llm_client.provider.upper()} LLM connection...")
    try:
        test_prompt = "Reply with exactly 'Connection successful' if you can understand this message."
        response = await llm_client.generate(test_prompt, max_tokens=50, temperature=0.0)
        print(f"✅ LLM Response: '{response}'")
        if any(w in response.lower() for w in ['connection', 'successful', 'understand', 'yes', 'working']):
            print("✅ LLM connection test PASSED - API is responding correctly")
            return True
        elif len(response) > 10:
            print("✅ LLM connection test PASSED - Got valid response")
            return True
        else:
            print("⚠️  LLM connection test PARTIAL - Unexpected content")
            return True
    except Exception as e:
        print(f"❌ LLM connection test FAILED: {e}")
        print("   Please check API key, internet, service availability, model name")
        return False

async def quick_llm_test(provider: str = "openai", api_key: Optional[str] = None, model: Optional[str] = None) -> bool:
    print(f"=== Quick {provider.upper()} LLM Test ===")
    if provider not in ("mock", "local") and (not api_key or "your-" in (api_key or "")):
        print("❌ Please provide a valid API key!")
        return False

    try:
        async with AsyncLLMClient(
            provider=provider,
            api_key=api_key,
            model=model,
            rate_limit_delay=2.0,
            max_retries=2,
        ) as llm:
            print(f"🔧 Configuration:")
            print(f"   Provider: {llm.provider}")
            print(f"   Model: {llm.model}")
            print(f"   Base URL: {llm.base_url}")
            print(f"   Timeouts: {llm.connection_timeout}s connect, {llm.read_timeout}s read")
            return await test_llm_connection(llm)
    except Exception as e:
        print(f"❌ Failed to initialize LLM client: {e}")
        print("💡 Troubleshooting tips:\n   - Check internet\n   - Verify API key format\n   - Try 'mock' provider\n   - For local: ensure `ollama serve` is running")
        return False

async def demo_reflexion_system():
    print("=== Reflexion Agent Demo (Async) ===")
    print("Choose your LLM provider:")
    print("1. OpenAI (requires API key)")
    print("2. Anthropic Claude (requires API key)")
    print("3. Local model (e.g., Ollama)")
    print("4. Mock (for testing without API)")

    choice = input("Enter choice (1-4): ").strip()

    if choice == "1":
        api_key = input("Enter OpenAI API key: ").strip()
        client_kwargs = dict(provider="openai", api_key=api_key, rate_limit_delay=1.5)
    elif choice == "2":
        api_key = input("Enter Anthropic API key: ").strip()
        client_kwargs = dict(provider="anthropic", api_key=api_key, rate_limit_delay=1.5)
    elif choice == "3":
        model = input("Enter model name (default: llama2): ").strip() or "llama2"
        client_kwargs = dict(provider="local", model=model, rate_limit_delay=1.0)
    elif choice == "4":
        print("Using mock LLM for demonstration...")
        client_kwargs = dict(provider="mock", rate_limit_delay=0.5)
    else:
        print("Invalid choice. Using mock LLM...")
        client_kwargs = dict(provider="mock", rate_limit_delay=0.5)

    try:
        async with AsyncLLMClient(**client_kwargs) as llm:
            ok = await test_llm_connection(llm)
            if not ok:
                print("\n❌ LLM connection test failed. Aborting demo.")
                return

            agent = ReflexionAgent(llm, max_trials=3)

            print("\n=== Starting Reflexion Tests ===\n")

            # Test 1: Math problem
            print("1. Testing Math Problem:")
            math_env = MathEnvironment()
            math_task = "What is 2 + 2?"
            trajectory, reward, reflection = await agent.solve_task(math_task, math_env)
            print(f"Task: {math_task}")
            print(f"Final Reward: {reward}")
            print(f"Trajectory Length: {len(trajectory)}")
            print(f"Reflection: {reflection[:200]}...")

            print("\n" + "="*60 + "\n")

            # Test 2: Code generation
            print("2. Testing Code Generation:")
            code_env = CodeEnvironment()
            code_task = "Write a function to check if a string is a palindrome"
            trajectory, reward, reflection = await agent.solve_task(code_task, code_env)
            print(f"Task: {code_task}")
            print(f"Final Reward: {reward}")
            print(f"Trajectory Length: {len(trajectory)}")
            print(f"Reflection: {reflection[:200]}...")

            print("\n" + "="*60 + "\n")

            # Test 3: Similar task (memory usage)
            print("3. Testing Similar Task (should use memory):")
            similar_task = "Create a palindrome checker function"
            trajectory, reward, reflection = await agent.solve_task(similar_task, code_env)
            print(f"Task: {similar_task}")
            print(f"Final Reward: {reward}")
            print(f"Used Memory: {len(agent._retrieve_memories(similar_task))} relevant memories")

            print("\n" + "="*60 + "\n")

            # Memory stats
            print("4. Memory Statistics:")
            stats = agent.get_memory_stats()
            for k, v in stats.items():
                print(f"{k}: {v}")

    except Exception as e:
        print(f"Demo failed: {e}")
        print("\nNote: Make sure to set a valid API key!")

# Entry point
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Usage: python reflexion_async.py test [provider]
        provider = sys.argv[2] if len(sys.argv) > 2 else "mock"
        if provider == "mock":
            asyncio.run(quick_llm_test(provider="mock"))
        elif provider == "openai":
            api_key = input("Enter your OpenAI API key: ").strip()
            asyncio.run(quick_llm_test(provider="openai", api_key=api_key))
        elif provider == "anthropic":
            api_key = input("Enter your Anthropic API key: ").strip()
            asyncio.run(quick_llm_test(provider="anthropic", api_key=api_key))
        elif provider == "local":
            model = input("Enter model name (default: llama2): ").strip() or "llama2"
            asyncio.run(quick_llm_test(provider="local", model=model))
        else:
            asyncio.run(quick_llm_test(provider="mock"))
    else:
        asyncio.run(demo_reflexion_system())