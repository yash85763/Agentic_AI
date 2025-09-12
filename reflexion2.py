import json
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class Reflection:
“”“Stores a reflection about a failed attempt”””
task_type: str
problem: str
failed_attempt: str
error_analysis: str
improvement_strategy: str
timestamp: str

```
def to_dict(self):
    return asdict(self)
```

class ReflexionAgent:
“”“Simple implementation of Reflexion reasoning method”””

```
def __init__(self):
    self.memory: List[Reflection] = []
    self.max_attempts = 3

def solve_math_problem(self, problem: str) -> Tuple[str, bool, List[str]]:
    """
    Solve a math problem with reflexion
    Returns: (solution, success, reflections_used)
    """
    problem_hash = self._hash_problem(problem)
    relevant_reflections = self._retrieve_reflections("math", problem)
    
    for attempt in range(self.max_attempts):
        print(f"\n--- Attempt {attempt + 1} ---")
        
        # Generate solution using past reflections
        solution = self._generate_solution(problem, relevant_reflections)
        print(f"Solution: {solution}")
        
        # Evaluate the solution
        is_correct = self._evaluate_math_solution(problem, solution)
        print(f"Correct: {is_correct}")
        
        if is_correct:
            return solution, True, [r.improvement_strategy for r in relevant_reflections]
        
        # Generate reflection on failure
        reflection = self._generate_reflection(
            task_type="math",
            problem=problem,
            failed_attempt=solution,
            attempt_number=attempt + 1
        )
        
        self.memory.append(reflection)
        relevant_reflections.append(reflection)
        print(f"Reflection: {reflection.improvement_strategy}")
    
    return solution, False, [r.improvement_strategy for r in relevant_reflections]

def solve_coding_problem(self, problem: str) -> Tuple[str, bool, List[str]]:
    """
    Solve a coding problem with reflexion
    Returns: (code, success, reflections_used)
    """
    relevant_reflections = self._retrieve_reflections("coding", problem)
    
    for attempt in range(self.max_attempts):
        print(f"\n--- Attempt {attempt + 1} ---")
        
        # Generate code using past reflections
        code = self._generate_code(problem, relevant_reflections)
        print(f"Generated code:\n{code}")
        
        # Test the code
        is_correct = self._test_code(problem, code)
        print(f"Tests passed: {is_correct}")
        
        if is_correct:
            return code, True, [r.improvement_strategy for r in relevant_reflections]
        
        # Generate reflection on failure
        reflection = self._generate_reflection(
            task_type="coding",
            problem=problem,
            failed_attempt=code,
            attempt_number=attempt + 1
        )
        
        self.memory.append(reflection)
        relevant_reflections.append(reflection)
        print(f"Reflection: {reflection.improvement_strategy}")
    
    return code, False, [r.improvement_strategy for r in relevant_reflections]

def _hash_problem(self, problem: str) -> str:
    """Create a hash for the problem for similarity matching"""
    return hashlib.md5(problem.encode()).hexdigest()[:8]

def _retrieve_reflections(self, task_type: str, problem: str) -> List[Reflection]:
    """Retrieve relevant reflections from memory"""
    relevant = []
    problem_words = set(problem.lower().split())
    
    for reflection in self.memory:
        if reflection.task_type == task_type:
            reflection_words = set(reflection.problem.lower().split())
            # Simple similarity based on word overlap
            similarity = len(problem_words & reflection_words) / len(problem_words | reflection_words)
            if similarity > 0.3:  # Threshold for relevance
                relevant.append(reflection)
    
    return relevant

def _generate_solution(self, problem: str, reflections: List[Reflection]) -> str:
    """Generate a math solution considering past reflections"""
    # This is a simplified solver - in practice, you'd use a more sophisticated method
    
    if "factorial" in problem.lower():
        if any("recursive" in r.improvement_strategy for r in reflections):
            return self._solve_factorial_iterative(problem)
        else:
            return self._solve_factorial_basic(problem)
    
    elif "fibonacci" in problem.lower():
        if any("memoization" in r.improvement_strategy for r in reflections):
            return self._solve_fibonacci_memo(problem)
        else:
            return self._solve_fibonacci_basic(problem)
    
    else:
        return "Unable to solve this type of problem"

def _generate_code(self, problem: str, reflections: List[Reflection]) -> str:
    """Generate code considering past reflections"""
    
    if "sort" in problem.lower():
        if any("edge cases" in r.improvement_strategy for r in reflections):
            return self._generate_robust_sort()
        else:
            return self._generate_basic_sort()
    
    elif "palindrome" in problem.lower():
        if any("case insensitive" in r.improvement_strategy for r in reflections):
            return self._generate_robust_palindrome()
        else:
            return self._generate_basic_palindrome()
    
    else:
        return "# Unable to generate code for this problem"

def _solve_factorial_basic(self, problem: str) -> str:
    """Basic factorial solution (might have issues)"""
    return "n! = n * (n-1) * ... * 1"

def _solve_factorial_iterative(self, problem: str) -> str:
    """Improved factorial solution"""
    return "def factorial(n): result = 1; [result := result * i for i in range(1, n+1)]; return result"

def _solve_fibonacci_basic(self, problem: str) -> str:
    """Basic fibonacci solution"""
    return "F(n) = F(n-1) + F(n-2), F(0)=0, F(1)=1"

def _solve_fibonacci_memo(self, problem: str) -> str:
    """Fibonacci with memoization"""
    return "def fib(n, memo={}): return memo.get(n) or memo.setdefault(n, fib(n-1, memo) + fib(n-2, memo)) if n > 1 else n"

def _generate_basic_sort(self) -> str:
    """Basic sort implementation"""
    return """def sort_list(arr):
for i in range(len(arr)):
    for j in range(len(arr)-1):
        if arr[j] > arr[j+1]:
            arr[j], arr[j+1] = arr[j+1], arr[j]
return arr"""

def _generate_robust_sort(self) -> str:
    """Sort with edge case handling"""
    return """def sort_list(arr):
if not arr or len(arr) <= 1:
    return arr
arr = arr.copy()  # Don't modify original
for i in range(len(arr)):
    for j in range(len(arr)-1-i):
        if arr[j] > arr[j+1]:
            arr[j], arr[j+1] = arr[j+1], arr[j]
return arr"""

def _generate_basic_palindrome(self) -> str:
    """Basic palindrome check"""
    return """def is_palindrome(s):
return s == s[::-1]"""

def _generate_robust_palindrome(self) -> str:
    """Robust palindrome check"""
    return """def is_palindrome(s):
s = ''.join(c.lower() for c in s if c.isalnum())
return s == s[::-1]"""

def _evaluate_math_solution(self, problem: str, solution: str) -> bool:
    """Evaluate if math solution is correct (simplified)"""
    if "factorial" in problem.lower():
        return "def factorial" in solution or "result" in solution
    elif "fibonacci" in problem.lower():
        return "memo" in solution or "F(n-1)" in solution
    return False

def _test_code(self, problem: str, code: str) -> bool:
    """Test if generated code is correct (simplified)"""
    if "sort" in problem.lower():
        return "not arr or len(arr)" in code  # Check for edge cases
    elif "palindrome" in problem.lower():
        return "lower()" in code and "isalnum()" in code  # Check for robustness
    return False

def _generate_reflection(self, task_type: str, problem: str, 
                       failed_attempt: str, attempt_number: int) -> Reflection:
    """Generate a reflection on the failed attempt"""
    
    # Simplified reflection generation - in practice, this would be more sophisticated
    error_analysis = ""
    improvement_strategy = ""
    
    if task_type == "math":
        if "factorial" in problem.lower() and "def factorial" not in failed_attempt:
            error_analysis = "Solution was too abstract, needed actual implementation"
            improvement_strategy = "Provide concrete recursive or iterative implementation"
        elif "fibonacci" in problem.lower() and "memo" not in failed_attempt:
            error_analysis = "Basic solution may be inefficient for large numbers"
            improvement_strategy = "Use memoization to avoid redundant calculations"
    
    elif task_type == "coding":
        if "sort" in problem.lower() and "not arr" not in failed_attempt:
            error_analysis = "Code doesn't handle edge cases like empty arrays"
            improvement_strategy = "Add edge cases handling for empty and single-element arrays"
        elif "palindrome" in problem.lower() and "lower()" not in failed_attempt:
            error_analysis = "Code doesn't handle case sensitivity and non-alphanumeric characters"
            improvement_strategy = "Make case insensitive and filter non-alphanumeric characters"
    
    return Reflection(
        task_type=task_type,
        problem=problem,
        failed_attempt=failed_attempt,
        error_analysis=error_analysis,
        improvement_strategy=improvement_strategy,
        timestamp=datetime.now().isoformat()
    )

def get_memory_summary(self) -> Dict:
    """Get a summary of stored reflections"""
    summary = {
        "total_reflections": len(self.memory),
        "by_task_type": {},
        "recent_reflections": []
    }
    
    for reflection in self.memory:
        task_type = reflection.task_type
        summary["by_task_type"][task_type] = summary["by_task_type"].get(task_type, 0) + 1
    
    # Get last 3 reflections
    for reflection in self.memory[-3:]:
        summary["recent_reflections"].append({
            "task": reflection.task_type,
            "problem": reflection.problem[:50] + "..." if len(reflection.problem) > 50 else reflection.problem,
            "improvement": reflection.improvement_strategy
        })
    
    return summary
```

# Example usage and testing

def demo_reflexion_agent():
“”“Demonstrate the Reflexion agent”””
agent = ReflexionAgent()

```
print("=== Reflexion Reasoning Demo ===\n")

# Test math problem solving
print("1. Math Problem: Calculate factorial")
solution, success, reflections = agent.solve_math_problem("Calculate factorial of n")
print(f"Final success: {success}")

print("\n" + "="*50)

# Test coding problem solving
print("2. Coding Problem: Sort a list")
code, success, reflections = agent.solve_coding_problem("Write a function to sort a list")
print(f"Final success: {success}")

print("\n" + "="*50)

# Test another similar problem to see memory usage
print("3. Another sort problem (should use previous reflection)")
code, success, reflections = agent.solve_coding_problem("Create a sorting algorithm for arrays")
print(f"Final success: {success}")
print(f"Used {len(reflections)} previous reflections")

print("\n" + "="*50)

# Show memory summary
print("4. Memory Summary:")
summary = agent.get_memory_summary()
print(json.dumps(summary, indent=2))
```

if **name** == “**main**”:
demo_reflexion_agent()