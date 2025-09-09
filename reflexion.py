"""
reflexion_dspy.py

A production-grade Reflexion pattern implemented with DSPy.

- Solver: produces an initial answer.
- Critic: analyzes the answer and returns specific critique + fix suggestions.
- Reviser: improves the answer using accumulated critiques (self-reflection).
- Judge (pluggable): scores an answer for early stopping / success criteria.

Key features:
- Strong typing & docstrings
- Structured logging
- Config via dataclasses
- Safe defaults (low temperature, max rounds, early stop)
- Pluggable judge (LM-based by default; you can inject your own function)
- Minimal, reproducible example in __main__

Author: You
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Dict, Any

import backoff
import dspy

# --------------------------- Logging Setup ---------------------------

def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("reflexion")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

log = _setup_logger()


# --------------------------- Config ---------------------------

@dataclass(frozen=True)
class ReflexionConfig:
    """Configuration for the Reflexion agent."""
    max_rounds: int = 3
    success_threshold: float = 0.85
    # Model config
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 800
    # Judge settings
    judge_weighting: Dict[str, float] = field(default_factory=lambda: {
        "correctness": 0.5,
        "relevance":   0.2,
        "completeness":0.2,
        "clarity":     0.1,
    })
    # Critique accumulation
    max_critique_history: int = 3


# --------------------------- DSPy Signatures ---------------------------

class SolveTask(dspy.Signature):
    """Solve a user question carefully and succinctly."""
    question: str = dspy.InputField(desc="The user's question.")
    answer: str = dspy.OutputField(desc="Final answer, concise and factual.")


class CritiqueAttempt(dspy.Signature):
    """Provide a targeted critique of an answer."""
    question: str = dspy.InputField(desc="Original question.")
    answer: str = dspy.InputField(desc="The answer to critique.")
    critique: str = dspy.OutputField(
        desc=("Specific, actionable critique. Identify factual errors, "
              "missing pieces, irrelevant parts, ambiguity. Provide concrete fix suggestions.")
    )


class ReviseAnswer(dspy.Signature):
    """Revise an answer using critiques."""
    question: str = dspy.InputField(desc="Original question.")
    prior_answer: str = dspy.InputField(desc="The previous answer.")
    critiques: str = dspy.InputField(desc="Concise list of critiques to address.")
    improved_answer: str = dspy.OutputField(
        desc="A corrected and improved answer; concise, accurate, and directly addressing the question."
    )


class EvaluateAnswer(dspy.Signature):
    """LM-based heuristic judge: returns sub-scores and an overall score [0,1]."""
    question: str = dspy.InputField()
    answer: str = dspy.InputField()
    rubric: str = dspy.InputField(
        desc="Scoring rubric and weights for correctness, relevance, completeness, clarity."
    )
    reasoning: str = dspy.OutputField(
        desc="Brief evaluation reasoning justifying the scores."
    )
    scores_json: str = dspy.OutputField(
        desc=('JSON with keys: correctness, relevance, completeness, clarity, overall. '
              'Each in [0,1]. Keep valid JSON.')
    )


# --------------------------- DSPy Modules ---------------------------

class Solver(dspy.Module):
    """Initial solution generator."""
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(SolveTask)

    @backoff.on_exception(backoff.expo, Exception, max_tries=3, jitter=None)
    def __call__(self, question: str) -> str:
        out = self.predict(question=question)
        return out.answer.strip()


class Critic(dspy.Module):
    """Answer critic providing actionable feedback."""
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(CritiqueAttempt)

    @backoff.on_exception(backoff.expo, Exception, max_tries=3, jitter=None)
    def __call__(self, question: str, answer: str) -> str:
        out = self.predict(question=question, answer=answer)
        # Keep critique crisp
        return out.critique.strip()


class Reviser(dspy.Module):
    """Refines the answer using accumulated critiques."""
    def __init__(self):
        super().__init__()
        # ChainOfThought tends to improve revision quality; adjust if you prefer Predict
        self.cot = dspy.ChainOfThought(ReviseAnswer)

    @backoff.on_exception(backoff.expo, Exception, max_tries=3, jitter=None)
    def __call__(self, question: str, prior_answer: str, critiques: List[str]) -> str:
        # Keep critiques compact to avoid prompt bloat
        joined = "- " + "\n- ".join(c.strip() for c in critiques[-8:])
        out = self.cot(question=question, prior_answer=prior_answer, critiques=joined)
        return out.improved_answer.strip()


class LMJudge(dspy.Module):
    """LM-based judge that outputs normalized scores in [0,1]."""
    def __init__(self, weights: Dict[str, float]):
        super().__init__()
        self.predict = dspy.Predict(EvaluateAnswer)
        self.weights = weights

    def _rubric_text(self) -> str:
        return (
            "Scoring rubric (each in [0,1]):\n"
            f"- correctness (weight {self.weights['correctness']}): factual accuracy.\n"
            f"- relevance (weight {self.weights['relevance']}): stays on topic.\n"
            f"- completeness (weight {self.weights['completeness']}): covers key parts.\n"
            f"- clarity (weight {self.weights['clarity']}): clear and concise.\n"
            "Return valid JSON in 'scores_json' with keys: correctness, relevance, completeness, clarity, overall.\n"
            "The 'overall' should be a weighted sum of sub-scores using the weights above."
        )

    @backoff.on_exception(backoff.expo, Exception, max_tries=3, jitter=None)
    def __call__(self, question: str, answer: str) -> Tuple[float, Dict[str, float], str]:
        out = self.predict(question=question, answer=answer, rubric=self._rubric_text())
        # Parse scores_json safely
        import json
        try:
            scores = json.loads(out.scores_json)
            overall = float(scores.get("overall", 0.0))
            # Clamp
            overall = max(0.0, min(1.0, overall))
            # Ensure all subs present
            for k in ["correctness", "relevance", "completeness", "clarity"]:
                scores[k] = max(0.0, min(1.0, float(scores.get(k, 0.0))))
            return overall, scores, out.reasoning.strip()
        except Exception as e:
            log.warning(f"Judge JSON parse failed ({e}); defaulting to 0.0")
            return 0.0, {"correctness": 0, "relevance": 0, "completeness": 0, "clarity": 0}, "Parse error"


# --------------------------- Reflexion Orchestrator ---------------------------

class ReflexionAgent:
    """
    Reflexion Agent orchestrates solve -> critique -> revise loops until success.

    You may inject a custom judge via `judge_fn`, which should return a tuple:
        (overall_score: float in [0,1], subscores: Dict[str,float], reasoning: str)
    """
    def __init__(
        self,
        config: ReflexionConfig,
        judge_fn: Optional[Callable[[str, str], Tuple[float, Dict[str, float], str]]] = None,
    ):
        self.config = config
        self.solver = Solver()
        self.critic = Critic()
        self.reviser = Reviser()
        self.judge = judge_fn or LMJudge(config.judge_weighting)

    def run(self, question: str) -> Dict[str, Any]:
        """
        Executes the Reflexion loop.

        Returns a dict with:
        - 'final_answer'
        - 'rounds'
        - 'history': list of rounds with {'answer','critique','score','subs','judge_reasoning'}
        """
        history: List[Dict[str, Any]] = []
        critiques: List[str] = []

        # Round 1: initial solve
        answer = self.solver(question)
        score, subs, jr = self.judge(question, answer)
        history.append({
            "round": 1, "answer": answer, "critique": "", "score": score, "subs": subs, "judge_reasoning": jr
        })
        log.info(f"[Round 1] score={score:.2f} subs={subs}")

        if score >= self.config.success_threshold:
            log.info("[Early Stop] Initial answer meets success threshold.")
            return {"final_answer": answer, "rounds": 1, "history": history}

        # Subsequent rounds: critique + revise
        for r in range(2, self.config.max_rounds + 1):
            critique = self.critic(question, answer)
            critiques.append(critique)
            # keep short history of critiques
            if len(critiques) > self.config.max_critique_history:
                critiques = critiques[-self.config.max_critique_history:]

            answer = self.reviser(question, prior_answer=answer, critiques=critiques)
            score, subs, jr = self.judge(question, answer)

            history.append({
                "round": r, "answer": answer, "critique": critique, "score": score, "subs": subs, "judge_reasoning": jr
            })
            log.info(f"[Round {r}] score={score:.2f} subs={subs}")

            if score >= self.config.success_threshold:
                log.info("[Early Stop] Success threshold reached.")
                break

        return {"final_answer": answer, "rounds": len(history), "history": history}


# --------------------------- Model Initialization ---------------------------

def init_lm(
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 800,
):
    """
    Configure DSPy with an OpenAI (or compatible) LM.

    - Standard OpenAI: set OPENAI_API_KEY
    - Azure OpenAI: set OPENAI_API_KEY + OPENAI_API_BASE + OPENAI_API_VERSION; pass model=your_deployment_name
    - OpenAI-compatible: set OPENAI_API_KEY + OPENAI_API_BASE
    """
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")  # optional for compatible endpoints
    api_version = os.getenv("OPENAI_API_VERSION")  # optional, Azure-style

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    if api_base and api_version:
        # Azure-style
        lm = dspy.AzureOpenAI(
            model=model,
            api_version=api_version,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif api_base:
        # OpenAI-compatible endpoint
        lm = dspy.OpenAI(
            model=model,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        # Standard OpenAI
        lm = dspy.OpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    dspy.settings.configure(lm=lm)
    log.info(f"Initialized LM: model={model}, temp={temperature}, max_tokens={max_tokens}")


# --------------------------- Example Usage ---------------------------

def simple_demo():
    """Minimal demonstration on a factual QA."""
    cfg = ReflexionConfig(
        max_rounds=3,
        success_threshold=0.9,   # adjust to taste
        model="gpt-4o-mini",
        temperature=0.2,
        max_tokens=800,
    )
    agent = ReflexionAgent(cfg)

    question = (
        "Explain what the Reflexion pattern is in LLM systems and give a concise example. "
        "Keep the answer under 120 words."
    )

    result = agent.run(question)

    print("\n=== FINAL ANSWER ===")
    print(result["final_answer"])
    print("\n=== TRACE ===")
    for h in result["history"]:
        print(f"Round {h['round']}:")
        if h['critique']:
            print(f"  Critique: {h['critique'][:200]}{'...' if len(h['critique'])>200 else ''}")
        print(f"  Score: {h['score']:.2f} | Subs: {h['subs']}")
        print(f"  Judge: {h['judge_reasoning'][:160]}{'...' if len(h['judge_reasoning'])>160 else ''}")


if __name__ == "__main__":
    # Initialize LM once per process
    init_lm(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
        max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "800")),
    )
    simple_demo()
