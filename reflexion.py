# reflexion.py
"""
Reflexion pattern implemented with:
1) LangGraph (if available), or
2) A framework-free orchestrator fallback.

Components:
- Solver: initial answer
- Critic: targeted feedback
- Reviser: improves answer using critiques
- Judge: scores (overall + sub-scores), controls early stop

Features:
- Pluggable LLM + judge
- Config via dataclass
- Structured logging
- Backoff retries
- Deterministic defaults
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, TypedDict, Any

import backoff

# --------------------------- Logging ---------------------------

def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("reflexion")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

log = _setup_logger()


# --------------------------- Config ---------------------------

@dataclass(frozen=True)
class ReflexionConfig:
    max_rounds: int = 3
    success_threshold: float = 0.85
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature: float = 0.2
    max_output_tokens: int = 800
    critique_history: int = 3
    # judge weights for overall score = sum(weights[k] * score_k)
    judge_weights: Dict[str, float] = field(default_factory=lambda: {
        "correctness": 0.5,
        "relevance": 0.2,
        "completeness": 0.2,
        "clarity": 0.1,
    })


# --------------------------- LLM Adapter ---------------------------

class ChatLLM:
    """
    Minimal OpenAI (or compatible) chat wrapper.
    Expects OPENAI_API_KEY. Optional: OPENAI_BASE_URL.
    """

    def __init__(self, model: str, temperature: float, max_tokens: int):
        from openai import OpenAI  # import here to avoid hard dependency at module load
        base_url = os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(base_url=base_url) if base_url else OpenAI()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @backoff.on_exception(backoff.expo, Exception, max_tries=3, jitter=None)
    def chat(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (resp.choices[0].message.content or "").strip()


# --------------------------- Prompts ---------------------------

SOLVER_SYSTEM = (
    "You are a careful, concise problem solver. Provide factual answers. "
    "If unsure, say you are unsure and suggest what would resolve uncertainty."
)

CRITIC_SYSTEM = (
    "You are a rigorous critic. Identify concrete issues in the answer: "
    "factual errors, missing details, irrelevance, ambiguity. "
    "Return actionable, specific suggestions for improvement, not a rewrite."
)

REVISER_SYSTEM = (
    "You revise answers using the critiques. Fix issues precisely and keep the answer "
    "concise, accurate, and directly responsive to the question."
)

JUDGE_SYSTEM = (
    "You are a strict evaluator. Return valid JSON only with keys: "
    "correctness, relevance, completeness, clarity, overall — each in [0,1]. "
    "The 'overall' is a weighted sum per the rubric."
)


def judge_user_prompt(question: str, answer: str, weights: Dict[str, float]) -> str:
    return f"""
Question:
{question}

Answer:
{answer}

Rubric (weights):
- correctness: {weights['correctness']}
- relevance: {weights['relevance']}
- completeness: {weights['completeness']}
- clarity: {weights['clarity']}

Score each dimension in [0,1].
Compute overall = sum_i w_i * score_i.
Return ONLY JSON, e.g.:
{{"correctness":0.9,"relevance":0.9,"completeness":0.8,"clarity":0.8,"overall":0.86}}
""".strip()


# --------------------------- Components ---------------------------

class Solver:
    def __init__(self, llm: ChatLLM):
        self.llm = llm

    def __call__(self, question: str) -> str:
        prompt = (
            "Solve the user's question carefully and succinctly. "
            "If numeric, show key steps briefly. Prefer <150 words unless asked."
            f"\n\nQuestion:\n{question}"
        )
        return self.llm.chat(SOLVER_SYSTEM, prompt)


class Critic:
    def __init__(self, llm: ChatLLM):
        self.llm = llm

    def __call__(self, question: str, answer: str) -> str:
        prompt = (
            "Critique the answer. Be specific and actionable. "
            "List issues as bullets, each with a suggested fix. "
            "Do NOT rewrite the whole answer.\n\n"
            f"Question:\n{question}\n\nAnswer:\n{answer}"
        )
        return self.llm.chat(CRITIC_SYSTEM, prompt)


class Reviser:
    def __init__(self, llm: ChatLLM):
        self.llm = llm

    def __call__(self, question: str, prior_answer: str, critiques: List[str]) -> str:
        # compact critique list
        joined = "- " + "\n- ".join(c.strip() for c in critiques[-8:])
        prompt = (
            "Revise the prior answer strictly according to the critiques. "
            "Keep it concise, correct, and directly answers the question.\n\n"
            f"Question:\n{question}\n\nPrior Answer:\n{prior_answer}\n\nCritiques:\n{joined}"
        )
        return self.llm.chat(REVISER_SYSTEM, prompt)


class Judge:
    def __init__(self, llm: ChatLLM, weights: Dict[str, float]):
        self.llm = llm
        self.weights = weights

    def __call__(self, question: str, answer: str) -> Tuple[float, Dict[str, float], str]:
        raw = self.llm.chat(JUDGE_SYSTEM, judge_user_prompt(question, answer, self.weights))
        try:
            scores = json.loads(raw)
            # normalize and clamp
            subs = {}
            for k in ["correctness", "relevance", "completeness", "clarity"]:
                subs[k] = max(0.0, min(1.0, float(scores.get(k, 0.0))))
            overall = float(scores.get("overall", 0.0))
            overall = max(0.0, min(1.0, overall))
            return overall, subs, "LM-judge"
        except Exception as e:
            log.warning(f"Judge JSON parse failed: {e}. Raw: {raw[:180]}")
            return 0.0, {"correctness": 0, "relevance": 0, "completeness": 0, "clarity": 0}, "parse_error"


# --------------------------- Shared State ---------------------------

class ReflexionState(TypedDict, total=False):
    question: str
    answer: str
    critiques: List[str]
    round: int
    score: float
    subs: Dict[str, float]
    history: List[Dict[str, Any]]
    max_rounds: int
    threshold: float


# --------------------------- Orchestrator (no framework) ---------------------------

class ReflexionLoop:
    """Framework-free Reflexion loop (used as fallback or for simple deployments)."""

    def __init__(self, cfg: ReflexionConfig, solver: Solver, critic: Critic, reviser: Reviser, judge: Judge):
        self.cfg = cfg
        self.solver = solver
        self.critic = critic
        self.reviser = reviser
        self.judge = judge

    def run(self, question: str) -> Dict[str, Any]:
        state: ReflexionState = {
            "question": question,
            "critiques": [],
            "round": 1,
            "history": [],
            "max_rounds": self.cfg.max_rounds,
            "threshold": self.cfg.success_threshold,
        }

        # Round 1: solve
        answer = self.solver(question)
        score, subs, _ = self.judge(question, answer)
        state["answer"] = answer
        state["score"] = score
        state["subs"] = subs
        state["history"].append({"round": 1, "answer": answer, "score": score, "subs": subs})
        log.info(f"[Round 1] score={score:.2f} subs={subs}")

        if score >= self.cfg.success_threshold:
            log.info("[Early Stop] Success at round 1.")
            return {"final_answer": answer, "rounds": 1, "history": state["history"]}

        # Rounds 2..N
        while state["round"] < self.cfg.max_rounds:
            state["round"] += 1
            critique = self.critic(question, state["answer"])
            state["critiques"].append(critique)
            # limit critique memory
            if len(state["critiques"]) > self.cfg.critique_history:
                state["critiques"] = state["critiques"][-self.cfg.critique_history:]

            improved = self.reviser(question, state["answer"], state["critiques"])
            score, subs, _ = self.judge(question, improved)
            state["answer"] = improved
            state["score"] = score
            state["subs"] = subs
            state["history"].append(
                {"round": state["round"], "answer": improved, "critique": critique, "score": score, "subs": subs}
            )
            log.info(f"[Round {state['round']}] score={score:.2f} subs={subs}")

            if score >= self.cfg.success_threshold:
                log.info("[Early Stop] Threshold reached.")
                break

        return {"final_answer": state["answer"], "rounds": state["round"], "history": state["history"]}


# --------------------------- LangGraph Graph (optional) ---------------------------

def build_langgraph_app(cfg: ReflexionConfig, solver: Solver, critic: Critic, reviser: Reviser, judge: Judge):
    """
    Build a LangGraph app that implements the Reflexion loop.
    Requires `pip install langgraph`. Falls back to framework-free if not installed.
    """
    try:
        from langgraph.graph import StateGraph, END
    except Exception as e:
        log.warning(f"LangGraph not available ({e}); using framework-free loop.")
        return None

    class LGState(TypedDict, total=False):
        # mirror ReflexionState keys used in nodes
        question: str
        answer: str
        critiques: List[str]
        round: int
        score: float
        subs: Dict[str, float]
        history: List[Dict[str, Any]]
        max_rounds: int
        threshold: float

    def node_solve(state: LGState) -> LGState:
        if not state.get("round"):
            state["round"] = 1
        ans = solver(state["question"])
        return {**state, "answer": ans}

    def node_judge(state: LGState) -> LGState:
        score, subs, _ = judge(state["question"], state["answer"])
        hist = list(state.get("history", []))
        # if last item is same round (after revise), append; if first judge after solve, also append
        hist.append({"round": state.get("round", 1), "answer": state["answer"], "score": score, "subs": subs})
        return {**state, "score": score, "subs": subs, "history": hist}

    def node_critique(state: LGState) -> LGState:
        c = critic(state["question"], state["answer"])
        critiques = list(state.get("critiques", [])) + [c]
        # trim
        if len(critiques) > cfg.critique_history:
            critiques = critiques[-cfg.critique_history:]
        return {**state, "critiques": critiques}

    def node_revise(state: LGState) -> LGState:
        improved = reviser(state["question"], state["answer"], state.get("critiques", []))
        return {**state, "answer": improved, "round": state.get("round", 1) + 1}

    def should_continue(state: LGState) -> str:
        if state.get("score", 0.0) >= state.get("threshold", cfg.success_threshold):
            return "end"
        if state.get("round", 1) >= state.get("max_rounds", cfg.max_rounds):
            return "end"
        return "critique"

    graph = StateGraph(LGState)
    graph.add_node("solve", node_solve)
    graph.add_node("judge", node_judge)
    graph.add_node("critique", node_critique)
    graph.add_node("revise", node_revise)

    graph.set_entry_point("solve")
    graph.add_edge("solve", "judge")
    graph.add_conditional_edges("judge", should_continue, {"end": END, "critique": "critique"})
    graph.add_edge("critique", "revise")
    graph.add_edge("revise", "judge")

    # Optional: add a checkpointer if available (won't be used in this simple run)
    # from langgraph.checkpoint.memory import MemorySaver
    # app = graph.compile(checkpointer=MemorySaver())
    app = graph.compile()
    log.info("LangGraph app compiled.")
    return app


# --------------------------- Wiring & Demo ---------------------------

def build_components(cfg: ReflexionConfig) -> Tuple[ChatLLM, Solver, Critic, Reviser, Judge]:
    llm = ChatLLM(model=cfg.model, temperature=cfg.temperature, max_tokens=cfg.max_output_tokens)
    return llm, Solver(llm), Critic(llm), Reviser(llm), Judge(llm, cfg.judge_weights)

def demo_question() -> str:
    return (
        "In 120 words or fewer, explain the Reflexion pattern for LLMs and give a tiny example. "
        "Be concrete."
    )

def run_with_langgraph(question: str, cfg: ReflexionConfig) -> Dict[str, Any]:
    _, solver, critic, reviser, judge = build_components(cfg)
    app = build_langgraph_app(cfg, solver, critic, reviser, judge)
    if app is None:
        # fallback
        loop = ReflexionLoop(cfg, solver, critic, reviser, judge)
        return loop.run(question)

    # initialize state
    init_state: ReflexionState = {
        "question": question,
        "critiques": [],
        "round": 1,
        "history": [],
        "max_rounds": cfg.max_rounds,
        "threshold": cfg.success_threshold,
    }

    # Run graph to completion (until END)
    final_state = None
    for event in app.stream(init_state):
        # `event` yields node outputs in order; capture the last
        for _, s in event.items():
            final_state = s

    # Prepare output
    history = final_state.get("history", []) if final_state else []
    rounds = history[-1]["round"] if history else 1
    answer = final_state.get("answer", "") if final_state else ""
    return {"final_answer": answer, "rounds": rounds, "history": history}

def run_framework_free(question: str, cfg: ReflexionConfig) -> Dict[str, Any]:
    _, solver, critic, reviser, judge = build_components(cfg)
    loop = ReflexionLoop(cfg, solver, critic, reviser, judge)
    return loop.run(question)

def main():
    cfg = ReflexionConfig()
    question = demo_question()

    prefer_plain = os.getenv("REFLEXION_NO_FRAMEWORK", "0") == "1"
    if prefer_plain:
        log.info("Running framework-free Reflexion loop...")
        result = run_framework_free(question, cfg)
    else:
        log.info("Attempting to run with LangGraph (fallback to plain if unavailable)...")
        result = run_with_langgraph(question, cfg)

    print("\n=== FINAL ANSWER ===\n" + result["final_answer"])
    print("\n=== TRACE ===")
    for h in result["history"]:
        print(f"Round {h['round']}: score={h.get('score', 0):.2f}, subs={h.get('subs', {})}")
        if "critique" in h:
            snippet = h["critique"][:200]
            print(f"  critique: {snippet}{'...' if len(h['critique'])>200 else ''}")

if __name__ == "__main__":
    main()
