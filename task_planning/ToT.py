# --- imports
from __future__ import annotations
from typing import List, Literal, Union, NamedTuple, Optional, Dict, Any
import operator, json
from typing_extensions import Annotated, TypedDict
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Send

# ====== Generic ToT scaffolding ======

# Utility reducer to append to a list in graph state
def _append_list(existing: Optional[list] = None,
                 updates: Optional[Union[list, Literal["clear"]]] = None) -> list:
    if existing is None: existing = []
    if updates is None:  return existing
    if updates == "clear": return []
    return existing + updates

# A "candidate" is the unit we expand/score/prune.
class Candidate(NamedTuple):
    text: str                  # task-specific content (equation, outline step, etc.)
    score: Optional[float] = None
    feedback: Optional[str] = None

class ScoredCandidate(Candidate):
    score: float
    feedback: str

# Graph state & context
class ToTState(TypedDict):
    problem: str
    candidates: Annotated[List[Candidate], _append_list]
    scored: Annotated[List[ScoredCandidate], _append_list]
    depth: Annotated[int, operator.add]

class Context(TypedDict, total=False):
    max_depth: int
    threshold: float
    k: int            # number of children proposed per node
    beam_size: int    # width b

class EnsuredContext(TypedDict):
    max_depth: int; threshold: float; k: int; beam_size: int

def _ctx(runtime: Runtime[Context]) -> EnsuredContext:
    c = runtime.context or {}
    return {
        "max_depth": c.get("max_depth", 8),
        "threshold": c.get("threshold", 0.95),
        "k": c.get("k", 5),
        "beam_size": c.get("beam_size", 3),
    }

# ====== Task interfaces ======

class ToTTask:
    """Implement these two hooks per task."""
    def build_solver(self, llm: ChatOpenAI, k: int):
        """Return a LangChain Runnable that maps {problem, seed?, k} -> {"candidates": [Candidate,...]}"""
        raise NotImplementedError

    def score(self, problem: str, cand: Candidate) -> ScoredCandidate:
        """Return ScoredCandidate with numeric score in [0,1] and textual feedback."""
        raise NotImplementedError

# ====== Task A: Game of 24 ======
# Following the paper setup: propose candidate equations; score by correctness & closeness.  [oai_citation:4‡LangChain](https://langchain-ai.github.io/langgraph/tutorials/tot/tot/) [oai_citation:5‡GitHub](https://github.com/princeton-nlp/tree-of-thought-llm)

OperatorType = Literal["+", "-", "*", "/"]
TokenType = Union[float, OperatorType]

class Equation(BaseModel):
    """Reverse-Polish notation (RPN) tokens; easier to validate & evaluate."""
    tokens: List[TokenType] = Field(
        description="RPN tokens, e.g., [3, 4, '+', 2, '*'] -> (3+4)*2"
    )
    def compute(self) -> float:
        ops = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}
        stack = []
        for t in self.tokens:
            if isinstance(t, float) or isinstance(t, int):
                stack.append(float(t))
            else:
                b, a = stack.pop(), stack.pop()
                stack.append(ops[t](a, b))
        return stack[0]

class GuessBatch(BaseModel):
    reasoning: str
    equations: List[Equation]

class Game24Task(ToTTask):
    def build_solver(self, llm: ChatOpenAI, k: int):
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are playing the Game of 24. Using the four numbers, propose exactly {k} candidate equations "
             "that evaluate to 24, expressed as reverse-polish notation tokens. "
             "Use each input number exactly once; you may use + - * /."),
            ("user",
             "Numbers: {problem}\n"
             "{seed}\n"
             "Return strictly in the requested structured schema.")
        ]).partial()

        # Ask the model for structured output -> GuessBatch Pydantic
        bound = llm.with_structured_output(GuessBatch)

        def _solver(inputs: Dict[str, Any]) -> Dict[str, List[Candidate]]:
            seed = f"Earlier candidate: {inputs['seed']}" if inputs.get("seed") else ""
            out: GuessBatch = (prompt | bound).invoke({"problem": inputs["problem"], "seed": seed, "k": k})
            cands = [Candidate(text=eq.model_dump_json()) for eq in out.equations]
            return {"candidates": cands}

        return _solver

    def score(self, problem: str, cand: Candidate) -> ScoredCandidate:
        # Validate uses exactly the given four numbers once, and score closeness to 24.
        numbers = sorted(list(map(int, problem.split())))
        try:
            eq = Equation.model_validate_json(cand.text)
            used_nums = sorted([int(x) for x in eq.tokens if isinstance(x, (int, float))])
            if used_nums != numbers:
                return ScoredCandidate(text=cand.text, score=0.0,
                                       feedback="Must use all four numbers exactly once.")
            val = eq.compute()
            # Reward 1.0 if exact; otherwise 1/(1+|24-val|).
            score = 1.0 if abs(val - 24) < 1e-6 else 1.0 / (1.0 + abs(24 - val))
            return ScoredCandidate(text=cand.text, score=score, feedback=f"Evaluates to {val}")
        except Exception as e:
            return ScoredCandidate(text=cand.text, score=0.0, feedback=f"Invalid equation: {e}")

# ====== Task B: Creative Writing (LLM value function) ======
# Mirrors the paper's "vote/value" style: propose diverse plot openings; LLM scores for criteria.  [oai_citation:6‡GitHub](https://github.com/princeton-nlp/tree-of-thought-llm)

class WritingTask(ToTTask):
    def __init__(self, rubric: str = "coherence, novelty, and emotional hook"):
        self.rubric = rubric

    def build_solver(self, llm: ChatOpenAI, k: int):
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a fiction writing assistant. Propose {k} distinct short openings (2-3 sentences each) "
             "that continue the user's premise. Output JSON list only."),
            ("user", "Premise: {problem}\nPrevious attempt (optional): {seed}")
        ])
        def _solver(inputs: Dict[str, Any]) -> Dict[str, List[Candidate]]:
            seed = inputs.get("seed") or ""
            text = (prompt | llm).invoke({"problem": inputs["problem"], "seed": seed, "k": k}).content
            # Try to parse a JSON list of strings; if not, split lines.
            try:
                proposals = json.loads(text)
                if isinstance(proposals, dict) and "openings" in proposals:
                    proposals = proposals["openings"]
            except Exception:
                proposals = [s.strip("-• ") for s in text.split("\n") if s.strip()][:k]
            cands = [Candidate(text=p) for p in proposals[:k]]
            return {"candidates": cands}
        return _solver

    def score(self, problem: str, cand: Candidate) -> ScoredCandidate:
        # LLM-as-a-judge: 0..1 score with brief feedback
        judge = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        jprompt = ChatPromptTemplate.from_messages([
            ("system",
             "Score the candidate on a 0..1 scale for the user's premise based on {rubric}. "
             "Return ONLY compact JSON: {\"score\": <float>, \"feedback\": \"...\"}"),
            ("user", "Premise: {premise}\nCandidate:\n{cand}")
        ]).partial(rubric=self.rubric)
        raw = (jprompt | judge).invoke({"premise": problem, "cand": cand.text}).content
        try:
            obj = json.loads(raw)
            score = float(obj.get("score", 0))
            fb = obj.get("feedback", "")
        except Exception:
            score, fb = 0.0, f"Judge parsing failed; raw={raw[:120]}"
        return ScoredCandidate(text=cand.text, score=score, feedback=fb)

# ====== Build the ToT graph (task-agnostic) ======

def build_tot_graph(task: ToTTask, model: str = "gpt-4o-mini"):
    llm = ChatOpenAI(model=model, temperature=0.7)
    builder = StateGraph(state_schema=ToTState, context_schema=Context)

    # Nodes
    def expand(state: Dict, *, runtime: Runtime[Context]):
        cfg = _ctx(runtime)
        # On first step we have no seed; afterwards we branch by seeding with each pruned candidate.
        seed = state.get("seed")
        solver = task.build_solver(llm, k=cfg["k"])
        return solver({"problem": state["problem"], "seed": seed})

    def score(state: Dict, *, runtime: Runtime[Context]):
        scored = [task.score(state["problem"], c) for c in state["candidates"]]
        return {"scored": scored, "candidates": "clear"}

    def prune(state: Dict, *, runtime: Runtime[Context]):
        cfg = _ctx(runtime)
        ordered = sorted(state["scored"], key=lambda sc: sc.score, reverse=True)
        keep = ordered[: cfg["beam_size"]]
        return {"candidates": keep, "scored": "clear", "depth": 1}

    def should_continue(state: Dict, runtime: Runtime[Context]):
        cfg = _ctx(runtime)
        solved = bool(state["candidates"]) and state["candidates"][0].score is not None \
                 and state["candidates"][0].score >= cfg["threshold"]
        if solved or state["depth"] >= cfg["max_depth"]:
            return "__end__"
        # Fan out: each kept candidate becomes a seed for the next expand
        return [Send("expand", {"problem": state["problem"], "seed": c}) for c in state["candidates"]]

    # Wire graph
    builder.add_node(expand)
    builder.add_node(score)
    builder.add_node(prune)
    builder.add_edge("__start__", "expand")
    builder.add_edge("expand", "score")
    builder.add_edge("score", "prune")
    builder.add_conditional_edges("prune", should_continue, path_map=["expand", "__end__"])

    return builder.compile(checkpointer=InMemorySaver())

# ====== Usage examples ======

if __name__ == "__main__":
    # Example 1: Game of 24
    game = Game24Task()
    graph_24 = build_tot_graph(game)
    problem_24 = "4 5 6 10"   # classic 24-puzzle instance
    # Stream steps (optional); the best candidate is always candidates[0]
    for ev in graph_24.stream({"problem": problem_24},
                              context={"beam_size": 5, "k": 5, "threshold": 0.999, "max_depth": 10},
                              config={"configurable": {"thread_id": "tot_24_demo"}}):
        print(ev)
    print("Best:", graph_24.invoke({"problem": problem_24},
                                   context={"beam_size": 5, "k": 5, "threshold": 0.999, "max_depth": 10}
                                   )["candidates"][0])

    # Example 2: Creative writing
    writing = WritingTask(rubric="coherence, novelty, hook, vivid imagery")
    graph_write = build_tot_graph(writing)
    premise = "A solar-punk city depends on a giant algae reef that suddenly stops growing."
    result = graph_write.invoke({"problem": premise},
                                context={"beam_size": 3, "k": 4, "threshold": 0.8, "max_depth": 4})
    best_opening = result["candidates"][0]
    print("\nBest opening:", best_opening.text, "\nScore:", best_opening.score, "\nWhy:", best_opening.feedback)