#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-file Tree-of-Thoughts (ToT) for TEXT tasks (Creative Writing),
adapted from the official paper/repo configuration:
  - Thought generation strategy: 'sample' (i.i.d. candidates)
  - State evaluation strategy:   'vote'  (LLM chooses best among candidates)
  - Search strategy:             BFS with pruning (keep top-b = n_select_sample)

References:
- Paper: Tree of Thoughts: Deliberate Problem Solving with Large Language Models (Yao et al., 2023)
- Official repo quick-start and flags layout (naive_run, prompt_sample, method_generate, method_evaluate, n_* knobs)

This script intentionally limits itself to the TEXT example.
"""

import os
import argparse
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field
from openai import OpenAI

# -----------------------------
# OpenAI helper
# -----------------------------
def chat_complete(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.7,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content.strip()


# -----------------------------
# Prompt templates (text task)
# -----------------------------
#
# In the paper/repo, the TEXT task uses:
#   - "sample" generator: generate k distinct continuations/passages in parallel
#   - "vote" evaluator: compare k candidates and pick the best one
#
# The vote/evaluator prompt follows the shape:
#   "Given an instruction and several choices, analyze each, then conclude:
#    'The best choice is s' (s is an integer id)."
#
# (See repo readme & public references describing ToT prompts for TEXT)  [citations in the chat]
#

SYSTEM_SAMPLE = (
    "You are a helpful writing model. "
    "Given a creative-writing instruction, produce DISTINCT candidate continuations. "
    "Be concise and follow instructions exactly."
)

# Standard sampling (IO) for TEXT: produce multiple i.i.d ideas
SAMPLE_USER_TEMPLATE = """Instruction:
{instruction}

Produce {k} DISTINCT short passages (2-4 sentences each) that continue or fulfill the instruction.
Return them as a numbered list 1..{k}. Keep each passage compact.
"""

# CoT sampling (optional baseline): ask model to think then answer once
SYSTEM_COT = (
    "You are a helpful reasoning model. Think step by step (briefly) then produce a single final passage."
)
COT_USER_TEMPLATE = """Instruction:
{instruction}

First, think briefly about the best direction (hidden thoughts). Then write ONE compact passage (3-5 sentences).
Return ONLY the final passage (no analysis)."""

# Vote evaluator: compare multiple choices and pick the single best index
SYSTEM_VOTE = (
    "You are a careful writing judge. Given an instruction and several candidate passages, "
    "analyze each for coherence, creativity, clarity, and how well it fulfills the instruction. "
    "Then CHOOSE the single best candidate."
)
VOTE_USER_TEMPLATE = """Instruction:
{instruction}

Choices:
{choices_str}

Analyze each choice briefly. On the LAST line, write exactly:
The best choice is {s}
(where s is the integer id)."""


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Candidate:
    text: str
    heuristic: float = 0.0  # number of votes (for vote-based heuristic)


class BatchSample(BaseModel):
    # Best-effort parser when the model returns JSON (not required)
    items: List[str] = Field(default_factory=list)


# -----------------------------
# Parsing helpers
# -----------------------------
def parse_numbered_list(text: str, k_expected: int) -> List[str]:
    """
    Parse a numbered list 1..k from a chat response.
    Falls back to splitting lines if formatting is off.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Try numbered prefix "1.", "2)", etc.
    out: List[str] = []
    cur = []
    cur_idx = 1

    def flush():
        nonlocal cur, out
        if cur:
            out.append(" ".join(cur).strip())
            cur = []

    for ln in lines:
        # line starts with the expected index?
        if ln.startswith(f"{cur_idx}.") or ln.startswith(f"{cur_idx})"):
            flush()
            cur = [ln.split(".", 1)[-1].split(")", 1)[-1].strip()]
            cur_idx += 1
        else:
            cur.append(ln)
    flush()

    # If nothing parsed, fallback to naive split into k chunks
    if not out:
        # try splitting by blank lines
        para = [p.strip() for p in text.split("\n\n") if p.strip()]
        out = para[:k_expected] if para else [text.strip()]

    # Ensure k items
    if len(out) < k_expected:
        # just duplicate last if too few (robustness)
        while len(out) < k_expected:
            out.append(out[-1] if out else "")
    return out[:k_expected]


def parse_vote_best_index(text: str, k: int) -> int:
    """
    Extract the integer s from line: 'The best choice is s'
    Return 1-based index; default to 1 if not found.
    """
    last_line = text.strip().splitlines()[-1].strip().lower()
    # Look for a trailing integer
    import re
    m = re.search(r"the best choice is\s+(\d+)", last_line)
    if not m:
        return 1
    s = int(m.group(1))
    if s < 1 or s > k:
        return 1
    return s


# -----------------------------
# ToT for TEXT (BFS + sample + vote)
# -----------------------------
def tot_text_bfs(
    client: OpenAI,
    model: str,
    instruction: str,
    *,
    temperature: float = 0.7,
    n_generate_sample: int = 5,   # k thoughts per level (paper often uses 5)
    n_evaluate_sample: int = 3,   # how many vote calls (majority vote)
    n_select_sample: int = 1,     # beam width b (paper used 1 for TEXT)
    max_depth: int = 3,           # levels of BFS
    verbose: bool = True,
) -> Tuple[str, Dict]:
    """
    Tree-of-Thoughts BFS for creative-writing style tasks:
    - generator: 'sample' → produce k parallel candidates
    - evaluator: 'vote'   → majority-vote among candidates

    Returns best passage and a small info dict with traces.
    """
    # Level 0: frontier holds one "state" → the instruction itself
    frontier: List[str] = [instruction]
    trace = []

    for depth in range(1, max_depth + 1):
        all_children: List[Candidate] = []

        for state_idx, state in enumerate(frontier, start=1):
            # 1) Generate k candidates (SAMPLE)
            user_prompt = SAMPLE_USER_TEMPLATE.format(instruction=state, k=n_generate_sample)
            gen = chat_complete(client, model, SYSTEM_SAMPLE, user_prompt, temperature)
            passages = parse_numbered_list(gen, n_generate_sample)

            # 2) EVALUATE by voting: run the judge n_evaluate_sample times
            votes = [0] * n_generate_sample
            choices_str = "\n".join([f"{i+1}. {p}" for i, p in enumerate(passages)])

            for _ in range(n_evaluate_sample):
                judge_resp = chat_complete(
                    client, model, SYSTEM_VOTE,
                    VOTE_USER_TEMPLATE.format(instruction=state, choices_str=choices_str, s="{s}"),
                    temperature=0.0,
                )
                best_idx = parse_vote_best_index(judge_resp, n_generate_sample)
                votes[best_idx - 1] += 1

            # 3) Add children with vote-heuristics
            for p, v in zip(passages, votes):
                all_children.append(Candidate(text=p, heuristic=float(v)))

            if verbose:
                print(f"\n[Depth {depth} | Parent {state_idx}]")
                for i, (p, v) in enumerate(zip(passages, votes), start=1):
                    print(f"  {i:>2}. (votes={v}) {p[:140]}")

        # 4) PRUNE: keep top-b by votes (ties keep earlier ones)
        all_children.sort(key=lambda c: c.heuristic, reverse=True)
        kept = all_children[: max(1, n_select_sample)]
        frontier = [c.text for c in kept]
        trace.append(
            {
                "depth": depth,
                "generated": [c.text for c in all_children],
                "votes": [c.heuristic for c in all_children],
                "kept": [c.text for c in kept],
            }
        )

        # optional early stop: if unanimous & strong, stop
        if kept and kept[0].heuristic >= max(2, n_evaluate_sample):
            break

    best = frontier[0] if frontier else ""
    return best, {"trace": trace}


# -----------------------------
# Naïve baselines (IO / CoT)
# -----------------------------
def naive_io(
    client: OpenAI, model: str, instruction: str, temperature: float = 0.7
) -> str:
    system = "You are a helpful writing model. Follow the instruction precisely."
    user = f"Instruction:\n{instruction}\n\nWrite a concise passage (4-6 sentences)."
    return chat_complete(client, model, system, user, temperature)

def naive_cot(
    client: OpenAI, model: str, instruction: str, temperature: float = 0.7
) -> str:
    gen = chat_complete(client, model, SYSTEM_COT, COT_USER_TEMPLATE.format(instruction=instruction), temperature)
    return gen


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="ToT (TEXT) single-file runner")
    parser.add_argument("--model", default="gpt-4o-mini", type=str, help="OpenAI chat model")
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--instruction", required=True, type=str, help="Creative writing instruction/premise")

    # flags to mirror official repo run.py semantics for paper tasks
    parser.add_argument("--naive_run", action="store_true", help="If set, run naive baseline instead of ToT")
    parser.add_argument("--prompt_sample", choices=["standard", "cot"], default="standard",
                        help="For naive runs: standard (IO) vs cot")
    parser.add_argument("--method_generate", choices=["sample"], default="sample",
                        help="Generator for TEXT is 'sample' (fixed here).")
    parser.add_argument("--method_evaluate", choices=["vote"], default="vote",
                        help="Evaluator for TEXT is 'vote' (fixed here).")

    parser.add_argument("--n_generate_sample", default=5, type=int, help="k (thoughts per level)")
    parser.add_argument("--n_evaluate_sample", default=3, type=int, help="# judge votes per level")
    parser.add_argument("--n_select_sample", default=1, type=int, help="beam width b")
    parser.add_argument("--max_depth", default=3, type=int, help="BFS depth")
    parser.add_argument("--verbose", action="store_true", help="Print intermediate results")

    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENAI_API_KEY environment variable.")

    client = OpenAI(api_key=api_key)

    if args.naive_run:
        if args.prompt_sample == "cot":
            out = naive_cot(client, args.model, args.instruction, args.temperature)
        else:
            out = naive_io(client, args.model, args.instruction, args.temperature)
        print("\n=== Output (Naive) ===\n" + out)
        return

    # ToT BFS (TEXT)
    best, info = tot_text_bfs(
        client=client,
        model=args.model,
        instruction=args.instruction,
        temperature=args.temperature,
        n_generate_sample=args.n_generate_sample,
        n_evaluate_sample=args.n_evaluate_sample,
        n_select_sample=args.n_select_sample,
        max_depth=args.max_depth,
        verbose=args.verbose,
    )

    print("\n=== Best Passage (ToT) ===\n" + best)
    # Optional: save trace JSON for analysis
    # print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()