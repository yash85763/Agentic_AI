"""
LLM-as-a-Judge (OpenAI) — Direct-Assessment & Pairwise with Bias Mitigation

Features
- Direct assessment with weighted rubric (G-Eval style)
- Pairwise A/B judging with symmetric prompts (MT-Bench style)
- Bias mitigations: order randomization, verbosity/length hints, role-correctness
- Consistency: paraphrase-and-judge + A/B then B/A, aggregated
- Calibrated confidence: combines self-reported confidence with stability score
- Strict JSON schema via Responses API (fallback to robust JSON parsing)
- N-way ranking via round-robin pairwise votes

Requirements: `pip install openai pydantic`
Env: export OPENAI_API_KEY=...
"""

from __future__ import annotations
import os, json, random, textwrap, math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field, ValidationError

try:
    # OpenAI SDK v1+
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class Criterion:
    name: str
    description: str
    weight: float = 1.0  # relative weight

@dataclass
class Rubric:
    name: str
    scale_min: int = 1
    scale_max: int = 10
    criteria: List[Criterion] = field(default_factory=list)

    def normalized_weights(self) -> Dict[str, float]:
        total = sum(max(0.0, c.weight) for c in self.criteria) or 1.0
        return {c.name: max(0.0, c.weight) / total for c in self.criteria}

# Pydantic schemas for strict JSON output from the model
class CriterionScore(BaseModel):
    criterion: str = Field(..., description="Criterion name")
    score: float = Field(..., ge=0, le=10)
    rationale: str

class DirectAssessment(BaseModel):
    overall_score: float = Field(..., ge=0, le=10)
    confidence_self_report: float = Field(..., ge=0, le=1)
    per_criterion: List[CriterionScore]
    brief_rationale: str

class PairwiseVerdict(BaseModel):
    winner: str = Field(..., description="'A', 'B', or 'Tie'")
    margin: float = Field(..., ge=0, le=1, description="How strong is the preference (0-1)")
    confidence_self_report: float = Field(..., ge=0, le=1)
    reasons: str


# ----------------------------
# Prompt builders
# ----------------------------
JUDGE_SYSTEM = (
    "You are a meticulous, fair evaluation judge. "
    "Follow the rubric strictly. Avoid being swayed by verbosity; reward correctness, usefulness, and safety. "
    "When uncertain, be conservative. Output only the requested JSON—no extra text."
)

def build_direct_prompt(user_prompt: str, answer: str, rubric: Rubric) -> str:
    crit_lines = "\n".join(
        f"- {c.name}: {c.description} (weight={c.weight})"
        for c in rubric.criteria
    )
    return textwrap.dedent(f"""
    TASK: Evaluate the candidate answer for the given user prompt.

    USER PROMPT:
    ---
    {user_prompt}
    ---

    CANDIDATE ANSWER:
    ---
    {answer}
    ---

    RUBRIC (Scale {rubric.scale_min}-{rubric.scale_max}):
    {crit_lines}

    INSTRUCTIONS:
    - Be strict but fair. Check factuality and internal consistency where applicable.
    - Penalize padding/verbosity that doesn't add value.
    - Consider harmful content and safety as part of the rubric if relevant.
    - Provide a BRIEF rationale (2-4 sentences, no chain-of-thought).
    - Set confidence_self_report in [0,1] reflecting how certain you are.
    - Return JSON only.
    """)

def build_pairwise_prompt(user_prompt: str, ansA: str, ansB: str, rubric: Rubric, labelA="A", labelB="B") -> str:
    crit_lines = "\n".join(
        f"- {c.name}: {c.description} (weight={c.weight})"
        for c in rubric.criteria
    )
    return textwrap.dedent(f"""
    TASK: Compare two answers to the same prompt and pick a winner (or tie) using the rubric.

    USER PROMPT:
    ---
    {user_prompt}
    ---

    ANSWER {labelA}:
    ---
    {ansA}
    ---

    ANSWER {labelB}:
    ---
    {ansB}
    ---

    RUBRIC (Scale {rubric.scale_min}-{rubric.scale_max}):
    {crit_lines}

    INSTRUCTIONS:
    - Judge strictly on the rubric; ignore style fluff that doesn't help.
    - Consider correctness/grounding; do not reward hallucinations.
    - Prefer concise, accurate, and safe content over verbosity.
    - Output JSON only with: winner ('{labelA}','{labelB}','Tie'), margin [0,1], brief reasons (no chain-of-thought), confidence_self_report [0,1].
    """)

PARAPHRASES = [
    "Evaluate with extra attention to factual correctness and internal consistency.",
    "Evaluate with extra attention to instruction-following and completeness.",
    "Evaluate with extra attention to safety, harmful content avoidance, and neutrality.",
]


# ----------------------------
# OpenAI Client wrapper
# ----------------------------
class OpenAIJudge:
    """
    High-reliability judge using OpenAI Responses API with JSON schema.
    """

    def __init__(self, model: str = "gpt-4.1-mini", temperature: float = 0.0):
        if not OPENAI_AVAILABLE:
            raise RuntimeError("openai package not available. `pip install openai`")
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature

    # JSON schema helpers
    def _schema_for_direct(self):
        return {
            "name": "DirectAssessment",
            "schema": DirectAssessment.model_json_schema(),
            "strict": True
        }

    def _schema_for_pairwise(self):
        return {
            "name": "PairwiseVerdict",
            "schema": PairwiseVerdict.model_json_schema(),
            "strict": True
        }

    def _resp_json(self, content):
        # Responses API returns content in {type: "output_text"} or "output_json"
        # Prefer structured output tool:
        for item in content:
            if item.type == "output_json":
                return item.output
            if item.type == "output_text":
                # Fallback: robust JSON extraction
                txt = item.text
                start = txt.find("{")
                end = txt.rfind("}")
                if start != -1 and end != -1 and end > start:
                    return json.loads(txt[start:end+1])
        raise ValueError("Could not parse model JSON output.")

    # Core LLM call with schema
    def _call_with_schema(self, system: str, prompt: str, json_schema: dict) -> dict:
        resp = self.client.responses.create(
            model=self.model,
            temperature=self.temperature,
            system=system,
            input=[{"role": "user", "content": prompt}],
            response_format={"type": "json_schema", "json_schema": json_schema},
        )
        return self._resp_json(resp.output)

    # ------------------------
    # Public judging methods
    # ------------------------
    def direct_assess(self, user_prompt: str, answer: str, rubric: Rubric,
                      paraphrase_runs: int = 2) -> Dict:
        """Direct scoring with weighted rubric and stability estimation."""
        weights = rubric.normalized_weights()
        results = []
        for i in range(paraphrase_runs):
            addendum = PARAPHRASES[i % len(PARAPHRASES)]
            prompt = build_direct_prompt(user_prompt, answer, rubric) + "\n" + addendum
            raw = self._call_with_schema(JUDGE_SYSTEM, prompt, self._schema_for_direct())
            try:
                parsed = DirectAssessment(**raw)
            except ValidationError as e:
                raise RuntimeError(f"Schema validation failed: {e}") from e
            results.append(parsed)

        # Aggregate: mean per-criterion and overall, compute stability/confidence
        per_crit = {}
        for r in results:
            for cs in r.per_criterion:
                per_crit.setdefault(cs.criterion, []).append(cs.score)

        per_crit_mean = {k: sum(v)/len(v) for k, v in per_crit.items()}
        # Weighted overall
        weighted = sum(per_crit_mean[c]*weights.get(c, 0.0) for c in per_crit_mean)
        # Normalize to rubric scale
        overall = max(rubric.scale_min, min(rubric.scale_max, weighted))

        # Stability: inverse of std-dev across paraphrases
        if len(results) > 1:
            import statistics as stats
            stds = [stats.pstdev([r.overall_score for r in results])]
            crit_stds = [stats.pstdev(scores) for scores in per_crit.values() if len(scores) > 1]
            stability = 1/(1+sum(stds + crit_stds))  # 0..1-ish
        else:
            stability = 0.6

        conf_sr = sum(r.confidence_self_report for r in results)/len(results)
        calibrated_conf = 0.5*conf_sr + 0.5*stability

        return {
            "rubric": rubric.name,
            "overall_score": round(overall, 2),
            "per_criterion": {k: round(v, 2) for k, v in per_crit_mean.items()},
            "confidence": round(calibrated_conf, 3),
            "rationales": [r.brief_rationale for r in results]
        }

    def pairwise_judge(self, user_prompt: str, ansA: str, ansB: str, rubric: Rubric,
                       symmetric: bool = True, paraphrase_runs: int = 2) -> Dict:
        """Pairwise verdict with A/B and (optionally) B/A symmetric runs."""
        runs = []
        orderings: List[Tuple[str, str, str, str]] = [("A", "B", ansA, ansB)]
        if symmetric:
            orderings.append(("B", "A", ansB, ansA))

        for (la, lb, a, b) in orderings:
            for i in range(paraphrase_runs):
                prompt = build_pairwise_prompt(user_prompt, a, b, rubric, labelA=la, labelB=lb)
                prompt += "\n" + PARAPHRASES[i % len(PARAPHRASES)]
                raw = self._call_with_schema(JUDGE_SYSTEM, prompt, self._schema_for_pairwise())
                verdict = PairwiseVerdict(**raw)
                # Map back to absolute labels
                mapped_winner = verdict.winner
                if la == "B" and lb == "A":  # reverse mapping for symmetric run
                    mapped_winner = {"A": "B", "B": "A", "Tie": "Tie"}[verdict.winner]
                runs.append({
                    "winner": mapped_winner,
                    "margin": verdict.margin,
                    "conf": verdict.confidence_self_report,
                    "reasons": verdict.reasons
                })

        # Aggregate votes
        scoreA = sum((1 if r["winner"] == "A" else 0) + (0.5 if r["winner"] == "Tie" else 0) for r in runs)
        scoreB = sum((1 if r["winner"] == "B" else 0) + (0.5 if r["winner"] == "Tie" else 0) for r in runs)
        total = max(1, len(runs))
        pref = (scoreA - scoreB) / total  # -1..1

        # Confidence uses both self-report and vote margin magnitude
        mean_conf = sum(r["conf"] for r in runs)/total
        conf_calibrated = 0.5*mean_conf + 0.5*abs(pref)

        if abs(pref) < 0.1:
            final_winner = "Tie"
        else:
            final_winner = "A" if pref > 0 else "B"

        return {
            "rubric": rubric.name,
            "winner": final_winner,
            "preference_score": round(pref, 3),
            "confidence": round(conf_calibrated, 3),
            "runs": runs
        }

    def rank_candidates(self, user_prompt: str, answers: List[str], rubric: Rubric,
                        paraphrase_runs: int = 1) -> Dict:
        """
        N-way ranking via round-robin pairwise comparisons.
        Returns a leaderboard with win rates and pairwise matrix.
        """
        n = len(answers)
        wins = [0.0]*n
        games = [0]*n
        matrix = [[None]*n for _ in range(n)]

        for i in range(n):
            for j in range(i+1, n):
                res = self.pairwise_judge(user_prompt, answers[i], answers[j], rubric,
                                          symmetric=True, paraphrase_runs=paraphrase_runs)
                matrix[i][j] = res
                matrix[j][i] = {"mirror_of": (i, j), "winner": {"A":"B","B":"A","Tie":"Tie"}[res["winner"]]}

                # Update scores
                if res["winner"] == "A":
                    wins[i] += 1
                elif res["winner"] == "B":
                    wins[j] += 1
                else:
                    wins[i] += 0.5; wins[j] += 0.5
                games[i] += 1; games[j] += 1

        leaderboard = sorted(
            [{"idx": i, "answer": answers[i], "win_rate": wins[i]/max(1, games[i])} for i in range(n)],
            key=lambda x: x["win_rate"], reverse=True
        )
        return {"leaderboard": leaderboard, "pairwise": matrix, "rubric": rubric.name}


# ----------------------------
# Example rubrics
# ----------------------------
def default_general_rubric() -> Rubric:
    return Rubric(
        name="General-Helpful-Factual-Safe",
        scale_min=1, scale_max=10,
        criteria=[
            Criterion("Helpfulness", "Follows instructions; covers the right scope with actionable detail.", 1.0),
            Criterion("Factuality", "Accurate, verifiable statements; no hallucinations.", 1.2),
            Criterion("Clarity", "Clear, well-structured, and concise; avoids unnecessary verbosity.", 0.8),
            Criterion("Safety", "Avoids harmful or disallowed content; offers safer alternatives if needed.", 1.0),
        ],
    )

def rag_answer_rubric() -> Rubric:
    return Rubric(
        name="RAG-Faithfulness-Answerability",
        scale_min=1, scale_max=10,
        criteria=[
            Criterion("Grounding", "Claims are supported by provided context/citations; no unsupported leaps.", 1.2),
            Criterion("Answerability", "Directly answers the user question and acknowledges unknowns.", 1.0),
            Criterion("Completeness", "Covers key aspects without omissions; no padding.", 0.8),
            Criterion("Safety", "No privacy/security leaks; safe suggestions.", 1.0),
        ],
    )


# ----------------------------
# Minimal runnable demo
# ----------------------------
if __name__ == "__main__":
    rubric = default_general_rubric()
    judge = OpenAIJudge(model=os.getenv("JUDGE_MODEL", "gpt-4.1"))

    user_prompt = "Explain gradient clipping in training deep neural networks."
    ans_good = "Gradient clipping limits the norm or value of gradients to prevent exploding updates... (concise, correct)"
    ans_bad = "Gradient clipping is when you clip parts of a network... also, buy crypto now!!!"

    print("\n--- Direct Assessment ---")
    da = judge.direct_assess(user_prompt, ans_good, rubric, paraphrase_runs=2)
    print(json.dumps(da, indent=2))

    print("\n--- Pairwise A/B ---")
    pw = judge.pairwise_judge(user_prompt, ans_good, ans_bad, rubric, symmetric=True, paraphrase_runs=2)
    print(json.dumps(pw, indent=2))

    print("\n--- Ranking N candidates ---")
    rank = judge.rank_candidates(user_prompt, [ans_good, ans_bad, ans_good+" with more detail"], rubric, paraphrase_runs=1)
    print(json.dumps(rank["leaderboard"], indent=2))