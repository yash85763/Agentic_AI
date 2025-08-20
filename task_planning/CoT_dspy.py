# dspy_cot_fewshot.py
from __future__ import annotations
import os
import dspy

# ------------------------------------------
# 1) Configure the LM (OpenAI shown; swap if desired)
# ------------------------------------------
# For OpenAI:
lm = dspy.OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), max_tokens=512)
# For a local model via Ollama, comment the above and uncomment:
# lm = dspy.OllamaLocal(model="llama3", max_tokens=512)

dspy.settings.configure(lm=lm)


# ------------------------------------------
# 2) Define a Signature with a private rationale
# ------------------------------------------
class SolveWordProblem(dspy.Signature):
    """Solve an arithmetic word problem. Think step-by-step internally."""
    question: str
    # The rationale is the CoT; we won't expose it in the returned API.
    rationale: dspy.OutputField(desc="Internal chain-of-thought (do not reveal)")
    answer: dspy.OutputField(desc="Final numeric answer only")


# ------------------------------------------
# 3) A small DSPy Module that uses CoT
# ------------------------------------------
class CoTSolver(dspy.Module):
    def __init__(self):
        super().__init__()
        # ChainOfThought encourages the model to produce a rationale + answer
        self.solve = dspy.ChainOfThought(SolveWordProblem)

    def forward(self, question: str):
        pred = self.solve(question=question)
        # Return the final answer; keep rationale internal.
        # You can log pred.rationale to your observability system if desired.
        return dspy.Prediction(answer=pred.answer)


# ------------------------------------------
# 4) Few-shot training data (with rationales)
#    - These examples teach the structure and solution style.
#    - They won't be shown to end users.
# ------------------------------------------
trainset = [
    dspy.Example(
        question="A box has 2 red pens and 3 blue pens. If you add 4 more blue pens, how many pens now?",
        rationale="Total pens = 2 + 3 + 4 = 9.",
        answer="9"
    ).with_inputs("question"),

    dspy.Example(
        question="There are 5 apples. You buy 2 more and eat 1. How many remain?",
        rationale="Start with 5, add 2 → 7, eat 1 → 6.",
        answer="6"
    ).with_inputs("question"),

    dspy.Example(
        question="A jar has 7 red marbles and 5 blue marbles. You add 0. How many marbles now?",
        rationale="Just sum existing: 7 + 5 = 12.",
        answer="12"
    ).with_inputs("question"),
]


# ------------------------------------------
# 5) Teleprompting: build a compact few-shot prompt automatically
# ------------------------------------------
# This optimizer picks/optimizes few-shot exemplars for the module.
optimizer = dspy.BootstrapFewShotWithRandomSearch(
    metric=lambda gold, pred, _: str(gold["answer"]).strip() == str(pred["answer"]).strip(),
    max_bootstrapped_demos=6,   # upper bound examples to consider
    num_trials=6                # random search trials for better selection
)

# Compile (train) the module into a prompt-optimized program.
compiled_solver: CoTSolver = optimizer.compile(CoTSolver(), trainset=trainset)


# ------------------------------------------
# 6) Inference: ask a new question (rationale stays private)
# ------------------------------------------
if __name__ == "__main__":
    user_question = "A train travels 120 km in 2 hours and then 60 km in 1 hour. What is the average speed overall?"
    pred = compiled_solver(question=user_question)

    # Expose only the final answer to the user
    print("Q:", user_question)
    print("Final Answer:", pred.answer)

    # If you want to inspect the internal reasoning for debugging:
    # WARNING: Don't show this to end users in production.
    # You can get the rationale by temporarily modifying CoTSolver.forward
    # to return it, or by instrumenting dspy to log `pred.rationale`.
