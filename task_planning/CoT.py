from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, validator
import textwrap

# ----------------------------
# 1) Data models for CoT exemplars and output schema
# ----------------------------

class CoTExample(BaseModel):
    """One few-shot exemplar."""
    question: str = Field(..., description="User question / problem.")
    # Keep internal reasoning minimal or redacted to avoid leaking chain-of-thought.
    # In production you can store richer internal notes here privately.
    reasoning_redacted: str = Field(
        default="(internal reasoning steps omitted)",
        description="Short placeholder instead of full reasoning."
    )
    final_answer: str = Field(..., description="The final answer for the exemplar.")

    def render(self) -> str:
        """Render an exemplar as instructions for the model—showing style but not revealing full CoT."""
        return textwrap.dedent(f"""
        Q: {self.question}
        (Model thinks step-by-step internally.)
        A: {self.final_answer}
        """).strip()


class OutputSchema(BaseModel):
    """
    What we expect the model to return.
    - final_answer: mandatory
    - brief_justification: optional, concise (not a full CoT)
    """
    final_answer: str
    brief_justification: Optional[str] = Field(
        default=None,
        description="One or two short sentences; do not include full reasoning steps."
    )

    @validator("brief_justification")
    def keep_justification_short(cls, v):
        if v and len(v.split()) > 60:
            raise ValueError("Justification is too long—keep it concise (<= ~60 words).")
        return v


# ----------------------------
# 2) Prompt configuration & builder
# ----------------------------

class CoTPromptConfig(BaseModel):
    task_instruction: str = Field(
        default="Solve the problem. Think step-by-step internally. "
                "Return only JSON with 'final_answer' and an optional 'brief_justification' "
                "(concise; no detailed steps).",
        description="High-level instruction for the model."
    )
    output_format_hint: str = Field(
        default=textwrap.dedent("""
        Output JSON only, like:
        {
          "final_answer": "...",
          "brief_justification": "..."
        }
        """).strip(),
        description="How the model should format its output."
    )


class CoTPrompt(BaseModel):
    config: CoTPromptConfig
    examples: List[CoTExample] = Field(default_factory=list)
    user_question: str

    def render(self) -> str:
        parts = []

        # Task instruction
        parts.append(self.config.task_instruction)

        # Few-shot exemplars
        if self.examples:
            parts.append("### Examples")
            for i, ex in enumerate(self.examples, start=1):
                parts.append(f"Example {i}:\n{ex.render()}")

        # Output format
        parts.append("### Output Format\n" + self.config.output_format_hint)

        # The new question
        parts.append("### Your Turn\nQ: " + self.user_question)

        # Gentle reminder about not exposing chain-of-thought
        parts.append("(Think step-by-step internally. Return JSON only.)")

        return "\n\n".join(parts)


# ----------------------------
# 3) Example usage (with a mock LLM call)
# ----------------------------

def mock_llm_call(prompt: str) -> str:
    """
    Replace this with your actual LLM client call.
    For demo, we return a plausible, concise JSON (no chain-of-thought).
    """
    # This mock is purely illustrative.
    return """
    {
      "final_answer": "12",
      "brief_justification": "Combine the red and blue marbles and count totals to reach 12."
    }
    """.strip()


def main():
    # Build a couple of few-shot exemplars.
    # Keep reasoning hidden/abridged—final answers are shown to teach the output style.
    exemplars = [
        CoTExample(
            question="A box has 2 red pens and 3 blue pens. If Sam adds 4 more blue pens, how many pens now?",
            reasoning_redacted="(omitted)",
            final_answer="9"
        ),
        CoTExample(
            question="There are 5 apples. You buy 2 more and eat 1. How many remain?",
            reasoning_redacted="(omitted)",
            final_answer="6"
        ),
    ]

    # Configure the prompt
    config = CoTPromptConfig()
    user_question = "A jar has 7 red marbles and 5 blue marbles. You add 0 red and 0 blue. How many marbles now?"

    prompt = CoTPrompt(
        config=config,
        examples=exemplars,
        user_question=user_question
    ).render()

    print("----- FEW-SHOT PROMPT -----")
    print(prompt)
    print("---------------------------\n")

    # Call your LLM (mocked here)
    raw = mock_llm_call(prompt)
    print("----- RAW MODEL OUTPUT -----")
    print(raw)
    print("----------------------------\n")

    # Parse with Pydantic to enforce structure & keep outputs concise
    parsed = OutputSchema.model_validate_json(raw)
    print("Parsed result:", parsed.model_dump())


if __name__ == "__main__":
    main()