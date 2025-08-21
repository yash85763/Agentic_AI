# pip install langchain langchain-openai pydantic>=2
from collections import Counter
from typing import Optional
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


# --------------------------
# 1) Define structured output schema
# --------------------------
class CoTOutput(BaseModel):
    final_answer: str = Field(..., description="The final numeric/text answer only")
    brief_justification: Optional[str] = Field(
        None, description="One short sentence (no step-by-step reasoning)."
    )

parser = PydanticOutputParser(pydantic_object=CoTOutput)


# --------------------------
# 2) Build prompt
# --------------------------
system_msg = (
    "Solve the problem. Think step by step internally but do NOT reveal reasoning. "
    "Return only JSON matching this schema:\n{format_instructions}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_msg),
    ("human", "{question}")
])


# --------------------------
# 3) LLM setup
# --------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=256)


# --------------------------
# 4) Self-consistent CoT runner
# --------------------------
def self_consistent_cot(question: str, n_samples: int = 5):
    answers = []

    for i in range(n_samples):
        chain = prompt | llm | parser
        result: CoTOutput = chain.invoke({
            "question": question,
            "format_instructions": parser.get_format_instructions()
        })
        answers.append(result.final_answer)

    # Majority vote
    majority = Counter(answers).most_common(1)[0][0]

    return {
        "all_answers": answers,
        "final_answer": majority
    }


# --------------------------
# 5) Example usage
# --------------------------
if __name__ == "__main__":
    q = "A train travels 120 km in 2 hours and then 60 km in 1 hour. What is the average speed overall?"
    result = self_consistent_cot(q, n_samples=7)

    print("All sampled answers:", result["all_answers"])
    print("Majority final answer:", result["final_answer"])
