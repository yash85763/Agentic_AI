# pip install langchain langchain-core langchain-openai pydantic>=2
from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# ----------------------------
# 1) Define a concise, structured output (no CoT leakage)
# ----------------------------
class OutputSchema(BaseModel):
    final_answer: str = Field(..., description="The final answer only.")
    brief_justification: Optional[str] = Field(
        default=None,
        description="One or two sentences max—no step-by-step details."
    )

parser = PydanticOutputParser(pydantic_object=OutputSchema)

# ----------------------------
# 2) Few-shot examples (answers only; do not include chain-of-thought)
# ----------------------------
examples = [
    {
        "question": "A box has 2 red pens and 3 blue pens. If you add 4 more blue pens, how many pens now?",
        "answer": '{"final_answer": "9", "brief_justification": "Sum existing pens and added pens."}'
    },
    {
        "question": "There are 5 apples. You buy 2 more and eat 1. How many remain?",
        "answer": '{"final_answer": "6", "brief_justification": "5 + 2 - 1."}'
    },
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{question}"),
    ("ai", "{answer}"),
])

few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# ----------------------------
# 3) System instruction: think internally; return only structured JSON
# ----------------------------
system_template = (
    "Solve the problem. Think step-by-step internally but DO NOT reveal your reasoning. "
    "Return only JSON that matches this schema:\n{format_instructions}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    few_shot,  # inserts the few-shot conversation turns
    ("human", "{user_question}"),
])

# ----------------------------
# 4) LLM + Chain
# ----------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chain = prompt | llm | parser

# ----------------------------
# 5) Run
# ----------------------------
if __name__ == "__main__":
    user_question = "A jar has 7 red marbles and 5 blue marbles. You add 0 red and 0 blue. How many marbles now?"

    result: OutputSchema = chain.invoke({
        "user_question": user_question,
        "format_instructions": parser.get_format_instructions()
    })

    # Parsed Pydantic object:
    print("Final Answer:", result.final_answer)
    print("Brief Justification:", result.brief_justification)
