import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from brief import tracing
from brief.agents.llm import with_retries

MAX_SUB_QUESTIONS = 4

SYSTEM_PROMPT = (
    "You are the planning agent of a research assistant. Break the user's topic into 3 focused "
    "sub-questions that together cover it well: for example background and definitions, how it "
    "works or what caused it, and current state, evidence or debate. Each sub-question must make "
    "sense on its own (it will be researched separately, without seeing the others). "
    "Write in English."
)


class PlanSchema(BaseModel):
    sub_questions: list[str] = Field(description="3 focused, self-contained sub-questions.")


def create_planner(model: ChatOpenAI):
    # function_calling rather than the default json_schema mode: it is the structured-output route
    # that free OpenRouter models support most consistently.
    planner = model.with_structured_output(PlanSchema, method="function_calling")

    async def plan(topic: str) -> list[str]:
        async def attempt():
            started = time.monotonic()
            result = await planner.ainvoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=topic)])
            # Same limitation as retrieval.py's query writer: with_structured_output() discards the
            # raw reply, so no token usage is available here, only latency.
            tracing.record_model_call("plan", None, time.monotonic() - started)
            return result

        result = await with_retries(attempt)
        # Models don't always respect "exactly 3", so clean up instead of trusting the count.
        cleaned = list(dict.fromkeys(q.strip() for q in result.sub_questions if q.strip()))[:MAX_SUB_QUESTIONS]
        if not cleaned:
            raise RuntimeError("The planning agent returned no sub-questions.")
        return cleaned

    return plan
