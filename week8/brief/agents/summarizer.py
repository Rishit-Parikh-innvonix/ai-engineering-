from langchain_openai import ChatOpenAI

from brief.agents.llm import ask_for_text
from brief.logic import render_sources_for_prompt
from brief.types import NumberedSource

SYSTEM_PROMPT = (
    "You are the summarization agent of a research assistant. You are given numbered sources and a "
    "list of research sub-questions. For each sub-question, write the heading 'Q: <the sub-question>' "
    "followed by at most 4 short bullet points that answer it. Rules:\n"
    "- Use ONLY information stated in the sources. Never add facts from your own memory.\n"
    "- Every bullet must end with the number of the source it came from, written like [3]. If a "
    "bullet draws on two sources, write [3][5].\n"
    "- If the sources say nothing useful for a sub-question, write 'No information found.' under it.\n"
    "Write in English."
)


def create_summarizer(model: ChatOpenAI):
    async def summarize(*, topic: str, sub_questions: list[str], library: list[NumberedSource]) -> str:
        questions = "\n".join(f"{i}. {q}" for i, q in enumerate(sub_questions, start=1))
        return await ask_for_text(
            model,
            SYSTEM_PROMPT,
            f"Topic: {topic}\n\nSub-questions:\n{questions}\n\nSources:\n{render_sources_for_prompt(library)}",
        )

    return summarize
