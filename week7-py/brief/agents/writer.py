from langchain_openai import ChatOpenAI

from brief.agents.llm import ask_for_text
from brief.types import NumberedSource

SYSTEM_PROMPT = (
    "You are the final-answer agent of a research assistant. Write a research brief in markdown "
    "from the provided notes. Rules:\n"
    "- Use ONLY facts from the notes. Never add facts, names or numbers from your own memory.\n"
    "- End every factual sentence with its source number, written like [2]. For two sources write "
    "[2][4] - never [2, 4]. Only use numbers that appear in the notes.\n"
    "- Do not write a top-level title (it is added for you) and do not write a Sources list "
    "(it is generated for you).\n"
    "- Use these sections, in this order: '## Overview' (2-3 sentences), '## Key points' (bullets), "
    "'## Limitations' (one or two sentences on what the notes did not cover).\n"
    "Write in English."
)


def create_writer(model: ChatOpenAI):
    async def write(
        *, topic: str, notes: str, library: list[NumberedSource], problems: list[str], previous_draft: str
    ) -> str:
        fix = ""
        if problems:
            bullets = "\n- ".join(problems)
            fix = f"\n\nYour previous draft was rejected for these problems - fix them:\n- {bullets}\n\nPrevious draft:\n{previous_draft}"
        return await ask_for_text(
            model,
            SYSTEM_PROMPT,
            f"Topic: {topic}\n\nThere are {len(library)} sources, numbered [1] to [{len(library)}].\n\nNotes:\n{notes}{fix}",
        )

    return write
