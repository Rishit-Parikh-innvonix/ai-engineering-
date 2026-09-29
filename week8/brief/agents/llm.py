import asyncio
import re
from typing import Awaitable, Callable, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

T = TypeVar("T")


async def with_retries(task: Callable[[], Awaitable[T]], attempts: int = 3) -> T:
    """Free-tier models fail in ways a paid API rarely does: 429 rate limits (several agents call
    the model at once), and occasional malformed responses. Both are transient, so each model call
    gets a few attempts with a growing pause."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await task()
        except Exception as error:  # noqa: BLE001 - any failure of one attempt is retried
            last_error = error
            if attempt < attempts:
                await asyncio.sleep(1.0 * attempt)
    assert last_error is not None
    raise last_error


def _strip_code_fence(text: str) -> str:
    """Some models wrap markdown in a ```markdown fence even when told not to."""
    return re.sub(r"\n```$", "", re.sub(r"^```[a-z]*\n", "", text, flags=re.IGNORECASE)).strip()


def _reply_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in content)
    return str(content)


async def ask_for(model: ChatOpenAI, system: str, human: str, parse: Callable[[str], T]) -> T:
    """Asks for plain text and runs `parse` on it, inside the retry loop: a reply that can't be
    parsed counts as a failed attempt and is asked for again, exactly like a network error would be.
    Plain text is the more dependable route on weak models than function calling, which sometimes
    returns a tool call with no arguments at all (seen live in the relevance grader)."""

    async def attempt() -> T:
        reply = await model.ainvoke([SystemMessage(content=system), HumanMessage(content=human)])
        cleaned = _strip_code_fence(_reply_text(reply.content).strip())
        if not cleaned:
            raise RuntimeError("The model returned an empty response.")
        return parse(cleaned)

    return await with_retries(attempt)


async def ask_for_text(model: ChatOpenAI, system: str, human: str) -> str:
    return await ask_for(model, system, human, lambda text: text)
