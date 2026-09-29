from langchain_openai import ChatOpenAI

from brief import config

DEFAULT_HEADERS = {
    "HTTP-Referer": "https://localhost",
    "X-Title": "AI Engineering Week 7 Assignment (Python)",
}
# The summarizer and writer generate a few hundred words, which on a busy free-tier model can
# take far longer than the short planner/grader calls.
REQUEST_TIMEOUT_S = 60


def get_cloud_chat_model(temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        model=config.CLOUD_MODEL,
        api_key=config.get_api_key(),
        base_url=config.OPENROUTER_BASE_URL,
        default_headers=DEFAULT_HEADERS,
        temperature=temperature,
        timeout=REQUEST_TIMEOUT_S,
        # Retrying is handled in one place (with_retries in agents/llm.py), so the client doesn't
        # silently retry underneath it and multiply the waiting time.
        max_retries=0,
    )
