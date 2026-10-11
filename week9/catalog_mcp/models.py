from langchain_openai import ChatOpenAI

from catalog_mcp import config

DEFAULT_HEADERS = {
    "HTTP-Referer": "https://localhost",
    "X-Title": "AI Engineering Week 9 Assignment",
}
REQUEST_TIMEOUT_S = 60


def get_cloud_chat_model(temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        model=config.CLOUD_MODEL,
        api_key=config.get_api_key(),
        base_url=config.OPENROUTER_BASE_URL,
        default_headers=DEFAULT_HEADERS,
        temperature=temperature,
        timeout=REQUEST_TIMEOUT_S,
        max_retries=0,
    )
