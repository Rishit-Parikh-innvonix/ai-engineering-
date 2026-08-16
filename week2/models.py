import logging

from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APIError, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

import config

logger = logging.getLogger("week2.models")

DEFAULT_HEADERS = {
    "HTTP-Referer": "https://localhost",
    "X-Title": "AI Engineering Week 2 Assignment",
}


def get_chat_model(model_name, temperature=0.7, top_p=1.0, max_tokens=None, streaming=False):
    return ChatOpenAI(
        model=model_name,
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        streaming=streaming,
        default_headers=DEFAULT_HEADERS,
    )


def get_streaming_chat_model(model_name, temperature=0.7):
    """Same model, with streaming=True so the provider sends the response token-by-token
    over its own SSE connection instead of one chunk at the end (see app.py's astream loop)."""
    return get_chat_model(model_name, temperature=temperature, streaming=True)


def get_json_mode_chat_model(model_name, temperature=0.2):
    """response_format=json_object (Level 1 structured output, see LEARNING_NOTES.txt) - tells
    the backend the reply must be valid JSON syntax. Combined in extraction.py with a
    PydanticOutputParser (Level 4) that checks the JSON actually matches our schema."""
    return ChatOpenAI(
        model=model_name,
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        temperature=temperature,
        model_kwargs={"response_format": {"type": "json_object"}},
        default_headers=DEFAULT_HEADERS,
    )


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((APIConnectionError, RateLimitError, APIError)),
)
async def ainvoke_with_retry(chat, messages):
    """Transport-level retry: exponential backoff on network errors / 429s / provider 5xxs.
    Separate from extraction.py's schema-validation retry loop - this one retries the same
    call because the request never reached (or never came back from) the model; that one
    retries because the model answered but the answer didn't parse."""
    return await chat.ainvoke(messages)


def extract_token_usage(ai_message):
    usage = getattr(ai_message, "usage_metadata", None)
    if usage:
        return {
            "prompt_tokens": usage.get("input_tokens"),
            "completion_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
    metadata_usage = ai_message.response_metadata.get("token_usage")
    if metadata_usage:
        return metadata_usage
    return None


def estimate_tokens(text):
    return max(1, len(text) // 4)


def estimate_cost_usd(model_name, prompt_tokens, completion_tokens):
    pricing = config.MODEL_PRICING_USD_PER_MILLION.get(model_name, {"input": 0.0, "output": 0.0})
    cost = (prompt_tokens / 1_000_000) * pricing["input"] + (completion_tokens / 1_000_000) * pricing["output"]
    return round(cost, 6)
