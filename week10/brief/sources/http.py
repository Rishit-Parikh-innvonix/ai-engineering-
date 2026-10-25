import os
from typing import TypeVar
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ValidationError

# Wikimedia's robot policy (https://w.wiki/4wJS) rejects generic User-Agents with HTTP 403 - seen
# live: the same request that works with a User-Agent naming a contact URL is refused without one.
# The contact is read from RESEARCH_BRIEF_CONTACT (a URL or email you choose, e.g. your repo's URL)
# so no personal details are hard-coded; the default is a neutral project URL, which is enough to
# pass the check but is not a real way to reach anyone - set your own if you run this a lot.
DEFAULT_CONTACT = "https://example.org/research-brief"


def user_agent() -> str:
    contact = os.environ.get("RESEARCH_BRIEF_CONTACT", "").strip() or DEFAULT_CONTACT
    return f"research-brief/1.0 ({contact}; educational project)"


REQUEST_TIMEOUT_S = 15

T = TypeVar("T", bound=BaseModel)


def _host(url: str) -> str:
    return urlparse(url).hostname or url


async def http_get_text(url: str) -> str:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S, follow_redirects=True) as client:
        response = await client.get(url, headers={"User-Agent": user_agent()})
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code} from {_host(url)}")
    return response.text


async def http_get_json(url: str, schema: type[T]) -> T:
    """External APIs are the one place data enters this program untrusted, so every response is
    validated against a schema instead of being trusted."""
    text = await http_get_text(url)
    try:
        return schema.model_validate_json(text)
    except ValidationError as error:
        raise RuntimeError(f"Unexpected response format from {_host(url)}") from error
