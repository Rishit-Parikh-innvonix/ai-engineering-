import html
import re


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def truncate(text: str, max_length: int) -> str:
    return text if len(text) <= max_length else f"{text[: max_length - 1].rstrip()}…"


def strip_html(fragment: str) -> str:
    """Hacker News comments come back as HTML fragments (<p>, <a>, <i>, entities like &#x27;)."""
    return collapse_whitespace(html.unescape(re.sub(r"<[^>]*>", " ", fragment)))
