import pytest

from brief.sources.arxiv import parse_feed
from brief.sources.hackernews import is_noise_thread
from brief.sources.http import user_agent
from brief.sources.text import collapse_whitespace, strip_html, truncate


def entry(id_, title, summary):
    return f"<entry><id>{id_}</id><title>{title}</title><summary>{summary}</summary></entry>"


def feed(entries):
    return f'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">{entries}</feed>'


def test_parse_feed_reads_several_entries_upgrades_links_to_https_and_tidies_whitespace():
    docs = parse_feed(
        feed(
            entry("http://arxiv.org/abs/1111.0001v1", "A   Title\n Split Over Lines", "  First\nsummary. ")
            + entry("http://arxiv.org/abs/2222.0002v1", "Second", "Second summary.")
        )
    )
    assert len(docs) == 2
    assert docs[0].model_dump() == {
        "kind": "arxiv",
        "title": "A Title Split Over Lines",
        "url": "https://arxiv.org/abs/1111.0001v1",
        "text": "First summary.",
    }


def test_parse_feed_handles_a_feed_with_exactly_one_entry():
    docs = parse_feed(feed(entry("http://arxiv.org/abs/3333.0003v1", "Only one", "Only summary.")))
    assert len(docs) == 1
    assert docs[0].title == "Only one"


def test_parse_feed_returns_an_empty_list_when_arxiv_finds_nothing():
    # The shape arXiv really sends for zero results (captured from a live request): feed metadata, no <entry>.
    zero_results = feed(
        "<id>https://arxiv.org/api/abc</id><title>arXiv Query: search_query=all:zzz</title>"
        "<updated>2026-09-28T16:42:56Z</updated><totalResults>0</totalResults>"
    )
    assert parse_feed(zero_results) == []


def test_parse_feed_rejects_a_response_that_is_not_an_arxiv_feed_instead_of_returning_nonsense():
    with pytest.raises(RuntimeError, match="Unexpected response format"):
        parse_feed("<html><body>Rate exceeded.</body></html>")
    with pytest.raises(RuntimeError, match="Unexpected response format"):
        parse_feed("this is not xml at all")


def test_is_noise_thread_filters_hiring_threads_and_moderator_removed_threads_not_real_discussions():
    assert is_noise_thread("Ask HN: Who wants to be hired? (September 2026)") is True
    assert is_noise_thread("Ask HN: Who is hiring? (July 2026)") is True
    assert is_noise_thread("[dead]") is True
    assert is_noise_thread("[flagged]") is True
    assert is_noise_thread("Show HN: LoopGain - stop agent loops") is False
    assert is_noise_thread(None) is False


def test_strip_html_turns_a_hacker_news_comment_fragment_into_plain_text():
    assert (
        strip_html('<p>It&#x27;s great &amp; fast<p>see <a href="https:&#x2F;&#x2F;x.com">this</a> &quot;link&quot;')
        == "It's great & fast see this \"link\""
    )


def test_truncate_and_collapse_whitespace_behave_at_their_edges():
    assert truncate("short", 10) == "short"
    assert truncate("abcdefghij", 5) == "abcd…"
    assert collapse_whitespace("  a \n\t b  ") == "a b"


def test_user_agent_names_a_contact_because_wikipedia_refuses_generic_ones(monkeypatch):
    monkeypatch.delenv("RESEARCH_BRIEF_CONTACT", raising=False)
    assert "https://" in user_agent()  # the default is a URL, which is what Wikimedia's robot policy looks for
    monkeypatch.setenv("RESEARCH_BRIEF_CONTACT", "https://github.com/someone/their-repo")
    assert "(https://github.com/someone/their-repo;" in user_agent()
    monkeypatch.setenv("RESEARCH_BRIEF_CONTACT", "   ")
    assert "https://example.org" in user_agent()  # blank falls back to the default
