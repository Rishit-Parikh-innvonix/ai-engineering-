"""The deterministic decisions of the workflow, kept as plain functions with no AI and no network
so they can be tested exactly: how sources get numbered, when retrieval counts as "enough",
whether a draft's citations are real, and how the router's reply is read."""

import re

from brief.sources.text import truncate
from brief.sources.types import SourceDocument, SourceKind
from brief.types import NumberedSource, QueryPlan, RetrievedBatch, RouteDecision


class LIMITS:
    # A sub-question counts as covered once ONE relevant source turned up. Sources are already
    # filtered for relevance, so demanding two only triggered slow retries whose reworded queries
    # drifted off-topic (seen live) - retries are for "found nothing relevant", not "found little".
    min_sources_per_question = 1
    # Initial retrieval plus up to two retries with rewritten queries.
    max_retrieval_rounds = 3
    # First draft plus up to two redrafts after a failed citation check.
    max_drafts = 3


KIND_ORDER: dict[SourceKind, int] = {"weather": 0, "wikipedia": 1, "arxiv": 2, "hackernews": 3}
KIND_LABEL: dict[SourceKind, str] = {
    "wikipedia": "Wikipedia",
    "arxiv": "arXiv",
    "hackernews": "Hacker News",
    "weather": "Open-Meteo weather",
}


def build_library(batches: list[RetrievedBatch], tool_documents: list[SourceDocument] | None = None) -> list[NumberedSource]:
    """Flattens every batch into one numbered list. Parallel branches finish in whatever order the
    network allows, so the order is fixed by content (sub-question, then source type, then rank)
    instead of arrival time - the same run always produces the same [1], [2], [3] numbering.
    Live-tool readings (weather) come first and belong to no sub-question (index -1)."""
    candidates = [(document, -1, 0, position) for position, document in enumerate(tool_documents or [])]
    for batch in batches:
        for position, document in enumerate(batch.documents):
            candidates.append((document, batch.sub_question_index, batch.attempt, position))
    candidates.sort(key=lambda c: (c[1], KIND_ORDER[c[0].kind], c[2], c[3]))

    seen_urls: set[str] = set()
    library: list[NumberedSource] = []
    for document, sub_question_index, _attempt, _position in candidates:
        if not document.text.strip() or document.url in seen_urls:
            continue
        seen_urls.add(document.url)
        library.append(NumberedSource(**document.model_dump(), id=len(library) + 1, sub_question_index=sub_question_index))
    return library


def _count_usable_sources(batches: list[RetrievedBatch], sub_question_index: int) -> int:
    urls = {
        document.url
        for batch in batches
        if batch.sub_question_index == sub_question_index
        for document in batch.documents
        if document.text.strip()
    }
    return len(urls)


def find_uncovered_sub_questions(sub_question_count: int, batches: list[RetrievedBatch]) -> list[int]:
    return [i for i in range(sub_question_count) if _count_usable_sources(batches, i) < LIMITS.min_sources_per_question]


def previous_queries_for(batches: list[RetrievedBatch], sub_question_index: int) -> list[QueryPlan]:
    """The queries already tried for a sub-question, so a retry can be told to word things differently."""
    return [batch.queries for batch in batches if batch.sub_question_index == sub_question_index]


# Accepts [1], [1][2] and also [1, 2], since models write all three.
CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def extract_citation_ids(text: str) -> list[int]:
    return [int(part.strip()) for match in CITATION_PATTERN.finditer(text) for part in match.group(1).split(",")]


def check_citations(draft: str, source_count: int) -> list[str]:
    """Checks that the writing agent only cites sources that exist. This is the guard that makes
    the output trustworthy: the AI writes the prose, but plain code decides whether its citations
    are real. Returns human-readable problems (also fed back to the writer); empty means clean."""
    cited = extract_citation_ids(draft)
    if not cited:
        return ["The draft contains no citations. Every factual claim must end with a citation such as [1]."]

    problems: list[str] = []
    is_valid = lambda i: 1 <= i <= source_count  # noqa: E731

    invalid = sorted({i for i in cited if not is_valid(i)}, key=cited.index)
    if invalid:
        problems.append(
            f"Citation(s) {', '.join(f'[{i}]' for i in invalid)} do not exist. "
            f"The only valid citations are [1] to [{source_count}]."
        )

    distinct_valid = {i for i in cited if is_valid(i)}
    if source_count >= 2 and len(distinct_valid) < 2:
        problems.append("The draft relies on a single source. Use at least two different sources.")
    return problems


def strip_source_label(query: str | None = "") -> str:
    """Seen live on a retry: the model echoed the source name into the query itself, producing
    searches like "Wikipedia: LangGraph framework adoption" that match worse than the plain words."""
    return re.sub(r"^\s*(wikipedia|arxiv|hacker\s?news)\s*[:\-–]\s*", "", query or "", flags=re.IGNORECASE).strip()


def parse_relevant_numbers(reply: str, count: int) -> list[int]:
    """Parses the grader's reply, which must be either "NONE" or a comma-separated list of result
    numbers ("1, 4"). Anything else raises, so the caller asks again instead of guessing what a
    rambling answer meant. Numbers outside 1..count are ignored (models invent them)."""
    cleaned = re.sub(r"[.\s]+$", "", reply.strip())
    if re.fullmatch(r"none", cleaned, re.IGNORECASE):
        return []
    if not re.fullmatch(r"\d+(\s*,\s*\d+)*", cleaned):
        raise ValueError(f'Unexpected relevance reply: "{cleaned[:60]}"')
    numbers = list(dict.fromkeys(int(n.strip()) for n in cleaned.split(",")))
    return [n for n in numbers if 1 <= n <= count]


def parse_route_reply(reply: str) -> RouteDecision:
    """Parses the router's reply, which must be one of:
        RESEARCH | UNSUPPORTED_LIVE | WEATHER: <city> | MIXED: <city> | <the rest of the question>
    Anything else raises, so the caller asks again instead of guessing. The city is validated here
    (real letters, sane length); whether it actually exists is the weather tool's job."""
    cleaned = re.sub(r"^[\"'`]+|[\"'`.]+$", "", reply.strip())
    if re.fullmatch(r"research", cleaned, re.IGNORECASE):
        return RouteDecision("research")
    if re.fullmatch(r"unsupported_live", cleaned, re.IGNORECASE):
        return RouteDecision("unsupported")

    match = re.fullmatch(r"(weather|mixed)\s*:\s*([^|]+?)\s*(?:\|\s*(.*))?", cleaned, re.IGNORECASE)
    if not match:
        raise ValueError(f'Unexpected router reply: "{cleaned[:60]}"')

    city = match.group(2).strip()
    # [^\W\d_] is "a letter in any language" (Python's re has no \p{L}).
    if len(city) < 2 or len(city) > 80 or not re.search(r"[^\W\d_]{2}", city):
        raise ValueError(f'The router gave an unusable city: "{city[:40]}"')
    if match.group(1).lower() == "weather":
        return RouteDecision("weather", city=city)

    research_topic = (match.group(3) or "").strip()
    if len(research_topic) < 3:
        raise ValueError("The router chose MIXED but gave no research question after the city.")
    return RouteDecision("mixed", city=city, research_topic=research_topic)


def keep_cited_sources(body: str, library: list[NumberedSource]) -> tuple[str, list[NumberedSource]]:
    """Keeps only the sources the text actually cites, and renumbers them 1..n in order of first
    number, rewriting the [n] markers to match. Retrieved-but-uncited sources are not listed: a
    reader should see the sources behind the claims, not everything the search happened to return.
    A citation to a source that doesn't exist is removed rather than left pointing at nothing."""
    by_id = {source.id: source for source in library}
    cited = sorted({i for i in extract_citation_ids(body) if i in by_id})
    new_id_for = {old: new for new, old in enumerate(cited, start=1)}

    def rewrite(match: re.Match[str]) -> str:
        ids = (new_id_for.get(int(part.strip())) for part in match.group(1).split(","))
        return "".join(f"[{i}]" for i in ids if i is not None)

    sources = [by_id[old].model_copy(update={"id": new_id_for[old]}) for old in cited]
    return CITATION_PATTERN.sub(rewrite, body), sources


def build_extractive_draft(library: list[NumberedSource]) -> str:
    """Used when the summarizing or writing agent fails after all its retries (free models time
    out). The retrieval work is the expensive part, so instead of throwing it away the report
    still lists every relevant excerpt, unedited. Each line carries its own [n], so it passes the
    citation check."""
    excerpts = "\n".join(f"- **{s.title}**: {truncate(s.text, 300)} [{s.id}]" for s in library)
    return (
        "## Retrieved excerpts\n\nThe writing step did not complete, so these are the relevant excerpts "
        f"that were retrieved, unedited.\n\n{excerpts}"
    )


def render_sources_for_prompt(library: list[NumberedSource]) -> str:
    return "\n\n".join(f"[{s.id}] {KIND_LABEL[s.kind]} - {s.title}\n{s.text}" for s in library)


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60].rstrip("-")
    return slug or "research-brief"


def build_report(*, topic: str, body: str, library: list[NumberedSource], run_notes: list[str], generated_on: str) -> str:
    """The Sources section is generated here, from the real fetched data, never by the AI - so
    every link in the report is one that was actually retrieved, and only sources the text cites
    are listed. `generated_on` is a date, optionally with a time (live data is a snapshot)."""
    body, sources = keep_cited_sources(body, library)
    notes = list(run_notes)
    uncited = len(library) - len(sources)
    if uncited > 0:
        plural = uncited != 1
        notes.append(
            f"{uncited} retrieved source{'s were' if plural else ' was'} not used in the brief "
            f"and {'are' if plural else 'is'} not listed."
        )

    source_list = (
        "\n".join(f"{s.id}. **{KIND_LABEL[s.kind]}**: [{s.title}]({s.url})" for s in sources)
        if sources
        else "_No sources are cited._"
    )
    sections = [
        f"# {topic}",
        f"_Research brief generated {generated_on}. Each [n] in the text refers to the numbered list under Sources._",
        body.strip(),
        f"## Sources\n\n{source_list}",
    ]
    if notes:
        sections.append("## About this run\n\n" + "\n".join(f"- {note}" for note in notes))
    return "\n\n".join(sections) + "\n"
