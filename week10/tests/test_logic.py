import re

import pytest
from pydantic import ValidationError

from brief.logic import (
    build_extractive_draft,
    build_library,
    build_report,
    check_citations,
    extract_citation_ids,
    find_uncovered_sub_questions,
    keep_cited_sources,
    parse_relevant_numbers,
    parse_route_reply,
    previous_queries_for,
    slugify,
    strip_source_label,
)
from brief.sources.types import SourceDocument
from brief.tools.weather import CurrentWeather, GeocodedPlace, build_weather_document, describe_weather_code, pick_place
from brief.types import QueryPlan, RetrievedBatch, RouteDecision


def doc(kind, name, text="some text"):
    return SourceDocument(kind=kind, title=f"{kind} {name}", url=f"https://example.com/{kind}/{name}", text=text)


def batch(sub_question_index, documents, attempt=1):
    return RetrievedBatch(
        sub_question_index=sub_question_index,
        attempt=attempt,
        queries=QueryPlan(wikipedia="www", arxiv="aaa", hackernews="hhh"),
        documents=documents,
    )


def test_build_library_numbers_sources_by_content_not_by_which_branch_finished_first():
    batches = [
        batch(1, [doc("hackernews", "h1"), doc("wikipedia", "w1")]),
        batch(0, [doc("arxiv", "a0"), doc("wikipedia", "w0")]),
    ]
    forward = [s.title for s in build_library(batches)]
    reversed_ = [s.title for s in build_library(list(reversed(batches)))]

    assert forward == ["wikipedia w0", "arxiv a0", "wikipedia w1", "hackernews h1"]
    assert reversed_ == forward
    assert [s.id for s in build_library(batches)] == [1, 2, 3, 4]


def test_build_library_drops_duplicate_urls_and_documents_with_no_text():
    library = build_library(
        [
            batch(0, [doc("wikipedia", "same"), doc("arxiv", "blank", "   ")]),
            batch(1, [doc("wikipedia", "same"), doc("arxiv", "real")]),
        ]
    )
    assert [s.url for s in library] == ["https://example.com/wikipedia/same", "https://example.com/arxiv/real"]


def test_a_sub_question_counts_as_covered_once_it_has_one_usable_source():
    batches = [
        batch(0, [doc("wikipedia", "a"), doc("arxiv", "b")]),  # covered
        batch(1, [doc("wikipedia", "c")]),  # one source is enough
        batch(2, [doc("wikipedia", "d", ""), doc("arxiv", "e", "")]),  # sources with no text don't count
    ]
    assert find_uncovered_sub_questions(4, batches) == [2, 3]  # 3 has no batch at all


def test_a_retry_batch_can_rescue_a_sub_question_whose_first_attempt_found_nothing_usable():
    first_attempt = batch(0, [doc("wikipedia", "a", "")])
    retry = batch(0, [doc("arxiv", "b")], 2)
    assert find_uncovered_sub_questions(1, [first_attempt]) == [0]
    assert find_uncovered_sub_questions(1, [first_attempt, retry]) == []
    assert len(previous_queries_for([first_attempt, retry], 0)) == 2


def test_build_extractive_draft_keeps_every_excerpt_each_with_a_citation_that_passes_the_checker():
    library = build_library([batch(0, [doc("wikipedia", "w", "Wikipedia says X."), doc("arxiv", "a", "A paper says Y.")])])
    draft = build_extractive_draft(library)

    assert re.search(r"Wikipedia says X\. \[1\]", draft)
    assert re.search(r"A paper says Y\. \[2\]", draft)
    assert check_citations(draft, len(library)) == []


def test_check_citations_accepts_real_citations_in_all_three_styles():
    assert check_citations("A fact [1]. Another [2].", 3) == []
    assert check_citations("Both agree [1][2].", 3) == []
    assert check_citations("Both agree [1, 2].", 3) == []
    assert extract_citation_ids("x [1, 2] y [3]") == [1, 2, 3]


def test_check_citations_rejects_a_draft_with_no_citations():
    assert len(check_citations("Plenty of confident claims, no sources.", 3)) == 1


def test_check_citations_rejects_citations_that_point_to_sources_that_do_not_exist():
    problems = check_citations("Real [1] and [2], but invented [7] and [0].", 3)
    assert len(problems) == 1
    assert "[7]" in problems[0] and "[0]" in problems[0] and "[1] to [3]" in problems[0]


def test_check_citations_rejects_a_draft_leaning_on_one_source_when_several_exist():
    assert len(check_citations("One [1]. Same one [1].", 3)) == 1
    assert check_citations("One [1]. Same one [1].", 1) == []  # nothing else to cite


def test_build_report_generates_the_sources_list_from_the_real_library():
    library = build_library([batch(0, [doc("wikipedia", "w"), doc("arxiv", "a")])])
    report = build_report(topic="My topic", body="## Overview\n\nText [1][2].", library=library, run_notes=[], generated_on="2026-09-28")

    assert report.startswith("# My topic")
    assert "1. **Wikipedia**: [wikipedia w](https://example.com/wikipedia/w)" in report
    assert "2. **arXiv**: [arxiv a](https://example.com/arxiv/a)" in report
    assert "About this run" not in report


def test_keep_cited_sources_lists_only_cited_sources_and_renumbers_them_to_match_the_text():
    library = build_library([batch(0, [doc("wikipedia", "w"), doc("arxiv", "a"), doc("hackernews", "h"), doc("arxiv", "b")])])
    # ids: 1=wikipedia w, 2=arxiv a, 3=arxiv b, 4=hackernews h
    body, sources = keep_cited_sources("First [4]. Second [2][4]. Third [2, 3].", library)

    # Cited originals 2, 3, 4 become 1, 2, 3 (ascending), so [4]->[3], [2]->[1], [3]->[2].
    assert body == "First [3]. Second [1][3]. Third [1][2]."
    assert [(s.id, s.title) for s in sources] == [(1, "arxiv a"), (2, "arxiv b"), (3, "hackernews h")]


def test_keep_cited_sources_removes_a_citation_that_points_at_nothing():
    library = build_library([batch(0, [doc("wikipedia", "w"), doc("arxiv", "a")])])
    body, sources = keep_cited_sources("Real [1] and made up [42].", library)
    assert body == "Real [1] and made up ."
    assert len(sources) == 1


def test_build_report_hides_retrieved_but_unused_sources_and_says_how_many_it_left_out():
    library = build_library([batch(0, [doc("wikipedia", "w"), doc("arxiv", "a"), doc("hackernews", "h")])])
    report = build_report(topic="t", body="Only one [3].", library=library, run_notes=[], generated_on="2026-09-28")

    assert "Only one [1]." in report  # renumbered
    assert "1. **Hacker News**" in report
    assert "Wikipedia**" not in report
    assert "2 retrieved sources were not used in the brief and are not listed" in report


def test_parse_relevant_numbers_accepts_none_or_a_comma_list_and_rejects_rambling():
    assert parse_relevant_numbers("1, 4", 5) == [1, 4]
    assert parse_relevant_numbers(" 2 ", 5) == [2]
    assert parse_relevant_numbers("NONE", 5) == []
    assert parse_relevant_numbers("None.", 5) == []
    assert parse_relevant_numbers("7, 2, 2, 0", 3) == [2]  # invented / duplicate numbers ignored
    with pytest.raises(ValueError):
        parse_relevant_numbers("Result 2 is relevant because it mentions agents", 5)
    with pytest.raises(ValueError):
        parse_relevant_numbers("", 5)


def test_build_report_lists_run_notes_and_handles_an_empty_library():
    report = build_report(topic="t", body="body", library=[], run_notes=["arXiv was down"], generated_on="2026-09-28")
    assert "No sources are cited" in report
    assert "## About this run\n\n- arXiv was down" in report


def test_search_queries_must_contain_real_words_the_model_once_returned_colon_for_every_source():
    QueryPlan(wikipedia="LangChain", arxiv="LLM agent orchestration", hackernews="LangGraph vs LangChain")
    with pytest.raises(ValidationError):
        QueryPlan(wikipedia=": ", arxiv=": ", hackernews=": ")
    with pytest.raises(ValidationError):
        QueryPlan(wikipedia="LangGraph", arxiv=", ", hackernews=", ")
    with pytest.raises(ValidationError):
        QueryPlan(wikipedia="ab", arxiv="ok fine", hackernews="ok fine")  # too short


def test_strip_source_label_removes_a_source_name_the_model_echoed_into_its_own_query():
    assert strip_source_label("Wikipedia: LangGraph framework adoption") == "LangGraph framework adoption"
    assert strip_source_label("arXiv: LangChain agent comparison") == "LangChain agent comparison"
    assert strip_source_label("Hacker News: LangGraph vs LangChain debate") == "LangGraph vs LangChain debate"
    assert strip_source_label("LangGraph vs LangChain") == "LangGraph vs LangChain"  # untouched
    assert strip_source_label("Wikipedia history of Rome") == "Wikipedia history of Rome"  # no colon: a real word
    assert strip_source_label(None) == ""


def test_slugify_makes_safe_bounded_file_names():
    assert slugify("How do LangGraph agents differ from LangChain agents?") == "how-do-langgraph-agents-differ-from-langchain-agents"
    assert slugify("???") == "research-brief"
    assert len(slugify("x" * 200)) <= 60


# ---------- router reply parsing ----------


def test_parse_route_reply_reads_each_of_the_four_route_shapes():
    assert parse_route_reply("RESEARCH") == RouteDecision("research")
    assert parse_route_reply("  research.\n") == RouteDecision("research")
    assert parse_route_reply("UNSUPPORTED_LIVE") == RouteDecision("unsupported")
    assert parse_route_reply("WEATHER: Ahmedabad") == RouteDecision("weather", city="Ahmedabad")
    assert parse_route_reply("weather : New York, US") == RouteDecision("weather", city="New York, US")
    assert parse_route_reply("MIXED: Surat | why the monsoon happens in Gujarat") == RouteDecision(
        "mixed", city="Surat", research_topic="why the monsoon happens in Gujarat"
    )


def test_parse_route_reply_rambling_empty_or_unusable_replies_raise_so_the_model_is_asked_again():
    with pytest.raises(ValueError, match="Unexpected router reply"):
        parse_route_reply("I think this is a weather question")
    with pytest.raises(ValueError, match="Unexpected router reply"):
        parse_route_reply("WEATHER:")
    with pytest.raises(ValueError, match="unusable city"):
        parse_route_reply("WEATHER: 12")
    with pytest.raises(ValueError, match="Unexpected router reply|no research question"):
        parse_route_reply("MIXED: Surat")
    with pytest.raises(ValueError, match="no research question"):
        parse_route_reply("MIXED: Surat | ")


def test_parse_route_reply_accepts_non_english_city_names():
    assert parse_route_reply("WEATHER: 東京") == RouteDecision("weather", city="東京")
    assert parse_route_reply("WEATHER: São Paulo") == RouteDecision("weather", city="São Paulo")


# ---------- live-tool readings in the library ----------


def test_build_library_a_weather_reading_is_numbered_first_and_belongs_to_no_sub_question():
    library = build_library([batch(0, [doc("wikipedia", "a")])], [doc("weather", "now")])
    assert [(s.id, s.kind, s.sub_question_index) for s in library] == [(1, "weather", -1), (2, "wikipedia", 0)]


# ---------- weather tool: pure parts ----------


def place(**over):
    fields = dict(name="Ahmedabad", latitude=23.02, longitude=72.58, country="India", country_code="IN", admin1="Gujarat")
    return GeocodedPlace(**{**fields, **over})


def test_pick_place_takes_the_top_result_or_the_one_matching_a_country_or_region_qualifier():
    india = place()
    pakistan = place(country="Pakistan", country_code="PK", admin1="Khyber Pakhtunkhwa")
    assert pick_place([india, pakistan], "") is india
    assert pick_place([india, pakistan], "Pakistan") is pakistan
    assert pick_place([india, pakistan], "pk") is pakistan
    assert pick_place([india, pakistan], "Atlantis") is india  # unknown qualifier: top result
    assert pick_place([], "India") is None


def test_describe_weather_code_maps_wmo_codes_and_admits_unknown_ones():
    assert describe_weather_code(1) == "mainly clear"
    assert describe_weather_code(95) == "thunderstorm"
    assert "unrecognised condition (WMO code 1234)" in describe_weather_code(1234)


def test_build_weather_document_every_number_in_the_text_comes_from_the_api_response():
    weather = CurrentWeather.model_validate(
        {
            "timezone": "Asia/Kolkata",
            "current": {
                "time": "2026-09-29T13:00",
                "temperature_2m": 34.0,
                "apparent_temperature": 36.1,
                "relative_humidity_2m": 42,
                "precipitation": 0.0,
                "wind_speed_10m": 3.2,
                "weather_code": 1,
            },
            "current_units": {"temperature_2m": "°C", "relative_humidity_2m": "%", "precipitation": "mm", "wind_speed_10m": "km/h"},
        }
    )
    document = build_weather_document(place(), weather, "https://api.example.com/forecast")

    assert document.kind == "weather"
    assert document.title == "Current weather in Ahmedabad, Gujarat, India"
    assert document.url == "https://api.example.com/forecast"
    assert "Observed: 2026-09-29 13:00 local time (Asia/Kolkata)" in document.text
    assert "Conditions: mainly clear" in document.text
    assert "Temperature: 34 °C (feels like 36.1 °C)" in document.text  # 34.0 prints as 34, like the API's JSON
    assert "Relative humidity: 42%" in document.text
    assert "Wind speed: 3.2 km/h" in document.text
