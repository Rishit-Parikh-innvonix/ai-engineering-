import assert from "node:assert/strict";
import { test } from "node:test";

import {
  buildExtractiveDraft,
  buildLibrary,
  buildReport,
  checkCitations,
  extractCitationIds,
  findUncoveredSubQuestions,
  keepCitedSources,
  parseRelevantNumbers,
  parseRouteReply,
  previousQueriesFor,
  slugify,
  stripSourceLabel,
} from "./logic.js";
import type { SourceDocument, SourceKind } from "./sources/types.js";
import { buildWeatherDocument, describeWeatherCode, pickPlace, type GeocodedPlace } from "./tools/weather.js";
import { queryPlanSchema, type RetrievedBatch } from "./types.js";

const doc = (kind: SourceKind, name: string, text = "some text"): SourceDocument => ({
  kind,
  title: `${kind} ${name}`,
  url: `https://example.com/${kind}/${name}`,
  text,
});

const batch = (subQuestionIndex: number, documents: SourceDocument[], attempt = 1): RetrievedBatch => ({
  subQuestionIndex,
  attempt,
  queries: { wikipedia: "w", arxiv: "a", hackernews: "h" },
  documents,
});

test("buildLibrary numbers sources by content, not by which parallel branch finished first", () => {
  const batches = [
    batch(1, [doc("hackernews", "h1"), doc("wikipedia", "w1")]),
    batch(0, [doc("arxiv", "a0"), doc("wikipedia", "w0")]),
  ];
  const forward = buildLibrary(batches).map((s) => s.title);
  const reversed = buildLibrary([...batches].reverse()).map((s) => s.title);

  assert.deepEqual(forward, ["wikipedia w0", "arxiv a0", "wikipedia w1", "hackernews h1"]);
  assert.deepEqual(reversed, forward);
  assert.deepEqual(buildLibrary(batches).map((s) => s.id), [1, 2, 3, 4]);
});

test("buildLibrary drops duplicate URLs and documents with no text", () => {
  const library = buildLibrary([
    batch(0, [doc("wikipedia", "same"), doc("arxiv", "blank", "   ")]),
    batch(1, [doc("wikipedia", "same"), doc("arxiv", "real")]),
  ]);
  assert.deepEqual(library.map((s) => s.url), ["https://example.com/wikipedia/same", "https://example.com/arxiv/real"]);
});

test("a sub-question counts as covered once it has one usable source", () => {
  const batches = [
    batch(0, [doc("wikipedia", "a"), doc("arxiv", "b")]), // covered
    batch(1, [doc("wikipedia", "c")]), // one source is enough
    batch(2, [doc("wikipedia", "d", ""), doc("arxiv", "e", "")]), // sources with no text don't count
  ];
  assert.deepEqual(findUncoveredSubQuestions(4, batches), [2, 3]); // 3 has no batch at all
});

test("a retry batch can rescue a sub-question whose first attempt found nothing usable", () => {
  const firstAttempt = batch(0, [doc("wikipedia", "a", "")]);
  const retry = batch(0, [doc("arxiv", "b")], 2);
  assert.deepEqual(findUncoveredSubQuestions(1, [firstAttempt]), [0]);
  assert.deepEqual(findUncoveredSubQuestions(1, [firstAttempt, retry]), []);
  assert.equal(previousQueriesFor([firstAttempt, retry], 0).length, 2);
});

test("buildExtractiveDraft keeps every excerpt, each with a citation that passes the checker", () => {
  const library = buildLibrary([batch(0, [doc("wikipedia", "w", "Wikipedia says X."), doc("arxiv", "a", "A paper says Y.")])]);
  const draft = buildExtractiveDraft(library);

  assert.match(draft, /Wikipedia says X\. \[1\]/);
  assert.match(draft, /A paper says Y\. \[2\]/);
  assert.deepEqual(checkCitations(draft, library.length), []);
});

test("checkCitations accepts real citations in [1], [1][2] and [1, 2] styles", () => {
  assert.deepEqual(checkCitations("A fact [1]. Another [2].", 3), []);
  assert.deepEqual(checkCitations("Both agree [1][2].", 3), []);
  assert.deepEqual(checkCitations("Both agree [1, 2].", 3), []);
  assert.deepEqual(extractCitationIds("x [1, 2] y [3]"), [1, 2, 3]);
});

test("checkCitations rejects a draft with no citations", () => {
  assert.equal(checkCitations("Plenty of confident claims, no sources.", 3).length, 1);
});

test("checkCitations rejects citations that point to sources that do not exist", () => {
  const problems = checkCitations("Real [1] and [2], but invented [7] and [0].", 3);
  assert.equal(problems.length, 1);
  assert.match(problems[0], /\[7\]/);
  assert.match(problems[0], /\[0\]/);
  assert.match(problems[0], /\[1\] to \[3\]/);
});

test("checkCitations rejects a draft leaning on one source when several exist", () => {
  assert.equal(checkCitations("One [1]. Same one [1].", 3).length, 1);
  assert.deepEqual(checkCitations("One [1]. Same one [1].", 1), []); // nothing else to cite
});

test("buildReport generates the Sources list from the real library", () => {
  const library = buildLibrary([batch(0, [doc("wikipedia", "w"), doc("arxiv", "a")])]);
  const report = buildReport({ topic: "My topic", body: "## Overview\n\nText [1][2].", library, runNotes: [], generatedOn: "2026-09-28" });

  assert.match(report, /^# My topic/);
  assert.match(report, /1\. \*\*Wikipedia\*\*: \[wikipedia w\]\(https:\/\/example\.com\/wikipedia\/w\)/);
  assert.match(report, /2\. \*\*arXiv\*\*: \[arxiv a\]\(https:\/\/example\.com\/arxiv\/a\)/);
  assert.doesNotMatch(report, /About this run/);
});

test("keepCitedSources lists only cited sources and renumbers them to match the text", () => {
  const library = buildLibrary([
    batch(0, [doc("wikipedia", "w"), doc("arxiv", "a"), doc("hackernews", "h"), doc("arxiv", "b")]),
  ]); // ids 1=wikipedia w, 2=arxiv a, 3=arxiv b, 4=hackernews h
  const { body, sources } = keepCitedSources("First [4]. Second [2][4]. Third [2, 3].", library);

  // Cited originals 2, 3, 4 become 1, 2, 3 (ascending), so [4]->[3], [2]->[1], [3]->[2].
  assert.equal(body, "First [3]. Second [1][3]. Third [1][2].");
  assert.deepEqual(sources.map((s) => [s.id, s.title]), [[1, "arxiv a"], [2, "arxiv b"], [3, "hackernews h"]]);
});

test("keepCitedSources removes a citation that points at nothing instead of leaving it dangling", () => {
  const library = buildLibrary([batch(0, [doc("wikipedia", "w"), doc("arxiv", "a")])]);
  const { body, sources } = keepCitedSources("Real [1] and made up [42].", library);
  assert.equal(body, "Real [1] and made up .");
  assert.equal(sources.length, 1);
});

test("buildReport hides retrieved-but-unused sources and says how many it left out", () => {
  const library = buildLibrary([batch(0, [doc("wikipedia", "w"), doc("arxiv", "a"), doc("hackernews", "h")])]);
  const report = buildReport({ topic: "t", body: "Only one [3].", library, runNotes: [], generatedOn: "2026-09-28" });

  assert.match(report, /Only one \[1\]\./); // renumbered
  assert.match(report, /1\. \*\*Hacker News\*\*/);
  assert.doesNotMatch(report, /Wikipedia\*\*/);
  assert.match(report, /2 retrieved sources were not used in the brief and are not listed/);
});

test("parseRelevantNumbers accepts NONE or a comma list, and rejects rambling", () => {
  assert.deepEqual(parseRelevantNumbers("1, 4", 5), [1, 4]);
  assert.deepEqual(parseRelevantNumbers(" 2 ", 5), [2]);
  assert.deepEqual(parseRelevantNumbers("NONE", 5), []);
  assert.deepEqual(parseRelevantNumbers("None.", 5), []);
  assert.deepEqual(parseRelevantNumbers("7, 2, 2, 0", 3), [2]); // invented / duplicate numbers ignored
  assert.throws(() => parseRelevantNumbers("Result 2 is relevant because it mentions agents", 5));
  assert.throws(() => parseRelevantNumbers("", 5));
});

test("buildReport lists run notes and handles an empty library", () => {
  const report = buildReport({ topic: "t", body: "body", library: [], runNotes: ["arXiv was down"], generatedOn: "2026-09-28" });
  assert.match(report, /No sources are cited/);
  assert.match(report, /## About this run\n\n- arXiv was down/);
});

test("search queries must contain real words - the model once returned ': ' for every source", () => {
  assert.ok(queryPlanSchema.safeParse({ wikipedia: "LangChain", arxiv: "LLM agent orchestration", hackernews: "LangGraph vs LangChain" }).success);
  assert.equal(queryPlanSchema.safeParse({ wikipedia: ": ", arxiv: ": ", hackernews: ": " }).success, false);
  assert.equal(queryPlanSchema.safeParse({ wikipedia: "LangGraph", arxiv: ", ", hackernews: ", " }).success, false);
  assert.equal(queryPlanSchema.safeParse({ wikipedia: "ab", arxiv: "ok fine", hackernews: "ok fine" }).success, false); // too short
});

test("stripSourceLabel removes a source name the model echoed into its own query", () => {
  assert.equal(stripSourceLabel("Wikipedia: LangGraph framework adoption"), "LangGraph framework adoption");
  assert.equal(stripSourceLabel("arXiv: LangChain agent comparison"), "LangChain agent comparison");
  assert.equal(stripSourceLabel("Hacker News: LangGraph vs LangChain debate"), "LangGraph vs LangChain debate");
  assert.equal(stripSourceLabel("LangGraph vs LangChain"), "LangGraph vs LangChain"); // untouched
  assert.equal(stripSourceLabel("Wikipedia history of Rome"), "Wikipedia history of Rome"); // no colon: a real word
  assert.equal(stripSourceLabel(undefined), "");
});

test("slugify makes safe, bounded file names", () => {
  assert.equal(slugify("How do LangGraph agents differ from LangChain agents?"), "how-do-langgraph-agents-differ-from-langchain-agents");
  assert.equal(slugify("???"), "research-brief");
  assert.ok(slugify("x".repeat(200)).length <= 60);
});

/* ---------- router reply parsing ---------- */

test("parseRouteReply: reads each of the four route shapes", () => {
  assert.deepEqual(parseRouteReply("RESEARCH"), { kind: "research" });
  assert.deepEqual(parseRouteReply("  research.\n"), { kind: "research" });
  assert.deepEqual(parseRouteReply("UNSUPPORTED_LIVE"), { kind: "unsupported" });
  assert.deepEqual(parseRouteReply("WEATHER: Ahmedabad"), { kind: "weather", city: "Ahmedabad" });
  assert.deepEqual(parseRouteReply("weather : New York, US"), { kind: "weather", city: "New York, US" });
  assert.deepEqual(parseRouteReply("MIXED: Surat | why the monsoon happens in Gujarat"), {
    kind: "mixed",
    city: "Surat",
    researchTopic: "why the monsoon happens in Gujarat",
  });
});

test("parseRouteReply: rambling, empty or unusable replies throw so the model is asked again", () => {
  assert.throws(() => parseRouteReply("I think this is a weather question"), /Unexpected router reply/);
  assert.throws(() => parseRouteReply("WEATHER:"), /Unexpected router reply/);
  assert.throws(() => parseRouteReply("WEATHER: 12"), /unusable city/);
  assert.throws(() => parseRouteReply("MIXED: Surat"), /Unexpected router reply|no research question/);
  assert.throws(() => parseRouteReply("MIXED: Surat | "), /no research question/);
});

/* ---------- live-tool readings in the library ---------- */

test("buildLibrary: a weather reading is numbered first and belongs to no sub-question", () => {
  const library = buildLibrary([batch(0, [doc("wikipedia", "a")])], [doc("weather", "now")]);
  assert.deepEqual(library.map((s) => [s.id, s.kind, s.subQuestionIndex]), [
    [1, "weather", -1],
    [2, "wikipedia", 0],
  ]);
});

/* ---------- weather tool: pure parts ---------- */

const place = (over: Partial<GeocodedPlace>): GeocodedPlace => ({
  name: "Ahmedabad",
  latitude: 23.02,
  longitude: 72.58,
  country: "India",
  country_code: "IN",
  admin1: "Gujarat",
  ...over,
});

test("pickPlace: takes the top result, or the one matching a country/region qualifier", () => {
  const india = place({});
  const pakistan = place({ country: "Pakistan", country_code: "PK", admin1: "Khyber Pakhtunkhwa" });
  assert.equal(pickPlace([india, pakistan], ""), india);
  assert.equal(pickPlace([india, pakistan], "Pakistan"), pakistan);
  assert.equal(pickPlace([india, pakistan], "pk"), pakistan);
  assert.equal(pickPlace([india, pakistan], "Atlantis"), india); // unknown qualifier: top result
  assert.equal(pickPlace([], "India"), undefined);
});

test("describeWeatherCode: maps WMO codes and admits unknown ones", () => {
  assert.equal(describeWeatherCode(1), "mainly clear");
  assert.equal(describeWeatherCode(95), "thunderstorm");
  assert.match(describeWeatherCode(1234), /unrecognised condition \(WMO code 1234\)/);
});

test("buildWeatherDocument: every number in the text comes from the API response", () => {
  const document = buildWeatherDocument(
    place({}),
    {
      timezone: "Asia/Kolkata",
      current: {
        time: "2026-09-29T13:00",
        temperature_2m: 34,
        apparent_temperature: 36.1,
        relative_humidity_2m: 42,
        precipitation: 0,
        wind_speed_10m: 3.2,
        weather_code: 1,
      },
      current_units: { temperature_2m: "°C", relative_humidity_2m: "%", precipitation: "mm", wind_speed_10m: "km/h" },
    },
    "https://api.example.com/forecast"
  );
  assert.equal(document.kind, "weather");
  assert.equal(document.title, "Current weather in Ahmedabad, Gujarat, India");
  assert.equal(document.url, "https://api.example.com/forecast");
  assert.match(document.text, /Observed: 2026-09-29 13:00 local time \(Asia\/Kolkata\)/);
  assert.match(document.text, /Conditions: mainly clear/);
  assert.match(document.text, /Temperature: 34 °C \(feels like 36.1 °C\)/);
  assert.match(document.text, /Relative humidity: 42%/);
  assert.match(document.text, /Wind speed: 3.2 km\/h/);
});
