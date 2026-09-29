/**
 * The flowchart. Everything about HOW the agents are coordinated lives here:
 *
 *   START -> router -> research:    plan (below)
 *                  -> weather:     weatherTool -> summarize (below)      live data: no searching
 *                  -> mixed:       weatherTool -> plan (below)           both, in one library
 *                  -> unsupported: save                                  needs live data we lack
 *            (the weather tool failing falls back to plan; the router failing means "research")
 *
 *   plan -> retrieve (one parallel branch per sub-question; each writes queries,
 *                     searches 3 sources at once, and drops off-topic results) -> coverage
 *                                 ^                                                 |
 *                                 '---- nothing relevant for some question? retry (max 2) ---'
 *   coverage -> summarize -> write -> checkCitations -> save -> END
 *                              ^              |
 *                              '-- bad cite? redraft (max 2)
 *   coverage -> noSources -> save    (nothing found at all: report that honestly, don't guess)
 *   summarize/write failing after retries -> save, listing the raw excerpts instead of prose
 *
 * The path is fixed by this code. The AI agents only fill in the content of each box, which is
 * what makes this a deterministic workflow instead of an agent deciding its own steps.
 */

import { END, Send, START, StateGraph } from "@langchain/langgraph";

import type { Agents } from "./agents/types.js";
import type { WeatherTool } from "./tools/weather.js";
import {
  buildExtractiveDraft,
  buildLibrary,
  buildReport,
  checkCitations,
  findUncoveredSubQuestions,
  LIMITS,
  previousQueriesFor,
} from "./logic.js";
import type { SourceFetcher } from "./sources/types.js";
import { ResearchState } from "./state.js";
import type { QueryPlan } from "./types.js";
import { logStep, since } from "./utils.js";

export interface GraphDependencies {
  agents: Agents;
  fetchers: SourceFetcher[];
  weather: WeatherTool;
  saveReport(topic: string, markdown: string): Promise<string>;
}

// What each parallel retrieval branch is handed (a Send payload, not the whole shared state).
interface RetrievalTask {
  topic: string;
  subQuestion: string;
  index: number;
  attempt: number;
  previousQueries: QueryPlan[];
}

const errorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

export function createResearchGraph({ agents, fetchers, weather, saveReport }: GraphDependencies) {
  // ---- Agent nodes: each one reads the notebook and returns only what it changed ----

  // The first diamond: research, a live-data tool, both, or neither. If the routing call itself
  // fails, plain research is the safe default - it is what the workflow did before it had tools.
  const route = async (state: typeof ResearchState.State) => {
    const startedAt = Date.now();
    try {
      const decision = await agents.route(state.topic);
      logStep(
        "Router",
        `${decision.kind}${"city" in decision ? ` (${decision.city})` : ""} (${since(startedAt)})`
      );
      return {
        route: decision.kind,
        weatherCity: "city" in decision ? decision.city : "",
        researchTopic: decision.kind === "mixed" ? decision.researchTopic : "",
      };
    } catch (error) {
      logStep("Router", `failed (${errorMessage(error)}) - treating the topic as plain research`);
      return {
        route: "research" as const,
        runNotes: [`The routing agent failed (${errorMessage(error)}), so the topic was researched without live-data tools.`],
      };
    }
  };

  // Plain code calling the weather API. On failure the run degrades to research, and says so.
  const weatherTool = async (state: typeof ResearchState.State) => {
    const startedAt = Date.now();
    try {
      const document = await weather.lookup(state.weatherCity);
      logStep("Weather tool", `${document.title} (${since(startedAt)})`);
      return {
        toolDocs: [document],
        // Weather-only runs skip retrieval, so the library and the one "sub-question" are set here.
        // (A mixed run rebuilds the library in the coverage check, tool reading first.)
        library: buildLibrary([], [document]),
        subQuestions: state.route === "weather" ? [state.topic] : [],
      };
    } catch (error) {
      logStep("Weather tool", `failed (${errorMessage(error)}) - falling back to research`);
      return {
        route: "research" as const,
        runNotes: [
          `The weather tool could not get the current weather for "${state.weatherCity}" (${errorMessage(error)}), ` +
            "so this brief comes from the research sources only and contains no live weather data.",
        ],
      };
    }
  };

  const unsupported = () => {
    logStep("Router", "needs live data that no tool provides - reporting that instead of guessing");
    return {
      draft:
        "## Overview\n\nThis question needs live, up-to-the-minute data (for example prices, scores, traffic or " +
        "breaking news) that this assistant has no tool for, and the research sources (Wikipedia, arXiv, Hacker News) " +
        "cannot provide it. No brief was written rather than guessing.",
      runNotes: ["The routing agent classified this topic as needing live data other than weather, which is not available."],
    };
  };

  const plan = async (state: typeof ResearchState.State) => {
    const startedAt = Date.now();
    const subQuestions = await agents.plan(state.researchTopic || state.topic);
    logStep("Planner", `split the topic into ${subQuestions.length} sub-questions (${since(startedAt)}):`);
    subQuestions.forEach((question, i) => console.log(`             ${i + 1}. ${question}`));
    return { subQuestions };
  };

  // Runs once per sub-question, all at the same time.
  const retrieve = async (task: RetrievalTask) => {
    const startedAt = Date.now();
    const notes: string[] = [];

    let queries: QueryPlan;
    try {
      queries = await agents.craftQueries(task);
    } catch (error) {
      // One flaky model call shouldn't sink the whole run: fall back to the plain question text.
      queries = { wikipedia: task.subQuestion, arxiv: task.subQuestion, hackernews: task.subQuestion };
      notes.push(
        `The retrieval agent could not write search queries for "${task.subQuestion}" (${errorMessage(error)}), ` +
          "so the question text itself was used as the search query."
      );
    }

    // Every source is searched at once; allSettled means one source being down never hides the others.
    const results = await Promise.allSettled(fetchers.map((fetcher) => fetcher.fetch(queries[fetcher.kind])));
    const fetched = results.flatMap((result, i) => {
      if (result.status === "fulfilled") return result.value;
      notes.push(`${fetchers[i].label} was unavailable for "${task.subQuestion}": ${errorMessage(result.reason)}`);
      return [];
    });

    // Search engines return *something* for any query, so found-something is not the same as
    // found-the-answer. The agent judges relevance and drops the rest; if that judging call
    // itself fails, keep everything (and say so) rather than lose good results.
    let documents = fetched;
    try {
      documents = await agents.gradeRelevance({ topic: task.topic, subQuestion: task.subQuestion, documents: fetched });
    } catch (error) {
      notes.push(`Relevance checking failed for "${task.subQuestion}" (${errorMessage(error)}); all results were kept unchecked.`);
    }

    logStep(
      "Retrieval",
      `sub-question ${task.index + 1} (attempt ${task.attempt}): kept ${documents.length} relevant of ${fetched.length} found (${since(startedAt)})`
    );
    console.log(`             searched: ${fetchers.map((f) => `${f.label} "${queries[f.kind]}"`).join(" | ")}`);
    return {
      retrieved: [{ subQuestionIndex: task.index, attempt: task.attempt, queries, documents }],
      runNotes: notes,
    };
  };

  // Deterministic check - no AI. Runs after ALL parallel retrieval branches have finished.
  const coverage = (state: typeof ResearchState.State) => {
    const library = buildLibrary(state.retrieved, state.toolDocs);
    const uncovered = findUncoveredSubQuestions(state.subQuestions.length, state.retrieved);
    const round = state.retrievalRounds + 1;

    logStep(
      "Coverage check",
      uncovered.length === 0
        ? `${library.length} unique sources, every sub-question covered`
        : `${library.length} unique sources; nothing relevant yet for sub-question(s) ` +
            `${uncovered.map((i) => i + 1).join(", ")} (round ${round} of ${LIMITS.maxRetrievalRounds})`
    );

    const retrying = uncovered.length > 0 && round < LIMITS.maxRetrievalRounds;
    return {
      library,
      retrievalRounds: round,
      runNotes: retrying
        ? [
            `Round ${round}: nothing relevant yet for ${uncovered.map((i) => `"${state.subQuestions[i]}"`).join("; ")}, ` +
              "so the queries were rewritten and searched again.",
          ]
        : uncovered.length > 0
          ? [
              `Retrieval gave up after ${LIMITS.maxRetrievalRounds} rounds; found nothing relevant for: ` +
                `${uncovered.map((i) => `"${state.subQuestions[i]}"`).join("; ")}.`,
            ]
          : [],
    };
  };

  const noSources = () => {
    logStep("Coverage check", "no sources at all were retrieved - reporting that instead of guessing");
    return {
      draft: "## Overview\n\nNo sources could be retrieved for this topic, so no brief could be written. " +
        "See the notes at the end of this report for what went wrong.",
    };
  };

  // The retrieval work is the expensive part, so if the summarizing or writing agent fails after
  // all its retries (free models time out) the run degrades instead of dying: the report still
  // lists the relevant excerpts that were found, unedited.
  const summarize = async (state: typeof ResearchState.State) => {
    const startedAt = Date.now();
    // In a mixed run the planner only saw the research half of the topic, so the weather reading
    // would have no question to belong to and the summarizer would leave it out (seen live).
    const subQuestions =
      state.route === "mixed" && state.toolDocs.length > 0
        ? [`What is the current weather in ${state.weatherCity}?`, ...state.subQuestions]
        : state.subQuestions;
    try {
      const notes = await agents.summarize({
        topic: state.topic,
        subQuestions,
        library: state.library,
      });
      logStep("Summarizer", `condensed ${state.library.length} sources into notes (${since(startedAt)})`);
      return { notes };
    } catch (error) {
      logStep("Summarizer", `failed (${errorMessage(error)}) - falling back to the raw excerpts`);
      return {
        draft: buildExtractiveDraft(state.library),
        runNotes: [
          `The summarization agent failed (${errorMessage(error)}), so this report lists the retrieved excerpts as-is instead of a written brief.`,
        ],
      };
    }
  };

  const write = async (state: typeof ResearchState.State) => {
    const startedAt = Date.now();
    const draftCount = state.draftCount + 1;
    try {
      const draft = await agents.write({
        topic: state.topic,
        notes: state.notes,
        library: state.library,
        problems: state.citationProblems,
        previousDraft: state.draft,
      });
      logStep(
        "Answer writer",
        `${draftCount === 1 ? "wrote the first draft" : `wrote draft ${draftCount} (fixing citation problems)`} (${since(startedAt)})`
      );
      return { draft, draftCount };
    } catch (error) {
      logStep("Answer writer", `failed (${errorMessage(error)}) - falling back to the raw excerpts`);
      return {
        draft: buildExtractiveDraft(state.library),
        draftCount,
        runNotes: [
          `The answer-writing agent failed (${errorMessage(error)}), so this report lists the retrieved excerpts as-is instead of a written brief.`,
        ],
      };
    }
  };

  // Deterministic check - no AI. The writer's prose is trusted only after this passes.
  const checkCitationsNode = (state: typeof ResearchState.State) => {
    const citationProblems = checkCitations(state.draft, state.library.length);
    logStep("Citation check", citationProblems.length === 0 ? "every citation points to a real source" : citationProblems.join(" "));
    return { citationProblems };
  };

  const save = async (state: typeof ResearchState.State) => {
    const runNotes = [...state.runNotes];
    if (state.citationProblems.length > 0) {
      runNotes.push(
        `Warning: the writer could not fix all citation problems after ${state.draftCount} drafts, ` +
          `so treat the [n] references with care: ${state.citationProblems.join(" ")}`
      );
    }
    const report = buildReport({
      topic: state.topic,
      body: state.draft,
      library: state.library,
      runNotes,
      generatedOn: `${new Date().toISOString().slice(0, 16).replace("T", " ")} UTC`,
    });
    const reportPath = await saveReport(state.topic, report);
    logStep("Saved", reportPath);
    return { report, reportPath };
  };

  // ---- Routers: these are the diamonds in the flowchart ----

  const afterRoute = (state: typeof ResearchState.State) =>
    state.route === "weather" || state.route === "mixed" ? "weatherTool" : state.route === "unsupported" ? "unsupported" : "plan";

  // Weather-only goes straight to summarizing; a mixed run (or a failed tool) goes on to research.
  const afterWeather = (state: typeof ResearchState.State) => (state.route === "weather" ? "summarize" : "plan");

  // Fan-out: one Send per sub-question makes LangGraph run that many `retrieve` branches in parallel.
  const fanOutRetrieval = (state: typeof ResearchState.State) =>
    state.subQuestions.map(
      (subQuestion, index) =>
        new Send("retrieve", { topic: state.topic, subQuestion, index, attempt: 1, previousQueries: [] } satisfies RetrievalTask)
    );

  const afterCoverage = (state: typeof ResearchState.State) => {
    const uncovered = findUncoveredSubQuestions(state.subQuestions.length, state.retrieved);
    if (uncovered.length > 0 && state.retrievalRounds < LIMITS.maxRetrievalRounds) {
      return uncovered.map(
        (index) =>
          new Send("retrieve", {
            topic: state.topic,
            subQuestion: state.subQuestions[index],
            index,
            attempt: state.retrievalRounds + 1,
            previousQueries: previousQueriesFor(state.retrieved, index),
          } satisfies RetrievalTask)
      );
    }
    return state.library.length > 0 ? "summarize" : "noSources";
  };

  // No notes means the summarizer failed and already set a fallback draft: skip the writer.
  const afterSummarize = (state: typeof ResearchState.State) => (state.notes ? "write" : "save");

  const afterCitationCheck = (state: typeof ResearchState.State) =>
    state.citationProblems.length > 0 && state.draftCount < LIMITS.maxDrafts ? "write" : "save";

  return new StateGraph(ResearchState)
    .addNode("router", route)
    .addNode("weatherTool", weatherTool)
    .addNode("unsupported", unsupported)
    .addNode("plan", plan)
    .addNode("retrieve", retrieve)
    .addNode("coverage", coverage)
    .addNode("noSources", noSources)
    .addNode("summarize", summarize)
    .addNode("write", write)
    .addNode("checkCitations", checkCitationsNode)
    .addNode("save", save)
    .addEdge(START, "router")
    .addConditionalEdges("router", afterRoute, ["weatherTool", "unsupported", "plan"])
    .addConditionalEdges("weatherTool", afterWeather, ["summarize", "plan"])
    .addEdge("unsupported", "save")
    .addConditionalEdges("plan", fanOutRetrieval, ["retrieve"])
    .addEdge("retrieve", "coverage")
    .addConditionalEdges("coverage", afterCoverage, ["retrieve", "summarize", "noSources"])
    .addEdge("noSources", "save")
    .addConditionalEdges("summarize", afterSummarize, ["write", "save"])
    .addEdge("write", "checkCitations")
    .addConditionalEdges("checkCitations", afterCitationCheck, ["write", "save"])
    .addEdge("save", END)
    .compile();
}
