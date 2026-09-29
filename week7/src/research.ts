/**
 * Week 7 - a research assistant built as a LangGraph workflow.
 *
 * Usage:  npm run research -- "How do LangGraph agents differ from LangChain agents?"
 *
 * Five agents (routing, planning, retrieval, summarization, final answer) are coordinated by the graph
 * in graph.ts. The result is saved as a markdown brief with numbered citations under reports/.
 */

import { createAgents } from "./agents/index.js";
import { createResearchGraph } from "./graph.js";
import { getCloudChatModel } from "./models.js";
import { saveReportToDisk } from "./report.js";
import { DEFAULT_FETCHERS } from "./sources/index.js";
import { openMeteoWeather } from "./tools/weather.js";
import { printSectionHeader } from "./utils.js";

// A whole run makes a dozen or so model calls plus web requests, and free-tier latency swings
// widely. This ceiling exists so a stalled model or network can never leave the terminal hanging
// forever; progress lines are printed throughout, so a long run is never silent.
const RUN_TIMEOUT_MS = 8 * 60_000;

async function main(): Promise<void> {
  const topic = process.argv.slice(2).join(" ").trim();
  if (!topic) {
    console.error('Usage: npm run research -- "your research topic"');
    console.error('Example: npm run research -- "How do LangGraph agents differ from LangChain agents?"');
    process.exitCode = 1;
    return;
  }

  printSectionHeader(`Researching: ${topic}`);

  const graph = createResearchGraph({
    agents: createAgents(getCloudChatModel()),
    fetchers: DEFAULT_FETCHERS,
    weather: openMeteoWeather,
    saveReport: saveReportToDisk,
  });

  try {
    const result = await graph.invoke({ topic }, { signal: AbortSignal.timeout(RUN_TIMEOUT_MS) });
    printSectionHeader("Finished", "-");
    console.log(result.report);
  } catch (error) {
    console.error(`\nResearch failed: ${error instanceof Error ? error.message : error}`);
    console.error(
      "\nMost likely causes: the free OpenRouter model is slow, unavailable or was removed from the " +
        "free tier (see CLOUD_MODEL in src/config.ts), or the network is down."
    );
    process.exitCode = 1;
  }
}

main();
