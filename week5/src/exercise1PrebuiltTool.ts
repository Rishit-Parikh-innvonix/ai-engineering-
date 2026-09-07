/**
 * Exercise 1: "Provide any langchain pre-built tool to llm."
 *
 * WikipediaQueryRun (from @langchain/community) bound directly to the chat model - no
 * custom code for the tool itself, just configuration. See boundToolRunner.ts for the
 * request -> tool_call -> tool_result -> final answer loop this runs.
 *
 * Run: npm run exercise1
 */

import { getCloudChatModel } from "./models.js";
import { wikipedia } from "./tools.js";
import { runBoundToolDemo } from "./boundToolRunner.js";
import { printSectionHeader, runIfMain } from "./utils.js";

export async function runExercise1(): Promise<void> {
  printSectionHeader("Exercise 1: pre-built LangChain tool (WikipediaQueryRun)");
  const chat = getCloudChatModel();
  await runBoundToolDemo(chat, wikipedia, "Who is tilak verma?");
}

runIfMain(import.meta.url, runExercise1);
