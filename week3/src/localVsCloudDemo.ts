/**
 * Exercise 3 - Ollama Experiment: run a local model and compare its responses
 * (and latency) with a cloud model, on the same prompts.
 *
 * Local: llama3.2:1b via Ollama (must have `ollama serve` running and
 *   `ollama pull llama3.2:1b` done beforehand - both true after the setup in README).
 * Cloud: openai/gpt-oss-20b:free via OpenRouter.
 *
 * Run: npm run local-vs-cloud
 */

import { HumanMessage } from "@langchain/core/messages";

import * as config from "./config.js";
import * as models from "./models.js";
import { printComparisonTable } from "./utils.js";

const PROMPTS = [
  "In one sentence, what is the capital of Australia?",
  "In two sentences, explain what a binary search does and why it's faster than checking every item.",
  "Write a 3-line haiku about the ocean.",
];

interface RunResult {
  prompt: string;
  text: string;
  seconds: number;
}

async function runPrompt(chat: { invoke: (m: HumanMessage[]) => Promise<{ content: unknown }> }, prompt: string): Promise<RunResult> {
  const start = performance.now();
  const response = await chat.invoke([new HumanMessage(prompt)]);
  const seconds = (performance.now() - start) / 1000;
  return { prompt, text: String(response.content), seconds };
}

async function main(): Promise<void> {
  console.log(`Local model:  ${config.LOCAL_MODEL} (via Ollama at ${config.OLLAMA_BASE_URL})`);
  console.log(`Cloud model:  ${config.CLOUD_MODEL} (via OpenRouter)\n`);

  const localChat = models.getLocalChatModel(0.7);
  const cloudChat = models.getCloudChatModel(0.7);

  const rows: (string | number)[][] = [];

  for (const prompt of PROMPTS) {
    console.log("=".repeat(70));
    console.log(`Prompt: ${prompt}`);
    console.log("=".repeat(70));

    const [local, cloud] = await Promise.all([runPrompt(localChat, prompt), runPrompt(cloudChat, prompt)]);

    console.log(`\n[LOCAL - ${config.LOCAL_MODEL}] (${local.seconds.toFixed(2)}s)`);
    console.log(local.text);

    console.log(`\n[CLOUD - ${config.CLOUD_MODEL}] (${cloud.seconds.toFixed(2)}s)`);
    console.log(cloud.text);
    console.log();

    rows.push([prompt.slice(0, 40) + (prompt.length > 40 ? "..." : ""), local.seconds.toFixed(2), cloud.seconds.toFixed(2)]);
  }

  printComparisonTable("Latency: local vs cloud (seconds)", ["Prompt", "Local", "Cloud"], rows);

  console.log(
    "\nObservations to check by hand:\n" +
      "- Latency: local runs entirely on your machine's CPU/GPU (no network round-trip) but is a\n" +
      "  much smaller model (1B params); cloud has network latency but runs a larger model on\n" +
      "  dedicated hardware. Which actually won on your machine, and by how much?\n" +
      "- Quality: does the small local model's phrasing/accuracy noticeably lag the cloud model's,\n" +
      "  especially on the binary search explanation (needs precise reasoning) vs the haiku\n" +
      "  (more creative, less precision-sensitive)?\n" +
      "- Cost/privacy: the local run made zero network calls and cost nothing - that's the actual\n" +
      "  trade being made, independent of which one 'won' on speed or quality this time.\n"
  );
}

main();
