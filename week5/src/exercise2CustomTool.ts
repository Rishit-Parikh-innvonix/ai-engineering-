/**
 * Exercise 2: "Create a custom tool like get current stock price or weather details and
 * bind the tool to llm."
 *
 * get_current_weather (tools.ts) is built from scratch with tool(), hitting a real external
 * API (Open-Meteo - free, no API key needed). Bound to the model the same way as exercise 1 -
 * see boundToolRunner.ts for the shared loop.
 *
 * Run: npm run exercise2
 */

import { getCloudChatModel } from "./models.js";
import { getCurrentWeather } from "./tools.js";
import { runBoundToolDemo } from "./boundToolRunner.js";
import { printSectionHeader, runIfMain } from "./utils.js";

export async function runExercise2(): Promise<void> {
  printSectionHeader("Exercise 2: custom tool (get_current_weather, calls Open-Meteo)");
  const chat = getCloudChatModel();
  await runBoundToolDemo(chat, getCurrentWeather, "What's the weather like in Paris right now?");
}

runIfMain(import.meta.url, runExercise2);
