/**
 * Runs all 3 exercises back to back - the assignment note explicitly allows combining them.
 * Each exercise's actual logic lives in its own file (exercise1PrebuiltTool.ts,
 * exercise2CustomTool.ts, exercise3ReactAgent.ts) and can also be run individually
 * (npm run exercise1/2/3); this just sequences all three.
 *
 * Run: npm run agents-demo
 */

import { runExercise1 } from "./exercise1PrebuiltTool.js";
import { runExercise2 } from "./exercise2CustomTool.js";
import { runExercise3 } from "./exercise3ReactAgent.js";
import { printPanel } from "./utils.js";

// A single hiccup (a real network blip, an API outage, a malformed response) in one
// exercise must not take the other two down with it - each exercise is independently
// useful, so one failing shouldn't hide whether the others work.
async function runIsolated(label: string, exercise: () => Promise<void>): Promise<void> {
  try {
    await exercise();
  } catch (error) {
    console.log(`\n(${label} failed: ${error instanceof Error ? error.message : error} - continuing with the next exercise.)`);
  }
}

async function main(): Promise<void> {
  printPanel(
    "Week 5: Agents",
    "1) pre-built tool bound directly  2) custom tool bound directly  3) ReAct agent with all 3 tools"
  );

  await runIsolated("Exercise 1", runExercise1);
  await runIsolated("Exercise 2", runExercise2);
  await runIsolated("Exercise 3", runExercise3);

  console.log(
    "\nObservations to check by hand:\n" +
      "- Exercises 1-2 show a tool bound but NOT wrapped in an agent - WE write the loop\n" +
      "  (boundToolRunner.ts) that checks tool_calls, runs the tool, and sends the result back.\n" +
      "- Exercise 3's agent runs that same kind of loop internally (that's the ReAct\n" +
      "  pattern: reason about what to do, act by calling a tool, observe the result,\n" +
      "  repeat until it can answer) - createAgent builds the loop, we just supply\n" +
      "  the model and the tool list.\n" +
      "- The last question (47 * 89) only needs the calculator - watch that the agent\n" +
      "  does NOT call wikipedia or get_current_weather for it; the tool set doesn't\n" +
      "  change, but which tool(s) get used depends on the question.\n"
  );
}

main();
