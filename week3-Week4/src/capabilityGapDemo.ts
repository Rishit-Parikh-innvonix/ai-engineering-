/**
 * Bonus follow-up to Exercise 3: localVsCloudDemo.ts used easy prompts (fact lookup,
 * a short explanation, a haiku) - all well within a 1B model's ability, so it measured
 * latency, not capability. This demo picks tasks specifically chosen because small
 * models reliably fail them while larger models tend not to: multi-step arithmetic
 * (error compounds across steps), a classic language-trick riddle (pattern-matches to
 * the wrong answer), a syllogism trap (tests whether "some" gets sloppily treated as
 * "all"), and strict multi-constraint instruction-following (tests whether ALL
 * constraints are tracked at once, not just the main one). Each task has an
 * objectively checkable answer, so this reports PASS/FAIL, not just "read and judge".
 * Run: npm run capability-gap
 */

import { HumanMessage } from "@langchain/core/messages";

import * as config from "./config.js";
import * as models from "./models.js";
import { printComparisonTable } from "./utils.js";

interface Task {
  label: string;
  prompt: string;
  check: (response: string) => boolean;
  why: string;
}

const TASKS: Task[] = [
  {
    label: "Multi-step arithmetic",
    prompt:
      "A store had 120 apples. They sold 35% of them in the morning, then sold 40 more in " +
      "the afternoon. They then restocked by adding twice as many apples as they had left " +
      "after those sales. How many apples do they have now, at the end of the day? Work " +
      "through it step by step, then give the final number on its own line as 'Answer: <n>'.",
    check: (r) => /\banswer:?\s*114\b/i.test(r) || /\b114\b/.test(r.split("\n").slice(-3).join(" ")),
    why: "120 -35%=78, -40=38 left, +2x38=76 restocked -> 38+76=114. Each step's error compounds into the next - small models often slip on one step (usually the percentage) and the final number is wrong even if the method looks reasonable.",
  },
  {
    label: "Classic language-trick riddle",
    prompt: "A farmer has 17 sheep, and all but 9 die. How many sheep does the farmer have left? Answer with just the number.",
    check: (r) => /\b9\b/.test(r) && !/\b8\b/.test(r.trim().split("\n")[0] ?? ""),
    why: '"all but 9 die" means 9 SURVIVE - the answer is 9, not 17-9=8. This is a known trap: models pattern-match "17 and 9" to a subtraction problem instead of parsing what the sentence actually says.',
  },
  {
    label: "Syllogism trap",
    prompt:
      "Some Bloops are Razzles. All Razzles are Lazzles. Are all Bloops necessarily Lazzles? " +
      "Answer with just Yes or No, then one sentence of reasoning.",
    check: (r) => /^\s*no\b/i.test(r.trim()) || /\bno\b.{0,40}necessarily/i.test(r),
    why: 'Only SOME Bloops are Razzles, so we can only conclude some Bloops are Lazzles, not all. The correct answer is No. Models often over-generalize "some -> all" through pattern completion.',
  },
  {
    label: "Multi-constraint instruction following",
    prompt:
      "Write a product description for a computer mouse in exactly 40 words. Do not use the " +
      "word 'wireless' anywhere in your answer. You must mention battery life somewhere. " +
      "End your description with a question.",
    check: (r) => {
      const text = r.trim();
      const words = text.split(/\s+/).filter(Boolean);
      const wordCountOk = Math.abs(words.length - 40) <= 6;
      const noForbiddenWord = !/\bwireless\b/i.test(text);
      const mentionsBattery = /\bbattery\b/i.test(text);
      const endsWithQuestion = text.endsWith("?");
      return wordCountOk && noForbiddenWord && mentionsBattery && endsWithQuestion;
    },
    why: "Four simultaneous constraints (word count, a forbidden word, a required topic, a required ending) - passing requires tracking ALL of them at once, not just satisfying the main request. Smaller models tend to drop one constraint (usually the word count or the ending) while satisfying the others.",
  },
];

async function runTask(chat: { invoke: (m: HumanMessage[]) => Promise<{ content: unknown }> }, task: Task) {
  const response = await chat.invoke([new HumanMessage(task.prompt)]);
  const text = String(response.content);
  return { text, passed: task.check(text) };
}

function verdict(passed: boolean): string {
  return passed ? "PASS" : "FAIL";
}

async function main(): Promise<void> {
  console.log(`Local model: ${config.LOCAL_MODEL} (via Ollama)`);
  console.log(`Cloud model: ${config.CLOUD_MODEL} (via OpenRouter)\n`);
  console.log(
    "Each task below has an objectively checkable answer/constraint set - this isn't asking\n" +
      "you to judge which text 'sounds better', it's checking who actually got it right.\n"
  );

  const localChat = models.getLocalChatModel(0.2);
  const cloudChat = models.getCloudChatModel(0.2);

  const rows: (string | number)[][] = [];

  for (const task of TASKS) {
    console.log("=".repeat(70));
    console.log(task.label);
    console.log(`Why this task separates model capability: ${task.why}`);
    console.log("=".repeat(70));

    const [local, cloud] = await Promise.all([runTask(localChat, task), runTask(cloudChat, task)]);

    console.log(`\n[LOCAL - ${verdict(local.passed)}]`);
    console.log(local.text);
    console.log(`\n[CLOUD - ${verdict(cloud.passed)}]`);
    console.log(cloud.text);
    console.log();

    rows.push([task.label, verdict(local.passed), verdict(cloud.passed)]);
  }

  printComparisonTable("Capability check results", ["Task", "Local", "Cloud"], rows);

  console.log(
    "\nThis is the honest version of the local-vs-cloud comparison: localVsCloudDemo.ts's\n" +
      "prompts (fact lookup, short explanation, haiku) were easy enough that a 1B model\n" +
      "handled them fine, which is why 'local won' there was purely a latency finding, not a\n" +
      "capability one. These four tasks are specifically the kind of thing model SIZE tends to\n" +
      "matter for: compounding multi-step arithmetic, resisting a pattern-matching shortcut,\n" +
      "avoiding a classic logical over-generalization, and tracking several constraints at once\n" +
      "instead of just the main one. Check the PASS/FAIL table above against what you'd expect.\n"
  );
}

main();
