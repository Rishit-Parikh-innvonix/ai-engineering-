/**
 * Exercise 2: Model Invocation Methods - invoke(), stream(), ainvoke(), astream().
 *
 * Note on the JS mapping: Python LangChain has a real sync/async split - invoke()
 * blocks the thread, ainvoke() is a coroutine you await, and the same split exists
 * for stream()/astream(). LangChain.js has no such split: every call (invoke, stream)
 * already returns a Promise/async-iterator and is non-blocking by default. So here
 * "invoke()"/"stream()" are demonstrated as sequential (one call awaited before the
 * next starts) and "ainvoke()"/"astream()" are demonstrated as the same calls fired
 * concurrently (Promise.all) - the same sync-vs-async *behavior* Python shows via
 * separate methods, expressed in JS via sequential-await vs concurrent-await.
 * Run: npm run invocation-demo
 */

import * as config from "./config.js";
import * as models from "./models.js";
import { printComparisonTable } from "./utils.js";

const PROMPTS = config.INVOCATION_DEMO_PROMPTS;

class Semaphore {
  private available: number;
  private readonly waiters: (() => void)[] = [];
  constructor(count: number) {
    this.available = count;
  }
  async acquire(): Promise<void> {
    if (this.available > 0) {
      this.available -= 1;
      return;
    }
    await new Promise<void>((resolve) => this.waiters.push(resolve));
  }
  release(): void {
    const next = this.waiters.shift();
    if (next) next();
    else this.available += 1;
  }
}

/** invoke(): one prompt at a time, each call awaited before the next starts. */
async function runInvokeSequential(): Promise<number> {
  const chat = models.getChatModel(config.INVOCATION_MODEL, { temperature: 0.3 });
  const start = performance.now();
  for (const prompt of PROMPTS) {
    await chat.invoke(prompt);
  }
  return (performance.now() - start) / 1000;
}

/** ainvoke() stand-in: all invoke() calls in flight at once, capped by a semaphore. */
async function runInvokeConcurrent(): Promise<number> {
  const semaphore = new Semaphore(config.INVOCATION_CONCURRENCY);
  const chat = models.getChatModel(config.INVOCATION_MODEL, { temperature: 0.3 });

  async function call(prompt: string): Promise<void> {
    await semaphore.acquire();
    try {
      await chat.invoke(prompt);
    } finally {
      semaphore.release();
    }
  }

  const start = performance.now();
  await Promise.all(PROMPTS.map((prompt) => call(prompt)));
  return (performance.now() - start) / 1000;
}

/** stream(): a single prompt, consumed chunk-by-chunk, printed as tokens arrive. */
async function runStreamSingle(): Promise<number> {
  const chat = models.getStreamingChatModel(config.INVOCATION_MODEL, 0.3);
  const start = performance.now();
  process.stdout.write("stream() tokens: ");
  const tokenStream = await chat.stream(PROMPTS[0]);
  for await (const chunk of tokenStream) {
    if (chunk.content) process.stdout.write(String(chunk.content));
  }
  process.stdout.write("\n");
  return (performance.now() - start) / 1000;
}

/** astream() stand-in: every prompt streamed concurrently, chunks interleaved as they arrive. */
async function runStreamConcurrent(): Promise<number> {
  const chat = models.getStreamingChatModel(config.INVOCATION_MODEL, 0.3);
  const start = performance.now();

  async function consume(prompt: string, index: number): Promise<void> {
    const tokenStream = await chat.stream(prompt);
    for await (const chunk of tokenStream) {
      if (chunk.content) process.stdout.write(`[${index}]`);
    }
  }

  await Promise.all(PROMPTS.map((prompt, index) => consume(prompt, index)));
  process.stdout.write("\n");
  return (performance.now() - start) / 1000;
}

async function main(): Promise<void> {
  console.log(
    `\nExercise 2: Model Invocation Methods (${PROMPTS.length} prompts, model=${config.INVOCATION_MODEL}, concurrency=${config.INVOCATION_CONCURRENCY})\n`
  );

  console.log("Running invoke() sequentially...");
  const invokeSeconds = await runInvokeSequential();

  console.log("Running invoke() concurrently (ainvoke() stand-in)...");
  const ainvokeSeconds = await runInvokeConcurrent();

  console.log("\nRunning stream() on a single prompt...");
  const streamSeconds = await runStreamSingle();

  console.log("\nRunning stream() on all prompts concurrently (astream() stand-in) - chunk labels show interleaving:");
  const astreamSeconds = await runStreamConcurrent();

  printComparisonTable(
    "Exercise 2: Invocation Method Timing",
    ["Method", "Prompts", "Total time (s)"],
    [
      ["invoke() sequential", PROMPTS.length, invokeSeconds.toFixed(2)],
      ["ainvoke() (concurrent invoke())", PROMPTS.length, ainvokeSeconds.toFixed(2)],
      ["stream() single prompt", 1, streamSeconds.toFixed(2)],
      ["astream() (concurrent stream())", PROMPTS.length, astreamSeconds.toFixed(2)],
    ]
  );

  const speedup = ainvokeSeconds > 0 ? invokeSeconds / ainvokeSeconds : 0;
  console.log(
    `\nConcurrent invoke() ("ainvoke()") finished ${speedup.toFixed(2)}x faster than sequential invoke() ` +
      `across the same ${PROMPTS.length} prompts - the same win async/await gives Python's ainvoke(), because ` +
      "each call is I/O-bound (waiting on OpenRouter) and Node's event loop overlaps that wait time.\n" +
      "stream() vs invoke() latency-to-first-token: streaming starts printing tokens as they arrive instead " +
      "of waiting for the full response, which is why it feels faster interactively even when total time is similar.\n"
  );
}

main();
