/**
 * Exercise 3: sync (sequential) vs async (concurrent) throughput comparison.
 * Run: npm run benchmark
 */

import * as config from "./config.js";
import * as models from "./models.js";
import { printBenchmarkTable, type BenchmarkSummary } from "./utils.js";

const PROMPTS = config.BENCHMARK_PROMPTS;

interface CallResult {
  prompt: string;
  elapsed: number;
  error: string | null;
}

interface RunResult {
  mode: string;
  results: CallResult[];
  totalSeconds: number;
  peakMemoryMb: number;
}

function peakMemoryMb(): number {
  return process.memoryUsage().rss / (1024 * 1024);
}

/** Sequential baseline: one prompt at a time, each call blocks (via await) before the next starts. */
async function runSync(): Promise<RunResult> {
  const results: CallResult[] = [];
  const start = performance.now();
  for (const prompt of PROMPTS) {
    const chat = models.getChatModel(config.BENCHMARK_MODEL, { temperature: 0.5 });
    const callStart = performance.now();
    try {
      await chat.invoke(prompt);
      results.push({ prompt, elapsed: (performance.now() - callStart) / 1000, error: null });
    } catch (error) {
      results.push({ prompt, elapsed: (performance.now() - callStart) / 1000, error: String(error) });
    }
  }
  const total = (performance.now() - start) / 1000;
  return { mode: "sync", results, totalSeconds: total, peakMemoryMb: peakMemoryMb() };
}

/** Simple counting semaphore - the JS analog of asyncio.Semaphore, used below to cap
 * concurrency so we don't blow through OpenRouter's free-tier per-minute request cap. */
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
    if (next) {
      next();
    } else {
      this.available += 1;
    }
  }
}

/** Concurrent version: all prompts in flight at once, capped by a semaphore. */
async function runAsync(): Promise<RunResult> {
  const semaphore = new Semaphore(config.ASYNC_CONCURRENCY);

  async function call(prompt: string): Promise<CallResult> {
    await semaphore.acquire();
    try {
      const chat = models.getChatModel(config.BENCHMARK_MODEL, { temperature: 0.5 });
      const callStart = performance.now();
      try {
        await chat.invoke(prompt);
        return { prompt, elapsed: (performance.now() - callStart) / 1000, error: null };
      } catch (error) {
        return { prompt, elapsed: (performance.now() - callStart) / 1000, error: String(error) };
      }
    } finally {
      semaphore.release();
    }
  }

  const start = performance.now();
  const results = await Promise.all(PROMPTS.map((prompt) => call(prompt)));
  const total = (performance.now() - start) / 1000;
  return { mode: "async", results, totalSeconds: total, peakMemoryMb: peakMemoryMb() };
}

function summarize(run: RunResult): BenchmarkSummary {
  const okTimes = run.results.filter((r) => !r.error).map((r) => r.elapsed);
  const nOk = okTimes.length;
  const nErr = run.results.length - nOk;
  const avgLatency = okTimes.length ? okTimes.reduce((a, b) => a + b, 0) / okTimes.length : 0;
  const throughput = run.totalSeconds ? nOk / run.totalSeconds : 0;
  return {
    mode: run.mode,
    total_seconds: Math.round(run.totalSeconds * 100) / 100,
    avg_latency_seconds: Math.round(avgLatency * 100) / 100,
    throughput_per_second: Math.round(throughput * 1000) / 1000,
    successes: nOk,
    errors: nErr,
    peak_memory_mb: Math.round(run.peakMemoryMb * 10) / 10,
  };
}

async function main(): Promise<void> {
  console.log(
    `\nExercise 3: Sync vs Async (${PROMPTS.length} prompts, model=${config.BENCHMARK_MODEL}, async concurrency=${config.ASYNC_CONCURRENCY})\n`
  );

  console.log("Running SYNC (sequential) benchmark...");
  const syncSummary = summarize(await runSync());

  console.log("Running ASYNC (concurrent) benchmark...");
  const asyncSummary = summarize(await runAsync());

  printBenchmarkTable(syncSummary, asyncSummary);

  const speedup = asyncSummary.total_seconds > 0 ? syncSummary.total_seconds / asyncSummary.total_seconds : 0;

  console.log(
    `\nAsync completed all ${PROMPTS.length} prompts ${speedup.toFixed(2)}x faster in wall-clock ` +
      `time than sync (total_seconds: ${syncSummary.total_seconds}s sync vs ` +
      `${asyncSummary.total_seconds}s async), using at most ${config.ASYNC_CONCURRENCY} requests ` +
      "in flight at once.\n" +
      "Note: throughput here is dominated by network wait time (each call is I/O-bound - waiting " +
      "on OpenRouter), not CPU. That's exactly why concurrency helps: while one request is waiting on " +
      "the network, Node's event loop can start the next one instead of sitting idle. Peak memory is " +
      "included for completeness, but for this workload it stays roughly flat between the two modes - " +
      "the win is entirely in wall-clock time, not resource usage.\n"
  );
}

main();
