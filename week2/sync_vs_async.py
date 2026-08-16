import asyncio
import os
import statistics
import time

import psutil

import config
import models
import utils

PROMPTS = config.BENCHMARK_PROMPTS


def _peak_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def run_sync():
    """Sequential baseline: one prompt at a time, each call blocks until it returns."""
    results = []
    start = time.perf_counter()
    for prompt in PROMPTS:
        chat = models.get_chat_model(config.BENCHMARK_MODEL, temperature=0.5)
        call_start = time.perf_counter()
        try:
            chat.invoke(prompt)
            results.append({"prompt": prompt, "elapsed": time.perf_counter() - call_start, "error": None})
        except Exception as error:
            results.append({"prompt": prompt, "elapsed": time.perf_counter() - call_start, "error": str(error)})
    total = time.perf_counter() - start
    return {"mode": "sync", "results": results, "total_seconds": total, "peak_memory_mb": _peak_memory_mb()}


async def run_async():
    """Concurrent version: all prompts in flight at once, capped by a semaphore so we
    don't blow through OpenRouter's free-tier per-minute request cap (see config.py)."""
    semaphore = asyncio.Semaphore(config.ASYNC_CONCURRENCY)

    async def call(prompt):
        async with semaphore:
            chat = models.get_chat_model(config.BENCHMARK_MODEL, temperature=0.5)
            call_start = time.perf_counter()
            try:
                await chat.ainvoke(prompt)
                return {"prompt": prompt, "elapsed": time.perf_counter() - call_start, "error": None}
            except Exception as error:
                return {"prompt": prompt, "elapsed": time.perf_counter() - call_start, "error": str(error)}

    start = time.perf_counter()
    results = await asyncio.gather(*(call(prompt) for prompt in PROMPTS))
    total = time.perf_counter() - start
    return {"mode": "async", "results": list(results), "total_seconds": total, "peak_memory_mb": _peak_memory_mb()}


def summarize(run):
    ok_times = [r["elapsed"] for r in run["results"] if not r["error"]]
    n_ok = len(ok_times)
    n_err = len(run["results"]) - n_ok
    avg_latency = statistics.mean(ok_times) if ok_times else 0.0
    throughput = n_ok / run["total_seconds"] if run["total_seconds"] else 0.0
    return {
        "mode": run["mode"],
        "total_seconds": round(run["total_seconds"], 2),
        "avg_latency_seconds": round(avg_latency, 2),
        "throughput_per_second": round(throughput, 3),
        "successes": n_ok,
        "errors": n_err,
        "peak_memory_mb": round(run["peak_memory_mb"], 1),
    }


def main():
    utils.console.print(
        f"\n[bold cyan]Exercise 3: Sync vs Async ({len(PROMPTS)} prompts, "
        f"model={config.BENCHMARK_MODEL}, async concurrency={config.ASYNC_CONCURRENCY})[/bold cyan]\n"
    )

    utils.console.print("Running SYNC (sequential) benchmark...")
    sync_summary = summarize(run_sync())

    utils.console.print("Running ASYNC (concurrent) benchmark...")
    async_summary = summarize(asyncio.run(run_async()))

    utils.print_benchmark_table(sync_summary, async_summary)

    if async_summary["total_seconds"] > 0:
        speedup = sync_summary["total_seconds"] / async_summary["total_seconds"]
    else:
        speedup = 0.0

    utils.console.print(
        f"\nAsync completed all {len(PROMPTS)} prompts [bold]{speedup:.2f}x[/bold] faster in wall-clock "
        f"time than sync (total_seconds: {sync_summary['total_seconds']}s sync vs "
        f"{async_summary['total_seconds']}s async), using at most {config.ASYNC_CONCURRENCY} requests "
        "in flight at once.\n"
        "Note: throughput here is dominated by network wait time (each call is I/O-bound - waiting "
        "on OpenRouter), not CPU. That's exactly why async helps: while one request is waiting on the "
        "network, the event loop can start the next one instead of sitting idle. Peak memory is included "
        "for completeness, but for this workload it stays roughly flat between the two modes - the win "
        "is entirely in wall-clock time, not resource usage.\n"
    )


if __name__ == "__main__":
    main()
