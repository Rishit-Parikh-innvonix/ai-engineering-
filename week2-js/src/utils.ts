import Table from "cli-table3";
import chalk from "chalk";

// JS stand-in for week2/utils.py's rich-based table/panel printers. cli-table3 +
// chalk play the same role here that rich's Table/Panel play in the Python version.

export interface BenchmarkSummary {
  mode: string;
  total_seconds: number;
  avg_latency_seconds: number;
  throughput_per_second: number;
  successes: number;
  errors: number;
  peak_memory_mb: number;
}

export function printBenchmarkTable(syncSummary: BenchmarkSummary, asyncSummary: BenchmarkSummary): void {
  const table = new Table({
    head: ["Mode", "Total time (s)", "Avg latency (s)", "Throughput (req/s)", "Successes", "Errors", "Peak memory (MB)"],
  });

  for (const summary of [syncSummary, asyncSummary]) {
    table.push([
      summary.mode,
      String(summary.total_seconds),
      String(summary.avg_latency_seconds),
      String(summary.throughput_per_second),
      String(summary.successes),
      String(summary.errors),
      String(summary.peak_memory_mb),
    ]);
  }

  console.log(chalk.bold.cyan("\nExercise 3: Sync vs Async Throughput"));
  console.log(table.toString());
}

export function printPanel(title: string, text: string): void {
  const width = Math.min(100, Math.max(title.length, ...text.split("\n").map((line) => line.length)) + 4);
  const border = "─".repeat(width);
  console.log(chalk.cyan(`┌${border}┐`));
  console.log(chalk.cyan("│ ") + chalk.bold(title));
  console.log(chalk.cyan(`├${border}┤`));
  for (const line of text.split("\n")) {
    console.log(chalk.cyan("│ ") + line);
  }
  console.log(chalk.cyan(`└${border}┘`));
}
