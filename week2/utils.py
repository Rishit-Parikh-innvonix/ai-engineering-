from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def print_benchmark_table(sync_summary, async_summary):
    table = Table(title="Exercise 3: Sync vs Async Throughput", show_lines=True)
    table.add_column("Mode")
    table.add_column("Total time (s)")
    table.add_column("Avg latency (s)")
    table.add_column("Throughput (req/s)")
    table.add_column("Successes")
    table.add_column("Errors")
    table.add_column("Peak memory (MB)")

    for summary in (sync_summary, async_summary):
        table.add_row(
            summary["mode"],
            str(summary["total_seconds"]),
            str(summary["avg_latency_seconds"]),
            str(summary["throughput_per_second"]),
            str(summary["successes"]),
            str(summary["errors"]),
            str(summary["peak_memory_mb"]),
        )

    console.print(table)


def print_panel(title, text):
    console.print(Panel(text, title=title, border_style="cyan"))
