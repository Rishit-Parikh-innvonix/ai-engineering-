import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def unique_word_ratio(text):
    words = text.lower().split()
    if not words:
        return 0.0
    return round(len(set(words)) / len(words), 2)


def wait_between_calls(seconds):
    time.sleep(seconds)


def print_model_comparison_table(results):
    table = Table(title="Exercise 1: Model Comparison", show_lines=True)
    table.add_column("Model")
    table.add_column("Prompt", overflow="fold", max_width=30)
    table.add_column("Time (s)")
    table.add_column("Length (chars)")
    table.add_column("Tokens")
    table.add_column("Response", overflow="fold", max_width=50)

    for r in results:
        if r["tokens"] and r["tokens"].get("total_tokens") is not None:
            tokens = str(r["tokens"]["total_tokens"])
        else:
            tokens = f"~{r['estimated_tokens']} (est.)"

        if r["error"]:
            response_preview = f"[red]ERROR: {r['error']}[/red]"
        else:
            response_preview = r["response"][:200] + "..." if len(r["response"]) > 200 else r["response"]

        table.add_row(
            r["model"],
            r["prompt"][:60] + ("..." if len(r["prompt"]) > 60 else ""),
            str(r["time_seconds"]),
            str(r["response_length"]),
            tokens,
            response_preview,
        )

    console.print(table)


def print_experiment_table(results, varying_param):
    table = Table(title=f"Exercise 2: Varying {varying_param}", show_lines=True)
    table.add_column(varying_param)
    table.add_column("Time (s)")
    table.add_column("Length (chars)")
    table.add_column("Unique Word Ratio")
    table.add_column("Response Preview", overflow="fold", max_width=60)

    for r in results:
        if r["error"]:
            preview = f"[red]ERROR: {r['error']}[/red]"
        else:
            preview = r["response"][:250] + "..." if len(r["response"]) > 250 else r["response"]

        table.add_row(
            str(r["param_value"]),
            str(r["time_seconds"]),
            str(r["response_length"]),
            str(r.get("unique_word_ratio", "N/A")),
            preview,
        )

    console.print(table)


def print_hallucination_table(results):
    table = Table(title="Exercise 2: Hallucination Experiments", show_lines=True)
    table.add_column("Prompt", max_width=35, overflow="fold")
    table.add_column("Response", max_width=65, overflow="fold")

    for r in results:
        response = r["response"] if not r["error"] else f"[red]ERROR: {r['error']}[/red]"
        table.add_row(r["prompt"], response)

    console.print(table)


def print_summary(title, text):
    console.print(Panel(text, title=title, border_style="cyan"))
