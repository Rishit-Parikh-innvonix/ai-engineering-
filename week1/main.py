import config
import models
import utils
import experiments
import switch_demo


def run_exercise_1():
    utils.console.print("\n[bold cyan]Exercise 1: Comparing Free OpenRouter Models[/bold cyan]\n")
    all_results = []

    for model_name in config.COMPARISON_MODELS:
        utils.console.print(f"Running prompts on [bold]{model_name}[/bold]...")
        for prompt in config.COMPARISON_PROMPTS:
            result = models.ask_model(model_name, prompt)
            all_results.append(result)
            utils.wait_between_calls(config.REQUEST_DELAY_SECONDS)

    utils.print_model_comparison_table(all_results)

    transcript_lines = []
    for r in all_results:
        text = r["response"] if not r["error"] else f"ERROR: {r['error']}"
        transcript_lines.append(
            f"Model: {r['model']}\nPrompt: {r['prompt']}\nResponse: {text}\nTime: {r['time_seconds']}s\n"
        )
    transcript = "\n".join(transcript_lines)

    question = (
        "You are comparing several LLMs for a beginner AI engineering student. "
        "Based ONLY on the actual model responses and timing data below, write a comparison covering: "
        "response quality, coding ability, reasoning ability, speed, clarity, overall strengths and "
        "weaknesses per model, which model is best for beginners, which model produced the best "
        "coding answer, and which model produced the best reasoning answer. "
        "Only use evidence from the data below, do not speculate beyond it.\n\n"
        f"{transcript}"
    )
    summary = models.ask_judge(question)
    utils.print_summary("Exercise 1 Summary (generated from the actual outputs above)", summary)


def run_exercise_2():
    utils.console.print("\n[bold cyan]Exercise 2: Experimenting with LLM Parameters[/bold cyan]\n")

    temp_results = experiments.run_temperature_experiment()
    top_p_results = experiments.run_top_p_experiment()
    max_tokens_results = experiments.run_max_tokens_experiment()
    hallucination_results = experiments.run_hallucination_experiment()

    summary = experiments.generate_experiment_summary(
        temp_results, top_p_results, max_tokens_results, hallucination_results
    )
    utils.print_summary("Exercise 2 Summary (generated from the actual outputs above)", summary)


def show_menu():
    utils.console.print("\n[bold green]AI Engineering - Week 1 Assignment[/bold green]")
    utils.console.print("1. Exercise 1 - Compare free OpenRouter models")
    utils.console.print("2. Exercise 2 - Experiment with LLM parameters")
    utils.console.print("3. Exercise 3 - LangChain model-switch demo (read README.md for the full lesson)")
    utils.console.print("4. Exit")


def main():
    if not config.OPENROUTER_API_KEY:
        utils.console.print("[red]OPENROUTER_API_KEY is missing. Copy .env.example to .env and add your key.[/red]")
        return

    while True:
        show_menu()
        choice = input("\nChoose an option (1-4): ").strip()

        if choice == "1":
            run_exercise_1()
        elif choice == "2":
            run_exercise_2()
        elif choice == "3":
            switch_demo.run_demo()
        elif choice == "4":
            break
        else:
            utils.console.print("[yellow]Please enter 1, 2, 3, or 4.[/yellow]")


if __name__ == "__main__":
    main()
