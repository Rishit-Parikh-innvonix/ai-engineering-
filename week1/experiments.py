import config
import models
import utils


def run_temperature_experiment():
    results = []
    for temp in config.TEMPERATURE_VALUES:
        result = models.ask_model(config.EXPERIMENT_MODEL, config.CREATIVITY_PROMPT, temperature=temp)
        result["param_value"] = temp
        result["unique_word_ratio"] = utils.unique_word_ratio(result["response"])
        results.append(result)
        utils.wait_between_calls(config.REQUEST_DELAY_SECONDS)
    utils.print_experiment_table(results, "Temperature")
    return results


def run_top_p_experiment():
    results = []
    for top_p in config.TOP_P_VALUES:
        result = models.ask_model(config.EXPERIMENT_MODEL, config.CREATIVITY_PROMPT, temperature=0.9, top_p=top_p)
        result["param_value"] = top_p
        result["unique_word_ratio"] = utils.unique_word_ratio(result["response"])
        results.append(result)
        utils.wait_between_calls(config.REQUEST_DELAY_SECONDS)
    utils.print_experiment_table(results, "Top-p")
    return results


def run_max_tokens_experiment():
    results = []
    for max_tokens in config.MAX_TOKENS_VALUES:
        result = models.ask_model(config.EXPERIMENT_MODEL, config.LENGTH_PROMPT, temperature=0.7, max_tokens=max_tokens)
        result["param_value"] = max_tokens
        result["unique_word_ratio"] = utils.unique_word_ratio(result["response"])
        results.append(result)
        utils.wait_between_calls(config.REQUEST_DELAY_SECONDS)
    utils.print_experiment_table(results, "Max Tokens")
    return results


def run_hallucination_experiment():
    results = []
    for prompt in config.HALLUCINATION_PROMPTS:
        result = models.ask_model(config.EXPERIMENT_MODEL, prompt, temperature=0.7)
        results.append(result)
        utils.wait_between_calls(config.REQUEST_DELAY_SECONDS)
    utils.print_hallucination_table(results)
    return results


def build_experiment_transcript(temp_results, top_p_results, max_tokens_results, hallucination_results):
    lines = []

    lines.append("TEMPERATURE EXPERIMENT (same prompt, varying temperature):")
    for r in temp_results:
        lines.append(
            f"- temperature={r['param_value']}: unique_word_ratio={r['unique_word_ratio']}, "
            f"length={r['response_length']} chars -> \"{r['response'][:150]}\""
        )

    lines.append("\nTOP-P EXPERIMENT (same prompt, varying top_p):")
    for r in top_p_results:
        lines.append(
            f"- top_p={r['param_value']}: unique_word_ratio={r['unique_word_ratio']}, "
            f"length={r['response_length']} chars -> \"{r['response'][:150]}\""
        )

    lines.append("\nMAX TOKENS EXPERIMENT (same prompt, varying max_tokens):")
    for r in max_tokens_results:
        lines.append(f"- max_tokens={r['param_value']}: length={r['response_length']} chars -> \"{r['response'][:150]}\"")

    lines.append("\nHALLUCINATION EXPERIMENT (fictional, future, or live-data prompts):")
    for r in hallucination_results:
        lines.append(f"- prompt: \"{r['prompt']}\" -> response: \"{r['response'][:200]}\"")

    return "\n".join(lines)


def generate_experiment_summary(temp_results, top_p_results, max_tokens_results, hallucination_results):
    transcript = build_experiment_transcript(temp_results, top_p_results, max_tokens_results, hallucination_results)
    question = (
        "You are helping a complete beginner understand LLM parameters. "
        "Based ONLY on the actual experiment data below, write a clear summary covering: "
        "1) how temperature affected creativity, 2) how temperature affected determinism, "
        "3) how top-p affected diversity, 4) how max_tokens affected response length, "
        "5) based of data what you think is there any hallucinations model done?, 6) lessons learned. "
        "Reference the actual numbers and text given. Do not invent data that is not shown below.\n\n"
        f"{transcript}"
    )
    return models.ask_judge(question)
