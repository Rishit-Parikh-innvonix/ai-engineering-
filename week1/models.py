import time
from langchain_openai import ChatOpenAI
import config


def get_chat_model(model_name, temperature=0.7, top_p=1.0, max_tokens=None):
    return ChatOpenAI(
        model=model_name,
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        default_headers={
            "HTTP-Referer": "https://localhost",
            "X-Title": "AI Engineering Week 1 Assignment",
        },
    )


def extract_token_usage(ai_message):
    usage = getattr(ai_message, "usage_metadata", None)
    if usage:
        return {
            "prompt_tokens": usage.get("input_tokens"),
            "completion_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
    metadata_usage = ai_message.response_metadata.get("token_usage")
    if metadata_usage:
        return metadata_usage
    return None


def estimate_tokens(text):
    return max(1, len(text) // 4)


def ask_model(model_name, prompt, temperature=0.7, top_p=1.0, max_tokens=None):
    start_time = time.time()
    try:
        chat = get_chat_model(model_name, temperature, top_p, max_tokens)
        response = chat.invoke(prompt)
        elapsed = round(time.time() - start_time, 2)
        text = response.content
        tokens = extract_token_usage(response)
        return {
            "model": model_name,
            "prompt": prompt,
            "response": text,
            "time_seconds": elapsed,
            "response_length": len(text),
            "tokens": tokens,
            "estimated_tokens": estimate_tokens(text),
            "error": None,
        }
    except Exception as error:
        elapsed = round(time.time() - start_time, 2)
        return {
            "model": model_name,
            "prompt": prompt,
            "response": "",
            "time_seconds": elapsed,
            "response_length": 0,
            "tokens": None,
            "estimated_tokens": 0,
            "error": str(error),
        }


def ask_judge(question):
    result = ask_model(config.JUDGE_MODEL, question, temperature=0.3)
    if result["error"]:
        return f"Could not generate summary because the judge model failed: {result['error']}"
    return result["response"]
