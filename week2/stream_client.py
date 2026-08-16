"""Manual test client for Exercise 1's SSE endpoint. Run the server first:
    uvicorn app:app --reload
then in another terminal:
    python stream_client.py "What is a race condition?"
"""

import json
import sys
import time

import httpx


def stream_question(question, base_url="http://127.0.0.1:8000"):
    start = time.perf_counter()
    event = "message"
    with httpx.Client(timeout=None) as client:
        with client.stream("GET", f"{base_url}/ask/stream", params={"question": question}) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line == "":
                    event = "message"
                    continue
                if line.startswith("event:"):
                    event = line[len("event:") :].strip()
                    continue
                if not line.startswith("data:"):
                    continue

                data = line[len("data:") :].strip()
                if event == "start":
                    print(f"[start] {data}\n")
                elif event == "done":
                    print(f"\n\n[done] {data}")
                elif event == "error":
                    print(f"\n\n[error] {data}")
                else:
                    print(_extract_token(data), end="", flush=True)

    print(f"\nClient-side wall time: {round(time.perf_counter() - start, 2)}s")


def _extract_token(data_json):
    try:
        return json.loads(data_json).get("token", "")
    except json.JSONDecodeError:
        return ""


if __name__ == "__main__":
    question_text = " ".join(sys.argv[1:]) or "Explain what Server-Sent Events are in three sentences."
    stream_question(question_text)
