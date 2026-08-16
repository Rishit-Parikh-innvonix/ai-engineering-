import json
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import config
import extraction
import models
import prompts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("week2.app")

app = FastAPI(
    title="AI Engineering - Week 2: Working with LLM APIs",
    description="FastAPI endpoints demonstrating SSE streaming and structured extraction over OpenRouter via LangChain.",
)


@app.get("/")
def root():
    return {
        "service": "week2-llm-apis",
        "endpoints": {
            "GET /ask/stream?question=...": "Server-Sent Events stream of an LLM answer (Exercise 1)",
            "POST /extract": "Extract a structured support ticket from raw email text (Exercise 2)",
            "GET /health": "Liveness check",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Exercise 1: FastAPI + SSE streaming
# ---------------------------------------------------------------------------


async def _stream_events(request: Request, question: str, request_id: str):
    start = time.perf_counter()
    chat = models.get_streaming_chat_model(config.STREAM_MODEL)
    messages = prompts.stream_prompt_template.format_messages(question=question)

    chunk_count = 0
    char_count = 0
    disconnected = False

    yield f"event: start\ndata: {json.dumps({'request_id': request_id, 'model': config.STREAM_MODEL})}\n\n"

    try:
        async for chunk in chat.astream(messages):
            # Checked every chunk (not just once) so we stop generating - and stop
            # burning OpenRouter free-tier quota - as soon as the client walks away.
            if await request.is_disconnected():
                disconnected = True
                break
            token = chunk.content
            if not token:
                continue
            chunk_count += 1
            char_count += len(token)
            yield f"data: {json.dumps({'token': token})}\n\n"
    except Exception as error:
        elapsed = round(time.perf_counter() - start, 2)
        logger.error(f"[{request_id}] stream failed after {elapsed}s: {error}")
        yield f"event: error\ndata: {json.dumps({'error': str(error)})}\n\n"
        return

    elapsed = round(time.perf_counter() - start, 2)

    if disconnected:
        logger.warning(
            f"[{request_id}] client disconnected after {chunk_count} chunks / {elapsed}s - stopped generating"
        )
        return

    estimated_tokens = models.estimate_tokens(question) + models.estimate_tokens("x" * char_count)
    logger.info(
        f"[{request_id}] request latency: {elapsed}s - {chunk_count} chunks, {char_count} chars, "
        f"~{estimated_tokens} tokens (estimated)"
    )
    yield f"event: done\ndata: {json.dumps({'elapsed_seconds': elapsed, 'chunks': chunk_count, 'characters': char_count})}\n\n"


@app.get("/ask/stream")
async def ask_stream(request: Request, question: str):
    if not question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] new stream request: {question!r}")
    return StreamingResponse(
        _stream_events(request, question, request_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Request-Id": request_id},
    )


# ---------------------------------------------------------------------------
# Exercise 2: structured extraction from customer support emails
# ---------------------------------------------------------------------------


class ExtractRequest(BaseModel):
    email_text: str = Field(min_length=1)


@app.post("/extract")
async def extract(payload: ExtractRequest):
    request_id = uuid.uuid4().hex[:8]
    start = time.perf_counter()
    result = await extraction.extract_support_ticket(payload.email_text, request_id=request_id)
    elapsed = round(time.perf_counter() - start, 2)

    if result["error"]:
        logger.warning(f"[{request_id}] request latency: {elapsed}s - rejected after {result['attempts']} attempts")
        raise HTTPException(status_code=422, detail=result["error"])

    logger.info(f"[{request_id}] request latency: {elapsed}s - succeeded on attempt {result['attempts']}")
    return {
        "ticket": result["ticket"],
        "attempts": result["attempts"],
        "elapsed_seconds": elapsed,
    }
