import { randomUUID } from "node:crypto";
import express, { type Request, type Response } from "express";

import * as config from "./config.js";
import * as models from "./models.js";
import * as extraction from "./extraction.js";
import { streamPromptTemplate } from "./prompts.js";

const app = express();
app.use(express.json());

app.get("/", (_req, res) => {
  res.json({
    service: "week2-llm-apis-js",
    endpoints: {
      "GET /ask/stream?question=...": "Server-Sent Events stream of an LLM answer (Exercise 1)",
      "POST /extract": "Extract a structured support ticket from raw email text (Exercise 2)",
      "GET /health": "Liveness check",
    },
  });
});

app.get("/health", (_req, res) => {
  res.json({ status: "ok" });
});

// ---------------------------------------------------------------------------
// Exercise 1: Express + SSE streaming
// ---------------------------------------------------------------------------

async function streamEvents(res: Response, question: string, requestId: string, isDisconnected: () => boolean): Promise<void> {
  const start = performance.now();
  const chat = models.getStreamingChatModel(config.STREAM_MODEL);
  const messages = await streamPromptTemplate.formatMessages({ question });

  let chunkCount = 0;
  let charCount = 0;
  let disconnected = false;

  res.write(`event: start\ndata: ${JSON.stringify({ request_id: requestId, model: config.STREAM_MODEL })}\n\n`);

  try {
    const tokenStream = await chat.stream(messages);
    for await (const chunk of tokenStream) {
      // Checked every chunk (not just once) so we stop generating - and stop
      // burning OpenRouter free-tier quota - as soon as the client walks away.
      if (isDisconnected()) {
        disconnected = true;
        break;
      }
      const token = chunk.content as string;
      if (!token) continue;
      chunkCount += 1;
      charCount += token.length;
      res.write(`data: ${JSON.stringify({ token })}\n\n`);
    }
  } catch (error) {
    const elapsed = Math.round((performance.now() - start) / 10) / 100;
    console.error(`[${requestId}] stream failed after ${elapsed}s: ${error}`);
    res.write(`event: error\ndata: ${JSON.stringify({ error: String(error) })}\n\n`);
    res.end();
    return;
  }

  const elapsed = Math.round((performance.now() - start) / 10) / 100;

  if (disconnected) {
    console.warn(`[${requestId}] client disconnected after ${chunkCount} chunks / ${elapsed}s - stopped generating`);
    res.end();
    return;
  }

  const estimatedTokens = models.estimateTokens(question) + models.estimateTokens("x".repeat(charCount));
  console.log(
    `[${requestId}] request latency: ${elapsed}s - ${chunkCount} chunks, ${charCount} chars, ~${estimatedTokens} tokens (estimated)`
  );
  res.write(
    `event: done\ndata: ${JSON.stringify({ elapsed_seconds: elapsed, chunks: chunkCount, characters: charCount })}\n\n`
  );
  res.end();
}

app.get("/ask/stream", async (req: Request, res: Response) => {
  const question = typeof req.query.question === "string" ? req.query.question : "";
  if (!question.trim()) {
    res.status(400).json({ detail: "question must not be empty" });
    return;
  }

  const requestId = randomUUID().slice(0, 8);
  console.log(`[${requestId}] new stream request: ${JSON.stringify(question)}`);

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("X-Request-Id", requestId);
  res.flushHeaders();

  // Express/Node equivalent of FastAPI's `await request.is_disconnected()`: the
  // 'close' event fires when the client's TCP connection goes away mid-stream.
  let disconnected = false;
  res.on("close", () => {
    disconnected = true;
  });

  await streamEvents(res, question, requestId, () => disconnected);
});

// ---------------------------------------------------------------------------
// Exercise 2: structured extraction from customer support emails
// ---------------------------------------------------------------------------

app.post("/extract", async (req: Request, res: Response) => {
  const emailText = req.body?.email_text;
  if (typeof emailText !== "string" || emailText.length < 1) {
    res.status(422).json({ detail: "email_text must be a non-empty string" });
    return;
  }

  const requestId = randomUUID().slice(0, 8);
  const start = performance.now();
  const result = await extraction.extractSupportTicket(emailText, requestId);
  const elapsed = Math.round((performance.now() - start) / 10) / 100;

  if (result.error) {
    console.warn(`[${requestId}] request latency: ${elapsed}s - rejected after ${result.attempts} attempts`);
    res.status(422).json({ detail: result.error });
    return;
  }

  console.log(`[${requestId}] request latency: ${elapsed}s - succeeded on attempt ${result.attempts}`);
  res.json({ ticket: result.ticket, attempts: result.attempts, elapsed_seconds: elapsed });
});

const PORT = Number(process.env.PORT ?? 8000);
app.listen(PORT, () => {
  console.log(`AI Engineering - Week 2 (JS): Working with LLM APIs`);
  console.log(`Uvicorn's Express equivalent running on http://127.0.0.1:${PORT}`);
});
