import { ChatOpenAI } from "@langchain/openai";
import type { BaseMessage, AIMessage } from "@langchain/core/messages";

import * as config from "./config.js";

const DEFAULT_HEADERS = {
  "HTTP-Referer": "https://localhost",
  "X-Title": "AI Engineering Week 2 Assignment (JS)",
};

// Free OpenRouter models occasionally never respond at all (as opposed to responding with
// an error) - without a timeout, that call would hang the process forever instead of
// failing in a way ainvokeWithRetry / safeInvoke can catch.
const REQUEST_TIMEOUT_MS = 30_000;

interface ChatModelOptions {
  temperature?: number;
  topP?: number;
  maxTokens?: number;
  streaming?: boolean;
}

export function getChatModel(
  modelName: string,
  { temperature = 0.7, topP = 1.0, maxTokens, streaming = false }: ChatModelOptions = {}
): ChatOpenAI {
  return new ChatOpenAI({
    model: modelName,
    apiKey: config.OPENROUTER_API_KEY,
    configuration: { baseURL: config.OPENROUTER_BASE_URL, defaultHeaders: DEFAULT_HEADERS },
    temperature,
    topP,
    maxTokens,
    streaming,
    timeout: REQUEST_TIMEOUT_MS,
  });
}

/** Same model, with streaming: true so the provider sends the response token-by-token
 * instead of one chunk at the end. Used by invocationMethodsDemo.ts's stream()/astream() runs. */
export function getStreamingChatModel(modelName: string, temperature = 0.7): ChatOpenAI {
  return getChatModel(modelName, { temperature, streaming: true });
}

function isRetryableError(error: unknown): boolean {
  const status = (error as { status?: number; response?: { status?: number } })?.status
    ?? (error as { response?: { status?: number } })?.response?.status;
  // Retry on network errors (no status at all) and on 429 / 5xx from the provider.
  if (status === undefined) return true;
  return status === 429 || status >= 500;
}

/** Transport-level retry: exponential backoff on network errors / 429s / provider 5xxs.
 * Used by reviewAnalysisPipeline.ts so one flaky free-model response doesn't fail an
 * entire 10-review batch. */
export async function ainvokeWithRetry(chat: ChatOpenAI, messages: BaseMessage[]): Promise<AIMessage> {
  const maxAttempts = 3;
  let lastError: unknown;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await chat.invoke(messages);
    } catch (error) {
      lastError = error;
      if (!isRetryableError(error) || attempt === maxAttempts) {
        throw error;
      }
      const waitMs = Math.min(8000, 1000 * 2 ** (attempt - 1));
      await new Promise((resolve) => setTimeout(resolve, waitMs));
    }
  }

  throw lastError;
}

export function extractTokenUsage(aiMessage: AIMessage): Record<string, unknown> | null {
  const usage = (aiMessage as { usage_metadata?: Record<string, unknown> }).usage_metadata;
  if (usage) {
    return {
      prompt_tokens: usage.input_tokens,
      completion_tokens: usage.output_tokens,
      total_tokens: usage.total_tokens,
    };
  }
  const metadataUsage = aiMessage.response_metadata?.tokenUsage ?? aiMessage.response_metadata?.token_usage;
  return (metadataUsage as Record<string, unknown> | undefined) ?? null;
}
