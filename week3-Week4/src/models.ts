import { ChatOpenAI } from "@langchain/openai";
import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";

import * as config from "./config.js";

const DEFAULT_HEADERS = {
  "HTTP-Referer": "https://localhost",
  "X-Title": "AI Engineering Week 3 Assignment",
};
const REQUEST_TIMEOUT_MS = 30_000;

export function getCloudChatModel(temperature = 0.7): ChatOpenAI {
  return new ChatOpenAI({
    model: config.CLOUD_MODEL,
    apiKey: config.OPENROUTER_API_KEY,
    configuration: { baseURL: config.OPENROUTER_BASE_URL, defaultHeaders: DEFAULT_HEADERS },
    temperature,
    timeout: REQUEST_TIMEOUT_MS,
  });
}

export function getLocalChatModel(temperature = 0.7): ChatOllama {
  return new ChatOllama({
    model: config.LOCAL_MODEL,
    baseUrl: config.OLLAMA_BASE_URL,
    temperature,
  });
}

export function getLocalEmbeddings(): OllamaEmbeddings {
  return new OllamaEmbeddings({
    model: config.LOCAL_EMBEDDING_MODEL,
    baseUrl: config.OLLAMA_BASE_URL,
  });
}
