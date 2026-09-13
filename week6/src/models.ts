import { ChatOpenAI } from "@langchain/openai";

import * as config from "./config.js";

const DEFAULT_HEADERS = {
  "HTTP-Referer": "https://localhost",
  "X-Title": "AI Engineering Week 6 Assignment",
};
const REQUEST_TIMEOUT_MS = 30_000;

export function getCloudChatModel(temperature = 0): ChatOpenAI {
  return new ChatOpenAI({
    model: config.CLOUD_MODEL,
    apiKey: config.OPENROUTER_API_KEY,
    configuration: { baseURL: config.OPENROUTER_BASE_URL, defaultHeaders: DEFAULT_HEADERS },
    temperature,
    timeout: REQUEST_TIMEOUT_MS,
  });
}
