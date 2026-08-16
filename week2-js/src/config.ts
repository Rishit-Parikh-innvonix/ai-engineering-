import path from "node:path";
import { fileURLToPath } from "node:url";
import fs from "node:fs";
import dotenv from "dotenv";

const BASE_DIR = path.dirname(fileURLToPath(import.meta.url)); // .../week2-js/src
const PROJECT_ROOT = path.dirname(BASE_DIR); // .../week2-js
const REPO_ROOT = path.dirname(PROJECT_ROOT); // repo root

// Same convention as week2 (Python): prefer a .env inside week2-js/, fall back
// to the repo-root .env so this also works out of the box in this repo.
const localEnv = path.join(PROJECT_ROOT, ".env");
const rootEnv = path.join(REPO_ROOT, ".env");
dotenv.config({ path: fs.existsSync(localEnv) ? localEnv : rootEnv, override: true });

export const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY ?? "";
export const OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1";

// Same free OpenRouter models as the Python version (week2/config.py), so the
// two implementations are directly comparable. Swap here if any get pulled.
export const STREAM_MODEL = "openai/gpt-oss-20b:free";
export const EXTRACTION_MODEL = "openai/gpt-oss-20b:free";
export const BENCHMARK_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free";
export const TOOL_DEMO_MODEL = "openai/gpt-oss-20b:free";

export const EXTRACTION_MAX_RETRIES = 3;

// USD per 1M tokens, as {input, output}. Every model above is free, so real
// cost is $0 - this exists to demonstrate the technique (see models.ts).
export const MODEL_PRICING_USD_PER_MILLION: Record<string, { input: number; output: number }> = {};

// Exercise 3: kept short and with modest concurrency on purpose - OpenRouter's
// free tier caps at 20 requests/minute, and this script runs the same prompts
// twice (once sequential, once concurrent) in a single process.
export const BENCHMARK_PROMPTS = [
  "In two sentences, explain what an API is.",
  "In two sentences, explain what a database index does.",
  "In two sentences, explain the difference between TCP and UDP.",
  "In two sentences, explain what caching is used for.",
  "In two sentences, explain what a load balancer does.",
  "In two sentences, explain what a race condition is.",
];
export const ASYNC_CONCURRENCY = 3;

// Exercise 2 demo input - realistic customer support emails with varying priority/type.
export const SAMPLE_SUPPORT_EMAILS = [
  "Subject: Can't log in since this morning\n\n" +
    "Hi team, this is Priya Shah (priya.shah@example.com). Since about 9am today I keep " +
    "getting 'invalid credentials' even though I'm sure my password is right, and I have a " +
    "client demo in an hour. Please help ASAP, this is blocking my whole team.",
  "Subject: Question about my invoice\n\n" +
    "Hello, I'm Daniel Cho (daniel.cho99@example.com). I noticed I was charged twice on my " +
    "last invoice. Could someone take a look when they get a chance? Not urgent, just want " +
    "it corrected eventually.",
  "Subject: Feature idea\n\n" +
    "Hey, this is Marta (marta.k@example.com). It would be great if the dashboard supported " +
    "dark mode. No rush at all, just a suggestion for a future release.",
];
