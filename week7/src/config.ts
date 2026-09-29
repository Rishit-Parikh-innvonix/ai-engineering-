import path from "node:path";
import { fileURLToPath } from "node:url";
import fs from "node:fs";
import dotenv from "dotenv";

const BASE_DIR = path.dirname(fileURLToPath(import.meta.url)); // .../week7/src
export const PROJECT_ROOT = path.dirname(BASE_DIR); // .../week7
const REPO_ROOT = path.dirname(PROJECT_ROOT); // repo root

// Prefer a .env inside week7/, fall back to the repo-root .env.
const localEnv = path.join(PROJECT_ROOT, ".env");
const rootEnv = path.join(REPO_ROOT, ".env");
const envPath = fs.existsSync(localEnv) ? localEnv : rootEnv;

// quiet: true suppresses dotenv's promotional "tip" banner on every run (dotenv 17.x).
dotenv.config({ path: envPath, override: true, quiet: true });

const rawApiKey = process.env.OPENROUTER_API_KEY;
if (!rawApiKey) {
  throw new Error(`OPENROUTER_API_KEY is not set. Add it to ${envPath} (see .env.example).`);
}

export const OPENROUTER_API_KEY: string = rawApiKey;
export const OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1";

// Free OpenRouter models come and go: week5's was pulled before week6, and week6's
// (nex-agi/nex-n2.5-mini:free) was pulled before week7 (OpenRouter now answers 404 "unavailable
// for free"). This model needs reliable tool/function calling, since the planner and retrieval
// agents return structured output. Probed 5 currently-free tool-calling models 3 times each
// (2026-09-28): qwen3.8-27b was rate-limited 3/3, gemma-4-31b 2/3, nemotron-3-super-120b returned
// one malformed response, while dots-3-note-preview succeeded 3/3 at a steady ~4s and
// nemotron-3-ultra-550b succeeded 3/3 but slower (4-10s). If this one is pulled too, that is the
// fallback - and this line is the only place to change.
export const CLOUD_MODEL = "dots-studio/dots-3-note-preview:free";

// Where finished research briefs are saved.
export const REPORTS_DIR = path.join(PROJECT_ROOT, "reports");
