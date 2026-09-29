import path from "node:path";
import { fileURLToPath } from "node:url";
import fs from "node:fs";
import dotenv from "dotenv";

const BASE_DIR = path.dirname(fileURLToPath(import.meta.url)); // .../week6/src
export const PROJECT_ROOT = path.dirname(BASE_DIR); // .../week6
const REPO_ROOT = path.dirname(PROJECT_ROOT); // repo root

// Prefer a .env inside week6/, fall back to the repo-root .env.
const localEnv = path.join(PROJECT_ROOT, ".env");
const rootEnv = path.join(REPO_ROOT, ".env");
const envPath = fs.existsSync(localEnv) ? localEnv : rootEnv;

// quiet: true suppresses dotenv's promotional "tip" banner on every run (dotenv 17.x) - unrelated
// to loading the actual env file, just console noise pointing at an external site.
dotenv.config({ path: envPath, override: true, quiet: true });

const rawApiKey = process.env.OPENROUTER_API_KEY;
if (!rawApiKey) {
  throw new Error(`OPENROUTER_API_KEY is not set. Add it to ${envPath} (see .env.example).`);
}

// Typed as `string`, not `string | undefined` - the check above already guarantees it.
export const OPENROUTER_API_KEY: string = rawApiKey;
export const OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1";

// week5's free-tier model (minimax/minimax-m2.7:free) was pulled from OpenRouter's free tier
// between week5 and week6 (confirmed live: OpenRouter now 404s it, pointing at a paid slug
// instead) - checked OpenRouter's /api/v1/models for currently free models whose
// supported_parameters list both "tools" and "tool_choice".
//
// Initially picked nex-agi/nex-n2.5-pro:free, but it hung completely (no response at all,
// confirmed by testing it directly with zero MCP involved) on two separate live runs, so
// switched to nex-n2.5-mini:free. That one later started hanging too (2026-09-15: confirmed
// via a direct curl call, zero MCP involved, full 35s timeout with no response at all -
// upstream model reliability, not this project). Switched to nvidia/nemotron-3-super-120b-a12b:free
// next - looked solid initially (4/4 direct calls), but within the same session started
// failing ~60% of calls (3/5 direct test calls) with OpenRouter's Nvidia provider returning
// "Service temporarily overloaded" (502) wrapped inside a 200 OK body with zero choices -
// invokeWithRetry() in models.ts exists specifically to survive that failure mode, but a 60%
// failure rate beats even a 3-attempt retry too often to be worth keeping as the default.
// Re-probed and switched to cohere/north-mini-code:free - 5/5 direct calls succeeded, and it
// completed all 3 exercises live with zero retries needed on either agent call.
export const CLOUD_MODEL = "cohere/north-mini-code:free";
