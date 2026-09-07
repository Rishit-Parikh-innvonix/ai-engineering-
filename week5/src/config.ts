import path from "node:path";
import { fileURLToPath } from "node:url";
import fs from "node:fs";
import dotenv from "dotenv";

const BASE_DIR = path.dirname(fileURLToPath(import.meta.url)); // .../week5/src
const PROJECT_ROOT = path.dirname(BASE_DIR); // .../week5
const REPO_ROOT = path.dirname(PROJECT_ROOT); // repo root

// Prefer a .env inside week5/, fall back to the repo-root .env.
const localEnv = path.join(PROJECT_ROOT, ".env");
const rootEnv = path.join(REPO_ROOT, ".env");
const envPath = fs.existsSync(localEnv) ? localEnv : rootEnv;

// quiet: true suppresses dotenv's promotional "tip" banner on every run (added in dotenv 17.x) -
// unrelated to loading the actual env file, just console noise pointing at an external site.
dotenv.config({ path: envPath, override: true, quiet: true });

const rawApiKey = process.env.OPENROUTER_API_KEY;
if (!rawApiKey) {
  throw new Error(`OPENROUTER_API_KEY is not set. Add it to ${envPath} (see .env.example).`);
}

// Typed as `string`, not `string | undefined` - the check above already guarantees it, so every
// consumer (models.ts) gets that guarantee too instead of having to handle "undefined" again.
export const OPENROUTER_API_KEY: string = rawApiKey;
export const OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1";

// Same free-tier model week3-Week4 uses - verified working against OpenRouter's
// live /models list as of 2026-08-29. Needs real tool-calling support for
// this week's exercises to work at all.
export const CLOUD_MODEL = "minimax/minimax-m2.7:free";
