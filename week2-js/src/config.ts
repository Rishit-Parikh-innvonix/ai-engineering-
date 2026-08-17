import path from "node:path";
import { fileURLToPath } from "node:url";
import fs from "node:fs";
import dotenv from "dotenv";

const BASE_DIR = path.dirname(fileURLToPath(import.meta.url)); // .../week2-js/src
const PROJECT_ROOT = path.dirname(BASE_DIR); // .../week2-js
const REPO_ROOT = path.dirname(PROJECT_ROOT); // repo root

// Prefer a .env inside week2-js/, fall back to the repo-root .env.
const localEnv = path.join(PROJECT_ROOT, ".env");
const rootEnv = path.join(REPO_ROOT, ".env");
dotenv.config({ path: fs.existsSync(localEnv) ? localEnv : rootEnv, override: true });

export const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY ?? "";
export const OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1";

// All exercises go through OpenRouter with a single API key, using a free-tier
// GPT-OSS model (see README for why OpenRouter rather than separate SDKs/keys).
// Note: meta-llama/llama-3.1-8b-instruct:free was dropped from OpenRouter's free
// tier, and google/gemma-4-31b-it:free was hitting persistent upstream 429s, so
// this uses openai/gpt-oss-20b:free instead (already verified working - it's what
// the bonus tool-calling demo below has always used).
export const GPT_OSS_MODEL = "openai/gpt-oss-20b:free";

// Exercise 1 (prompt templates) and Exercise 2 (invocation methods) both use this.
export const PROMPT_DEMO_MODEL = GPT_OSS_MODEL;
export const INVOCATION_MODEL = GPT_OSS_MODEL;

// Exercise 3 (review analysis pipeline).
// gpt-oss-20b:free was unreliable specifically for structured output here - it kept
// wrapping replies in markdown ("**Review A...") that broke JSON parsing, on top of
// frequent 429s. nemotron-3-nano is already proven reliable (week1's EXPERIMENT_MODEL)
// and returned clean, unwrapped JSON in testing.
export const REVIEW_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free";

// Bonus tool-calling demo.
export const TOOL_DEMO_MODEL = GPT_OSS_MODEL;

// Exercise 2: prompts used to compare invoke()/stream() (sequential) against
// concurrent invoke()/stream() (the JS stand-in for Python's ainvoke()/astream(),
// since every LangChain.js call is already Promise-based - see invocationMethodsDemo.ts).
export const INVOCATION_DEMO_PROMPTS = [
  "In two sentences, explain what an API is.",
  "In two sentences, explain what a database index does.",
  "In two sentences, explain the difference between TCP and UDP.",
  "In two sentences, explain what caching is used for.",
];
export const INVOCATION_CONCURRENCY = 3;

// Exercise 3: 10 human-written product reviews with a deliberate spread of
// sentiment (clearly positive, clearly negative, mixed, neutral) so the pipeline's
// output is easy to sanity-check by eye against the schema.
export const SAMPLE_REVIEWS: string[] = [
  "This blender has completely changed my morning routine. It's powerful, easy to clean, and the smoothies come out perfectly smooth every time. Worth every penny.",
  "I was excited for this laptop but the battery barely lasts 3 hours, even on battery saver mode. The screen is nice but I can't use this away from an outlet, which defeats the purpose.",
  "Decent noise-cancelling headphones for the price. Not as good as the premium brands but honestly close enough for daily commuting. Bluetooth pairing was a bit finicky at first.",
  "Absolutely terrible experience. The package arrived damaged, customer support took a week to respond, and when they finally did they just told me to 'try resetting it'. Asking for a refund.",
  "It's fine. Does what it says on the box. Nothing about it stands out as great or bad, I just use it and don't think about it much.",
  "The build quality on this backpack is fantastic - waterproof zippers, reinforced straps, and it still looks brand new after six months of daily use commuting to work.",
  "Mixed feelings here. The app's interface is beautiful and intuitive, but it crashes at least once a day and I've lost unsaved work twice now. Please fix the stability issues.",
  "Best purchase I've made all year. Setup took two minutes, the app connected instantly, and the air quality readings actually match what my more expensive sensor shows.",
  "Way overpriced for what you get. The material feels cheap and after two weeks the stitching on one seam already came undone. Would not recommend at this price point.",
  "Great value coffee maker. Brews quickly, doesn't take up much counter space, and the descaling reminder is a nice touch. My only complaint is the carafe lid is a little loose.",
];
