import path from "node:path";
import { fileURLToPath } from "node:url";
import fs from "node:fs";
import dotenv from "dotenv";

const BASE_DIR = path.dirname(fileURLToPath(import.meta.url)); // .../week3/src
const PROJECT_ROOT = path.dirname(BASE_DIR); // .../week3
const REPO_ROOT = path.dirname(PROJECT_ROOT); // repo root

// Prefer a .env inside week3/, fall back to the repo-root .env.
const localEnv = path.join(PROJECT_ROOT, ".env");
const rootEnv = path.join(REPO_ROOT, ".env");
dotenv.config({ path: fs.existsSync(localEnv) ? localEnv : rootEnv, override: true });

export const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY ?? "";
export const OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1";

// The "cloud" side of every exercise here. week2-js's openai/gpt-oss-20b:free was
// dropped from OpenRouter's free tier since that project was built - verified
// working as of 2026-08-29 against OpenRouter's live /models list.
export const CLOUD_MODEL = "minimax/minimax-m2.7:free";

// Ollama (local). Requires `ollama serve` running (the Windows installer starts
// it automatically) and `ollama pull llama3.2:1b` / `ollama pull nomic-embed-text`
// done once beforehand.
export const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL ?? "http://127.0.0.1:11434";
export const LOCAL_MODEL = "llama3.2:1b";
export const LOCAL_EMBEDDING_MODEL = "nomic-embed-text";

// Chroma vector DB, run via Docker: docker run -p 8010:8000 chromadb/chroma
// (8010, not 8000, so it doesn't collide with week2-js's Express server default).
export const CHROMA_URL = process.env.CHROMA_URL ?? "http://localhost:8010";
export const CHROMA_COLLECTION = "week3-vector-search-demo";

// Exercise 1: same task, several different prompt phrasings.
export const PROMPT_EXPERIMENT_TASK = "Explain recursion to someone learning to program for the first time.";

// Exercise 4: a small, deliberately mixed set of short "documents" (support-style
// knowledge base snippets) so similarity search results are easy to sanity-check by eye.
export const VECTOR_SEARCH_DOCUMENTS: string[] = [
  "To reset your password, go to Settings > Security > Reset Password and follow the emailed link.",
  "Refunds are issued within 5-7 business days after a return is received at our warehouse.",
  "Our API rate limit is 100 requests per minute per API key on the free tier, 1000 on paid tiers.",
  "You can export your data as a CSV or JSON file from the Account > Data Export page at any time.",
  "Two-factor authentication can be enabled under Settings > Security > Two-Factor Authentication.",
  "Shipping usually takes 3-5 business days domestically and 10-14 business days internationally.",
  "The mobile app supports offline mode - changes sync automatically once you're back online.",
  "Enterprise plan customers get a dedicated account manager and a 30-minute SLA on support tickets.",
  "To cancel your subscription, go to Billing > Manage Plan > Cancel Subscription before your renewal date.",
  "Our REST API uses standard HTTP status codes; a 429 response means you've hit the rate limit.",
];

export const VECTOR_SEARCH_QUERIES: string[] = [
  "How do I get my money back after returning an item?",
  "I'm locked out of my account, what do I do?",
  "What happens if I call the API too many times?",
];

// Full RAG capstone (pdfRagDemo.ts) - the official MCC Laws of Cricket (2017
// Code, 3rd edition, 2022), used to demonstrate chunking + retrieval +
// generation end to end on real content far too large to paste into a single
// prompt.
export const PDF_PATH = path.join(PROJECT_ROOT, "data", "laws-of-cricket-2017.pdf");
export const PDF_CHROMA_COLLECTION = "week3-pdf-rag-cricket";
export const PDF_CHUNK_MAX_CHARS = 800;
export const PDF_CHUNK_OVERLAP_CHARS = 150;
export const PDF_RETRIEVAL_TOP_K = 4;

export const PDF_RAG_QUESTIONS: string[] = [
  "What is the LBW (leg before wicket) law?",
  "How many players are on each side in a match?",
  "What happens if a fielder deliberately fields the ball with their foot?",
  "What is the capital of France?", // deliberately unrelated - should say "not in the document", not guess
];
