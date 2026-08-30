/**
 * Shared Chroma helpers - factored out of vectorSearchDemo.ts and
 * pdfRagDemo.ts, which both built a ChromaClient from CHROMA_URL and
 * reset a collection the same way.
 */

import { ChromaClient } from "chromadb";
import * as config from "./config.js";

export type Collection = Awaited<ReturnType<ChromaClient["createCollection"]>>;

export function getChromaClient(): ChromaClient {
  const url = new URL(config.CHROMA_URL);
  return new ChromaClient({
    host: url.hostname,
    port: Number(url.port || (url.protocol === "https:" ? 443 : 80)),
    ssl: url.protocol === "https:",
  });
}

/**
 * Deletes any existing collection with this name and creates a fresh empty
 * one, so re-running a demo doesn't duplicate documents from a prior run.
 * embeddingFunction: null because every embedding here is supplied
 * explicitly (via Ollama) on add()/query() - Chroma never needs to
 * instantiate its own default embedder.
 */
export async function resetCollection(client: ChromaClient, name: string): Promise<Collection> {
  await client.deleteCollection({ name }).catch(() => undefined);
  return client.createCollection({ name, embeddingFunction: null });
}
