/**
 * Exercise 4 - Vector Search: embed a small knowledge base, store it in Chroma
 * (running via Docker), and run similarity search for a few queries that don't
 * share exact keywords with the documents they should match - the point is to
 * show similarity search working on MEANING, not keyword overlap.
 *
 * Uses the `chromadb` client directly (not @langchain/community's wrapper) -
 * its Chroma integration pulls in a heavy, unrelated peer dependency chain
 * (browser-automation tooling) purely for this one small feature, so a thin
 * direct client + LangChain's OllamaEmbeddings for the vectors is simpler here.
 *
 * Requires:
 *   docker run -p 8010:8000 chromadb/chroma
 *   ollama pull nomic-embed-text   (embeddings, fully local/free)
 *
 * Run: npm run vector-search
 */

import * as config from "./config.js";
import * as models from "./models.js";
import { getChromaClient, resetCollection } from "./chroma.js";
import { printComparisonTable } from "./utils.js";

async function main(): Promise<void> {
  console.log(`Chroma:     ${config.CHROMA_URL} (collection "${config.CHROMA_COLLECTION}")`);
  console.log(`Embeddings: ${config.LOCAL_EMBEDDING_MODEL} (via Ollama, local, free)\n`);

  const embeddings = models.getLocalEmbeddings();
  const client = getChromaClient();
  const collection = await resetCollection(client, config.CHROMA_COLLECTION);

  console.log(`Embedding and storing ${config.VECTOR_SEARCH_DOCUMENTS.length} documents in Chroma...`);
  const documentVectors = await embeddings.embedDocuments(config.VECTOR_SEARCH_DOCUMENTS);
  await collection.add({
    ids: config.VECTOR_SEARCH_DOCUMENTS.map((_, i) => `doc-${i}`),
    embeddings: documentVectors,
    documents: config.VECTOR_SEARCH_DOCUMENTS,
  });
  console.log("Done.\n");

  for (const query of config.VECTOR_SEARCH_QUERIES) {
    console.log("=".repeat(70));
    console.log(`Query: "${query}"`);
    console.log("=".repeat(70));

    const queryVector = await embeddings.embedQuery(query);
    const results = await collection.query({ queryEmbeddings: [queryVector], nResults: 3 });
    const matches = results.rows()[0] ?? [];

    const rows = matches.map((match, rank) => [rank + 1, (match.distance ?? 0).toFixed(4), match.document ?? ""]);
    printComparisonTable("Top matches (lower distance = more similar)", ["Rank", "Distance", "Document"], rows);
    console.log();
  }

  console.log(
    "Observations to check by hand:\n" +
      '- None of the queries share many exact words with the document they best match (e.g.\n' +
      '  "money back" vs "refunds", "locked out" vs "reset your password") - that\'s the point:\n' +
      "  this is matching on MEANING (embedding similarity), not keyword search.\n" +
      "- Lower distance = more similar (Chroma's default here is a distance metric, not a\n" +
      "  similarity score - smaller number means closer vectors).\n" +
      "- This is retrieval only - no LLM call happens in this file. RAG (see docs/RAG_PIPELINE.md)\n" +
      "  is this exact step PLUS feeding the top results into a model as context.\n"
  );
}

main();
