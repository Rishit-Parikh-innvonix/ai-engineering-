/**
 * A LangChain Retriever wrapping our Chroma collection.
 *
 * Everywhere else in this project we called collection.query() directly and
 * unpacked the results by hand. LangChain standardizes this behind one
 * interface - BaseRetriever - so that ANY retrieval backend (Chroma, FAISS,
 * a plain in-memory list, a web search API) looks the same to whatever code
 * uses it: hand it a plain text query via .invoke(query), get back Document[].
 * That uniformity is what lets a retriever plug directly into a RunnableSequence
 * chain (see pdfRagDemo.ts's ragChain) alongside prompts and chat models.
 */

import { BaseRetriever, type BaseRetrieverInput } from "@langchain/core/retrievers";
import { Document } from "@langchain/core/documents";
import type { OllamaEmbeddings } from "@langchain/ollama";
import type { Collection } from "./chroma.js";

interface ChromaRetrieverFields extends BaseRetrieverInput {
  collection: Collection;
  embeddings: OllamaEmbeddings;
  topK: number;
}

export class ChromaRetriever extends BaseRetriever {
  lc_namespace = ["week3", "retrievers", "chroma"];

  private readonly collection: Collection;
  private readonly embeddings: OllamaEmbeddings;
  private readonly topK: number;

  constructor(fields: ChromaRetrieverFields) {
    super(fields);
    this.collection = fields.collection;
    this.embeddings = fields.embeddings;
    this.topK = fields.topK;
  }

  async _getRelevantDocuments(query: string): Promise<Document[]> {
    const queryVector = await this.embeddings.embedQuery(query);
    const results = await this.collection.query({ queryEmbeddings: [queryVector], nResults: this.topK });
    const matches = results.rows()[0] ?? [];

    // Distance isn't part of a plain Document, but we still want to display it
    // (as the "Retrieved chunks" table does) - metadata is the sanctioned place
    // to carry retrieval-specific extras like this through the Retriever interface.
    return matches.map(
      (match) =>
        new Document({
          pageContent: match.document ?? "",
          metadata: { ...match.metadata, distance: match.distance },
        })
    );
  }
}
