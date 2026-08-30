/**
 * The full RAG capstone: everything vectorSearchDemo.ts does (chunk -> embed ->
 * store -> retrieve), PLUS the missing generation step (retrieve -> augment
 * prompt -> call the LLM -> real answer), run against a real PDF (see
 * config.ts's PDF_PATH - currently the MCC Laws of Cricket) instead of 10
 * toy sentences.
 *
 * Loading uses SimplePdfLoader (pdfLoader.ts) - a from-scratch LangChain
 * Document Loader (implements the same load(): Document[] contract as
 * @langchain/community's real PDFLoader, which we can't install cleanly here
 * - see pdfLoader.ts for why). Because it extracts text per PAGE rather than
 * as one flattened string, every chunk keeps a real page number in metadata,
 * shown in the "Page" column below.
 *
 * Chunking uses LangChain's own RecursiveCharacterTextSplitter (from
 * @langchain/textsplitters - a small standalone package, unlike
 * @langchain/community, so it installs with no conflict). It tries paragraph
 * breaks first, then line breaks, then word breaks, only falling back to a
 * mid-word cut as a last resort - and its splitDocuments() automatically
 * carries each source Document's metadata (our pageNumber) onto every chunk
 * it produces.
 *
 * Retrieval + generation are wired together as a real LangChain chain
 * (RunnableSequence, the same composition primitive from week2-js's
 * structuredOutputStreamingDemo.ts) instead of hand-called steps:
 * ChromaRetriever (chromaRetriever.ts) is a proper LangChain Retriever
 * wrapping our Chroma collection, RunnablePassthrough.assign() carries the
 * retrieved documents alongside the running input so they're still
 * available for display even after the chain moves on to generation, and
 * ChatPromptTemplate/StringOutputParser replace the manual message-building
 * and .content unwrapping we did by hand before.
 *
 * This is the reference implementation for "what's still missing" in
 * docs/RAG_PIPELINE.md - steps 6-8 (augmented prompt, LLM call, final answer)
 * are exactly what this file adds on top of vectorSearchDemo.ts's steps 1-5.
 *
 * Requires: docker run -p 8010:8000 chromadb/chroma, ollama pull nomic-embed-text
 * Run: npm run pdf-rag
 */

import type { ChatOpenAI } from "@langchain/openai";
import { Document } from "@langchain/core/documents";
import type { OllamaEmbeddings } from "@langchain/ollama";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";

import * as config from "./config.js";
import * as models from "./models.js";
import { SimplePdfLoader } from "./pdfLoader.js";
import { getChromaClient, resetCollection, type Collection } from "./chroma.js";
import { ChromaRetriever } from "./chromaRetriever.js";
import { printComparisonTable } from "./utils.js";

const SYSTEM_PROMPT_TEXT =
  "You are a helpful assistant that answers questions using ONLY the context provided below. " +
  "The context comes from a real document (retrieved automatically because it seemed relevant " +
  "to the question). If the context does not actually contain the answer, say plainly that the " +
  "document doesn't cover it - do NOT use outside knowledge, and do NOT guess.";

interface RagChainInput {
  question: string;
}
interface RagChainOutput {
  question: string;
  sourceDocuments: Document[];
  answer: string;
}

function formatContext(docs: Document[]): string {
  return docs.map((doc, i) => `[Chunk ${i + 1}]\n${doc.pageContent}`).join("\n\n");
}

/**
 * The RAG chain: retrieve -> augment -> generate, expressed as one
 * RunnableSequence instead of four hand-called steps.
 * RunnablePassthrough.assign() is what makes this possible without losing
 * the retrieved documents along the way - each .assign() adds a new field
 * to the running object while keeping everything already on it, so
 * `sourceDocuments` (needed for the "Retrieved chunks" table) is still
 * there next to `answer` at the end, even though generation runs after
 * retrieval.
 */
function buildRagChain(retriever: ChromaRetriever, chat: ChatOpenAI) {
  const promptTemplate = ChatPromptTemplate.fromMessages([
    ["system", SYSTEM_PROMPT_TEXT],
    ["human", 'Context:\n"""\n{context}\n"""\n\nQuestion: {question}'],
  ]);

  const generateAnswer = RunnableSequence.from([
    (input: { question: string; sourceDocuments: Document[] }) => ({
      context: formatContext(input.sourceDocuments),
      question: input.question,
    }),
    promptTemplate,
    chat,
    new StringOutputParser(),
  ]);

  return RunnableSequence.from([
    RunnablePassthrough.assign({
      sourceDocuments: (input: Record<string, unknown>) => retriever.invoke(input.question as string),
    }),
    RunnablePassthrough.assign({ answer: generateAnswer }),
  ]) as unknown as RunnableSequence<RagChainInput, RagChainOutput>;
}

async function loadAndChunkPdf(): Promise<Document[]> {
  console.log(`Reading PDF: ${config.PDF_PATH}`);
  const loader = new SimplePdfLoader(config.PDF_PATH);
  const pageDocuments = await loader.load();
  const totalChars = pageDocuments.reduce((sum, d) => sum + d.pageContent.length, 0);
  console.log(`Extracted ${totalChars} characters from ${pageDocuments.length} pages.`);

  // PDF extraction leaves hard line-wraps mid-sentence - collapse whitespace
  // runs first so the splitter's paragraph/line boundaries fall on real
  // prose breaks, not wherever the PDF happened to wrap a line.
  const normalizedPages = pageDocuments.map(
    (doc) => new Document({ pageContent: doc.pageContent.replace(/\s+/g, " ").trim(), metadata: doc.metadata })
  );

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: config.PDF_CHUNK_MAX_CHARS,
    chunkOverlap: config.PDF_CHUNK_OVERLAP_CHARS,
  });
  const chunkedDocuments = await splitter.splitDocuments(normalizedPages);
  const avgLen = Math.round(chunkedDocuments.reduce((sum, d) => sum + d.pageContent.length, 0) / chunkedDocuments.length);
  console.log(`Split into ${chunkedDocuments.length} chunks (LangChain's RecursiveCharacterTextSplitter, target ${config.PDF_CHUNK_MAX_CHARS} chars each, ${config.PDF_CHUNK_OVERLAP_CHARS} char overlap, actual average ${avgLen} chars, each tagged with its source page number).\n`);

  return chunkedDocuments;
}

async function buildVectorStore(chunkedDocuments: Document[]): Promise<{ collection: Collection; embeddings: OllamaEmbeddings }> {
  const embeddings = models.getLocalEmbeddings();
  const client = getChromaClient();
  const collection = await resetCollection(client, config.PDF_CHROMA_COLLECTION);

  const ids: string[] = [];
  const texts: string[] = [];
  const metadatas: Record<string, number>[] = [];
  chunkedDocuments.forEach((doc, i) => {
    ids.push(`chunk-${i}`);
    texts.push(doc.pageContent);
    metadatas.push({ pageNumber: doc.metadata.pageNumber as number });
  });

  console.log(`Embedding ${texts.length} chunks with ${config.LOCAL_EMBEDDING_MODEL} (this is the slow one-time indexing step)...`);
  const start = performance.now();
  const vectors = await embeddings.embedDocuments(texts);
  console.log(`Done in ${((performance.now() - start) / 1000).toFixed(1)}s. Storing in Chroma...\n`);

  await collection.add({ ids, embeddings: vectors, documents: texts, metadatas });

  return { collection, embeddings };
}

/** Runs the RAG chain for one question, printing the retrieved chunks and the answer. */
async function answerQuestion(question: string, ragChain: RunnableSequence<RagChainInput, RagChainOutput>): Promise<void> {
  console.log("=".repeat(70));
  console.log(`Question: ${question}`);
  console.log("=".repeat(70));

  const result = await ragChain.invoke({ question });

  printComparisonTable(
    "Retrieved chunks",
    ["Rank", "Distance", "Page", "Chunk preview"],
    result.sourceDocuments.map((doc, i) => [
      i + 1,
      ((doc.metadata.distance as number | undefined) ?? 0).toFixed(4),
      (doc.metadata.pageNumber as number | undefined) ?? "?",
      doc.pageContent.slice(0, 60) + "...",
    ])
  );

  console.log("\nAnswer:");
  console.log(result.answer);
  console.log();
}

async function main(): Promise<void> {
  const chunkedDocuments = await loadAndChunkPdf();
  const { collection, embeddings } = await buildVectorStore(chunkedDocuments);
  const chat = models.getCloudChatModel(0.2);
  const retriever = new ChromaRetriever({ collection, embeddings, topK: config.PDF_RETRIEVAL_TOP_K });
  const ragChain = buildRagChain(retriever, chat);

  for (const question of config.PDF_RAG_QUESTIONS) {
    await answerQuestion(question, ragChain);
  }

  console.log(
    "Observations to check by hand:\n" +
      "- The last question ('capital of France') is deliberately unrelated to this document.\n" +
      "  A correctly grounded RAG answer should say the document doesn't cover it, NOT confidently\n" +
      "  answer 'Paris' from the model's own outside knowledge - that would mean the system prompt's\n" +
      '  "use ONLY the context" instruction failed to hold, which is a real RAG failure mode\n' +
      "  (retrieval returned irrelevant chunks anyway since it always returns top-K regardless of\n" +
      "  how weak the actual match is, and the model used outside knowledge instead of admitting it).\n" +
      "- Compare the retrieved chunks' distances for the unrelated question against the on-topic ones -\n" +
      "  are they noticeably worse (higher/less similar), even though Chroma still returned SOMETHING?\n" +
      "  A vector search always returns its top-K, even when nothing is actually a good match.\n" +
      "- This is the complete pipeline from docs/RAG_PIPELINE.md: chunk -> embed -> store (this file's\n" +
      "  buildVectorStore) -> retrieve -> augment -> generate (this file's main loop) -> final answer.\n"
  );
}

main();
