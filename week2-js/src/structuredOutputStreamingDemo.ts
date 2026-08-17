/**
 * Bonus: Structured Output + Streaming - combining chat.withStructuredOutput(schema)
 * with .stream() instead of .invoke(), so you see the structured object being filled
 * in incrementally (more fields/array items appear as the model keeps generating)
 * instead of waiting for the whole response before you get anything back.
 * Run: npm run structured-streaming-demo
 */

import * as config from "./config.js";
import * as models from "./models.js";
import { ReviewAnalysisSchema } from "./schemas.js";
import { reviewAnalysisPromptTemplate, reviewAnalysisWithFormatInstructionsPromptTemplate } from "./prompts.js";
import { StructuredOutputParser } from "@langchain/core/output_parsers";

const REVIEW =
  "Mixed feelings here. The app's interface is beautiful and intuitive, but it crashes at least " +
  "once a day and I've lost unsaved work twice now. Please fix the stability issues.";

/** Part 1: .stream() on a withStructuredOutput() chain - schema bound via tool-calling.
 * On most providers this arrives as ONE chunk, not incrementally - tool-call arguments
 * are typically only emitted once complete, since partial/invalid JSON can't be parsed
 * as valid tool-call args mid-stream. Streaming the call doesn't guarantee incremental
 * output; it depends on what's actually being streamed underneath. */
async function demoStructuredOutputStream(): Promise<void> {
  console.log("Part 1: .stream() on a withStructuredOutput() chain (tool-calling based)\n");

  const chat = models.getChatModel(config.REVIEW_MODEL, { temperature: 0.2, streaming: true });
  const structuredChat = chat.withStructuredOutput(ReviewAnalysisSchema);
  const chain = reviewAnalysisPromptTemplate.pipe(structuredChat);

  const stream = await chain.stream({ review: REVIEW });
  let finalChunk: unknown;
  let chunkCount = 0;

  for await (const chunk of stream) {
    chunkCount += 1;
    finalChunk = chunk;
    console.log(`  chunk ${chunkCount}:`, JSON.stringify(chunk));
  }

  console.log(`\n  -> ${chunkCount} chunk(s) received.`);
  if (chunkCount === 1) {
    console.log(
      "  -> Only one chunk: this provider sends tool-call arguments as a single complete\n" +
        "     block once generation finishes, not token-by-token. This is normal - structured\n" +
        "     output via tool-calling usually isn't truly incremental."
    );
  }
  console.log("\nFinal object:", JSON.stringify(finalChunk), "\n");
}

/** Part 2: .stream() on the plain-text StructuredOutputParser chain's model step, to show
 * what GENUINE token-by-token streaming looks like for contrast - real text arriving in
 * small pieces, which only becomes valid JSON once the stream is fully consumed and
 * concatenated. You can't safely parser.parse() a partial chunk - JSON isn't valid until
 * the closing brace arrives - so this prints raw text chunks, then parses once at the end. */
async function demoRawTextStreamThenParse(): Promise<void> {
  console.log("Part 2: .stream() on the model directly (no schema binding) + parse after\n");

  const parser = StructuredOutputParser.fromZodSchema(ReviewAnalysisSchema);
  const chat = models.getChatModel(config.REVIEW_MODEL, { temperature: 0.2, streaming: true });
  const messages = await reviewAnalysisWithFormatInstructionsPromptTemplate.formatMessages({
    review: REVIEW,
    format_instructions: parser.getFormatInstructions(),
  });

  const stream = await chat.stream(messages);
  let rawText = "";
  let chunkCount = 0;

  process.stdout.write("  raw text as it streams in: ");
  for await (const chunk of stream) {
    chunkCount += 1;
    const piece = String(chunk.content ?? "");
    rawText += piece;
    process.stdout.write(piece);
  }
  console.log(`\n\n  -> ${chunkCount} chunk(s) received (this is what real incremental streaming looks like).`);

  const parsed = await parser.parse(rawText);
  console.log("\nParsed once the full text arrived:", JSON.stringify(parsed), "\n");
}

async function main(): Promise<void> {
  console.log(`\nStructured Output + Streaming (model=${config.REVIEW_MODEL})`);
  console.log(`Review: "${REVIEW}"\n`);

  await demoStructuredOutputStream();
  await demoRawTextStreamThenParse();

  console.log(
    "Takeaway: streaming + structured output don't automatically combine into incremental\n" +
      "structured data. Tool-calling-based structured output (Part 1) usually arrives whole.\n" +
      "Genuine incremental streaming (Part 2) only works on RAW TEXT - to get it as structured\n" +
      "data you must buffer the full text and parse it after the stream ends, same as the\n" +
      "StructuredOutputParser chain in reviewAnalysisPipeline.ts does with .invoke().\n"
  );
}

main();
