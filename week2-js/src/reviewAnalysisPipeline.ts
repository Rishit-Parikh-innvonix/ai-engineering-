/**
 * Exercise 3: Review Analysis Pipeline - run 10 human-written reviews through a
 * LangChain LCEL chain (RunnableSequence: prompt -> model -> parser) that produces
 * structured JSON matching ReviewAnalysisSchema, then save the results to JSON.
 *
 * Two implementations of the same pipeline are run and compared, per the assignment:
 *   A) prompt.pipe(model.withStructuredOutput(schema))     - schema bound to the call
 *   B) prompt.pipe(model).pipe(StructuredOutputParser)      - schema enforced by a parser step
 * Run: npm run review-pipeline
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { RunnableSequence } from "@langchain/core/runnables";
import { StructuredOutputParser } from "@langchain/core/output_parsers";

import * as config from "./config.js";
import * as models from "./models.js";
import { ReviewAnalysisSchema, type ReviewAnalysis } from "./schemas.js";
import { reviewAnalysisPromptTemplate, reviewAnalysisWithFormatInstructionsPromptTemplate } from "./prompts.js";
import { printComparisonTable } from "./utils.js";

const OUTPUT_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "output");

interface PipelineResult {
  review: string;
  analysis: ReviewAnalysis | null;
  error: string | null;
}

interface PipelineRun {
  label: string;
  results: PipelineResult[];
  totalSeconds: number;
}

/** Implementation A: schema bound straight to the model via withStructuredOutput() -
 * LangChain handles the format instructions and parsing/validation internally. */
function buildStructuredOutputChain(): RunnableSequence {
  const chat = models.getChatModel(config.REVIEW_MODEL, { temperature: 0.2 });
  const structuredChat = chat.withStructuredOutput(ReviewAnalysisSchema);
  return RunnableSequence.from([reviewAnalysisPromptTemplate, structuredChat]);
}

/** Implementation B: explicit StructuredOutputParser - format instructions are injected
 * into the prompt ourselves, and the parser step turns the raw text reply into JSON,
 * then validates it against the zod schema. */
function buildOutputParserChain(): { chain: RunnableSequence; parser: StructuredOutputParser<typeof ReviewAnalysisSchema> } {
  const parser = StructuredOutputParser.fromZodSchema(ReviewAnalysisSchema);
  const chat = models.getChatModel(config.REVIEW_MODEL, { temperature: 0.2 });
  const chain = RunnableSequence.from([reviewAnalysisWithFormatInstructionsPromptTemplate, chat, parser]);
  return { chain, parser };
}

async function runPipeline(
  label: string,
  invokeOne: (review: string) => Promise<ReviewAnalysis>
): Promise<PipelineRun> {
  const results: PipelineResult[] = [];
  const start = performance.now();

  for (const [i, review] of config.SAMPLE_REVIEWS.entries()) {
    process.stdout.write(`  review ${i + 1}/${config.SAMPLE_REVIEWS.length}... `);
    try {
      const analysis = await invokeOne(review);
      results.push({ review, analysis, error: null });
      console.log("ok");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      results.push({ review, analysis: null, error: message });
      console.log(`failed (${message})`);
    }
  }

  const totalSeconds = (performance.now() - start) / 1000;
  return { label, results, totalSeconds };
}

function saveResults(run: PipelineRun, filename: string): void {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  const outPath = path.join(OUTPUT_DIR, filename);
  fs.writeFileSync(outPath, JSON.stringify(run.results, null, 2));
  console.log(`Saved ${run.results.length} results to ${path.relative(process.cwd(), outPath)}`);
}

function summarizeRow(run: PipelineRun): (string | number)[] {
  const succeeded = run.results.filter((r) => r.analysis !== null).length;
  const avgSeconds = run.results.length ? run.totalSeconds / run.results.length : 0;
  return [run.label, `${succeeded}/${run.results.length}`, run.totalSeconds.toFixed(2), avgSeconds.toFixed(2)];
}

async function main(): Promise<void> {
  console.log(
    `\nExercise 3: Review Analysis Pipeline (${config.SAMPLE_REVIEWS.length} reviews, model=${config.REVIEW_MODEL})\n`
  );

  console.log("Running Implementation A: prompt.pipe(model.withStructuredOutput(schema))...");
  const structuredOutputChain = buildStructuredOutputChain();
  const runA = await runPipeline("withStructuredOutput()", (review) =>
    structuredOutputChain.invoke({ review }) as Promise<ReviewAnalysis>
  );
  saveResults(runA, "review_analysis_with_structured_output.json");

  console.log("\nRunning Implementation B: prompt.pipe(model).pipe(StructuredOutputParser)...");
  const { chain: outputParserChain, parser } = buildOutputParserChain();
  const runB = await runPipeline("StructuredOutputParser", async (review) => {
    const formatted = await outputParserChain.invoke({
      review,
      format_instructions: parser.getFormatInstructions(),
    });
    return formatted as ReviewAnalysis;
  });
  saveResults(runB, "review_analysis_with_output_parser.json");

  printComparisonTable(
    "Exercise 3: Pipeline Comparison",
    ["Implementation", "Success", "Total time (s)", "Avg time/review (s)"],
    [summarizeRow(runA), summarizeRow(runB)]
  );

  console.log(
    "\nBoth implementations run the same RunnableSequence (prompt -> model -> structured result) LCEL " +
      "chain shape over the same 10 reviews. withStructuredOutput() lets LangChain manage the format " +
      "instructions and JSON parsing/validation internally (fewer moving parts, less control); the " +
      "explicit StructuredOutputParser chain shows exactly what's happening at each step (format " +
      "instructions injected into the prompt, raw text parsed and validated afterward) - more control, " +
      "more code. Results for both are saved under output/ for inspection.\n"
  );
}

main();
