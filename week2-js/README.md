# Week 2 (JS) - LangChain Fundamentals

Three exercises exploring LangChain.js fundamentals: reusable prompt templates,
the different ways to invoke a chat model (invoke/stream, and their concurrent
equivalents), and an end-to-end structured-output pipeline built with LCEL
Runnables/Chains.

All exercises run through **OpenRouter** with a single API key, using a
free-tier GPT-OSS model. See `src/config.ts` for the exact model ID.

## Setup

```
cd week2-js
npm install
```

Copy `.env.example` to `.env` and put your OpenRouter key in it:
```
OPENROUTER_API_KEY=your_key_here
```

## Exercise 1 - Prompt Templates

Builds two reusable `ChatPromptTemplate`s - one formatted with different
`{topic}`/`{audience}` pairs, one built with `.partial()` to fix a `{style}`
value ahead of time - and shows the same template producing different outputs
without rewriting the prompt string per call.

```
npm run prompt-templates-demo
```

## Exercise 2 - Model Invocation Methods

Compares `invoke()`, `stream()`, and their concurrent equivalents. LangChain.js
has no real sync/async split the way Python does (every call already returns a
Promise), so "ainvoke()"/"astream()" are demonstrated as the same `invoke()`/
`stream()` calls fired concurrently via `Promise.all` instead of sequentially -
same behavior Python shows through separate methods, expressed in JS through
sequential-vs-concurrent await. Prints a timing comparison table.

```
npm run invocation-demo
```

## Exercise 3 - Review Analysis Pipeline

Runs 10 human-written product reviews through an LCEL `RunnableSequence`
(prompt -> model -> parser) that extracts `{sentiment, key_issues, summary}`
as structured JSON, validated against a zod schema (`src/schemas.ts`). Two
implementations of the same chain shape are run and compared:

- **A**: `prompt.pipe(model.withStructuredOutput(schema))` - LangChain manages
  format instructions and JSON parsing/validation internally.
- **B**: `prompt.pipe(model).pipe(StructuredOutputParser)` - format
  instructions are injected into the prompt explicitly, and the parser step
  turns raw text into validated JSON.

Results from both are saved to `output/review_analysis_with_structured_output.json`
and `output/review_analysis_with_output_parser.json`, and a comparison table
(success rate, total time, avg time/review) is printed to the console.

```
npm run review-pipeline
```

## Bonus: tool/function calling demo

```
npm run tool-calling-demo
```
Shows the model requesting `lookup_priority_policy`/`lookup_plan_sla`, the
code executing them locally, and the results being fed back for a final
answer - unchanged from the previous version of this project.

## Bonus: chain basics demo

```
npm run chain-basics-demo
```
The simplest possible LCEL chain (prompt -> model -> `StringOutputParser`),
with no schema or structured output involved - isolates what "a chain" is
before Exercise 3 layers structured output on top.

## Bonus: structured output + streaming demo

```
npm run structured-streaming-demo
```
Combines `.stream()` with structured output and shows the two implementations
behave differently: `withStructuredOutput()` (tool-calling based) typically
delivers the whole object in one chunk, not incrementally, while streaming the
plain-text model call directly shows genuine token-by-token output - which
then has to be fully buffered before it can be parsed into valid JSON.

## Project layout

```
week2-js/
  src/
    config.ts                  models, sample reviews, invocation-demo prompts
    schemas.ts                  zod ReviewAnalysis schema
    prompts.ts                   ChatPromptTemplate definitions (LangChain.js)
    models.ts                    ChatOpenAI factory functions + transport retry
    utils.ts                     console table/panel printers
    promptTemplateDemo.ts         Exercise 1: reusable ChatPromptTemplate demo
    invocationMethodsDemo.ts       Exercise 2: invoke/stream/concurrent comparison
    reviewAnalysisPipeline.ts       Exercise 3: LCEL structured-output pipeline
    toolCallingDemo.ts               bonus: bind_tools-equivalent function calling demo
    chainBasicsDemo.ts               bonus: simplest possible LCEL chain
    structuredOutputStreamingDemo.ts bonus: structured output + streaming comparison
  output/                         JSON results from the review analysis pipeline (gitignored)
```
