/**
 * Bonus: Chains/LCEL basics - the simplest possible chain, with no schemas or
 * structured output involved, just to see prompt -> model -> parser composition
 * in isolation. Contrast with the more complex chains in reviewAnalysisPipeline.ts.
 * Run: npm run chain-basics-demo
 */

import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence } from "@langchain/core/runnables";

import * as config from "./config.js";
import * as models from "./models.js";

async function main(): Promise<void> {
  const prompt = ChatPromptTemplate.fromMessages([
    ["human", "Write a one-sentence tagline for a product called {product}."],
  ]);
  const chat = models.getChatModel(config.PROMPT_DEMO_MODEL, { temperature: 0.7 });
  const outputParser = new StringOutputParser(); // just extracts the plain text reply - no schema to validate against

  // A 3-step chain: format the prompt -> call the model -> unwrap the AI reply to a plain string.
  const chain = RunnableSequence.from([prompt, chat, outputParser]);

  const tagline = await chain.invoke({ product: "a smart water bottle" });
  console.log(`\nChain input: {product: "a smart water bottle"}`);
  console.log(`Chain output (plain string, not an AIMessage): "${tagline}"\n`);
}

main();
