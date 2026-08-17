/**
 * Exercise 1: Prompt Templates - build reusable prompts with ChatPromptTemplate and
 * format/invoke the same template with different inputs, instead of hand-building a
 * new prompt string per call.
 * Run: npm run prompt-templates-demo
 */

import * as config from "./config.js";
import * as models from "./models.js";
import { explainerPromptTemplate, reviewRewritePromptTemplateBase } from "./prompts.js";
import { printPanel } from "./utils.js";

async function demoExplainerTemplate(): Promise<void> {
  const chat = models.getChatModel(config.PROMPT_DEMO_MODEL, { temperature: 0.5 });

  // Same template, three different {topic}/{audience} pairs - this is the reuse
  // ChatPromptTemplate is for: build the template once, format() it per call.
  const inputs = [
    { topic: "recursion", audience: "10 year old" },
    { topic: "recursion", audience: "senior backend engineer" },
    { topic: "database indexing", audience: "product manager" },
  ];

  for (const input of inputs) {
    const messages = await explainerPromptTemplate.formatMessages(input);
    const response = await chat.invoke(messages);
    printPanel(`explainerPromptTemplate({topic: "${input.topic}", audience: "${input.audience}"})`, String(response.content));
  }
}

async function demoPartialTemplate(): Promise<void> {
  const chat = models.getChatModel(config.PROMPT_DEMO_MODEL, { temperature: 0.5 });
  const review = "The shipping took forever and the box arrived crushed, but the product itself works great once I got it set up.";

  // .partial() fixes {style} once per derived template; only {review} needs to be
  // supplied at call time - demonstrates ChatPromptTemplate composition, not just formatting.
  for (const style of ["formal", "cheerful"]) {
    const styledTemplate = await reviewRewritePromptTemplateBase.partial({ style });
    const messages = await styledTemplate.formatMessages({ review });
    const response = await chat.invoke(messages);
    printPanel(`reviewRewritePromptTemplateBase.partial({style: "${style}"})`, String(response.content));
  }
}

async function main(): Promise<void> {
  console.log(`\nExercise 1: Prompt Templates (model=${config.PROMPT_DEMO_MODEL})\n`);
  await demoExplainerTemplate();
  await demoPartialTemplate();
}

main();
