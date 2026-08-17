import { ChatPromptTemplate } from "@langchain/core/prompts";

// Exercise 1: Prompt Templates
// ---------------------------------------------------------------------------
// One reusable template, invoked with different {topic}/{audience} values -
// the point of ChatPromptTemplate is that the template is built once and
// formatted many times, rather than re-writing the prompt string per call.
export const explainerPromptTemplate = ChatPromptTemplate.fromMessages([
  ["system", "You are a patient teacher who explains technical topics to a specific audience. Keep it to 3 sentences."],
  ["human", "Explain {topic} to a {audience}."],
]);

// A second reusable template, built with .partial() so the "style" instruction
// is fixed once and only {review} varies per call - demonstrates that
// ChatPromptTemplate supports partial application, not just plain formatting.
export const reviewRewritePromptTemplateBase = ChatPromptTemplate.fromMessages([
  ["system", "Rewrite the customer review the user gives you in a {style} tone. Keep it to 2 sentences."],
  ["human", "{review}"],
]);

// Exercise 3: Review Analysis Pipeline
// ---------------------------------------------------------------------------
// Used by the .withStructuredOutput() implementation - no format instructions
// needed since the schema is bound directly to the model call.
export const reviewAnalysisPromptTemplate = ChatPromptTemplate.fromMessages([
  [
    "system",
    "You are an assistant that analyzes customer product reviews. Base your answer only on what the review actually says.",
  ],
  ["human", "Analyze this review:\n\n\"\"\"\n{review}\n\"\"\""],
]);

// Used by the StructuredOutputParser implementation - {format_instructions} is
// injected by reviewAnalysisPipeline.ts from the parser itself.
export const reviewAnalysisWithFormatInstructionsPromptTemplate = ChatPromptTemplate.fromMessages([
  [
    "system",
    "You are an assistant that analyzes customer product reviews. Base your answer only on what the review actually says.\n\n{format_instructions}",
  ],
  ["human", "Analyze this review:\n\n\"\"\"\n{review}\n\"\"\""],
]);
