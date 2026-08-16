import { ChatPromptTemplate } from "@langchain/core/prompts";

// Exercise 1: system message sets behavior once; {question} is the human message.
export const STREAM_SYSTEM_PROMPT =
  "You are a concise, helpful assistant answering a user's question in a live chat interface. " +
  "Keep answers clear, accurate, and to the point.";

export const streamPromptTemplate = ChatPromptTemplate.fromMessages([
  ["system", STREAM_SYSTEM_PROMPT],
  ["human", "{question}"],
]);

// Exercise 2: system message constrains the model to the schema and to only using
// facts present in the email; {format_instructions} is injected by extraction.ts.
export const EXTRACTION_SYSTEM_PROMPT =
  "You are a backend service that extracts structured ticket data from raw customer support " +
  "emails. You only use information that is actually present in the email. You never invent an " +
  "email address, name, or fact that isn't stated. Respond with a single JSON object and nothing else.";

export const extractionPromptTemplate = ChatPromptTemplate.fromMessages([
  ["system", EXTRACTION_SYSTEM_PROMPT],
  ["human", '{format_instructions}\n\nCustomer support email:\n"""\n{email_text}\n"""'],
]);
