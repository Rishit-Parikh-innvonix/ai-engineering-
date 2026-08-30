/**
 * Exercise 2 - Week 2 Pipeline: improve one prompt and observe the difference.
 * Takes a week2-js-style extraction system prompt (same shape as
 * ../week2-js/src/prompts.ts's extraction prompts) and improves it by adding an
 * explicit anti-hallucination instruction for missing fields, then runs both the
 * original and improved versions against the SAME email that's missing an email
 * address (a real edge case that came up while testing week2-js's extraction
 * exercise), on both a local and a cloud model.
 * Run: npm run prompt-improvement
 */

import { HumanMessage, SystemMessage } from "@langchain/core/messages";

import * as models from "./models.js";

// Generic extraction prompt - no guidance on what to do when a field simply
// isn't in the text, which is exactly when a model tends to fabricate one.
const ORIGINAL_SYSTEM_PROMPT =
  "You are an assistant that extracts support ticket details from customer emails. " +
  "Extract the customer name, email address, and a one-sentence summary of the issue. " +
  "Reply in exactly this format:\nName: <value>\nEmail: <value>\nSummary: <value>";

// Improved: one added sentence, directly targeting the fabrication failure mode.
const IMPROVED_SYSTEM_PROMPT =
  ORIGINAL_SYSTEM_PROMPT +
  "\n\nIf any field is not explicitly stated in the email text, output exactly MISSING for " +
  "that field - never invent, guess, or infer a plausible-looking value that isn't actually " +
  "present in the text.";

// Real edge case: no email address anywhere in the body. This is what an
// extraction pipeline actually receives sometimes - customers don't always
// include an email in the message body itself.
const TEST_EMAIL =
  "Hi, this is Alex. I can't log into my account since this morning and I have a client " +
  "demo in an hour, please help urgently.";

async function runWith(
  chat: { invoke: (m: (SystemMessage | HumanMessage)[]) => Promise<{ content: unknown }> },
  systemPrompt: string,
  label: string
): Promise<void> {
  const response = await chat.invoke([new SystemMessage(systemPrompt), new HumanMessage(TEST_EMAIL)]);
  console.log(`\n[${label}]`);
  console.log(response.content);
}

async function main(): Promise<void> {
  console.log(`Email under test (deliberately missing an email address):\n"${TEST_EMAIL}"\n`);

  const local = models.getLocalChatModel(0.2);
  const cloud = models.getCloudChatModel(0.2);

  console.log("=".repeat(70));
  console.log("LOCAL model - small model, most likely to fabricate a missing field");
  console.log("=".repeat(70));
  await runWith(local, ORIGINAL_SYSTEM_PROMPT, "BEFORE - no missing-field instruction");
  await runWith(local, IMPROVED_SYSTEM_PROMPT, "AFTER - explicit MISSING instruction");

  console.log(`\n\n${"=".repeat(70)}`);
  console.log("CLOUD model - for comparison, does a larger model need this fix too?");
  console.log("=".repeat(70));
  await runWith(cloud, ORIGINAL_SYSTEM_PROMPT, "BEFORE - no missing-field instruction");
  await runWith(cloud, IMPROVED_SYSTEM_PROMPT, "AFTER - explicit MISSING instruction");

  console.log(
    "\n\nObservation to check by hand: on a first verified run, the local 1B model's BEFORE\n" +
      "output literally fabricated a placeholder value for the missing email (not marked as\n" +
      "missing at all - a value that would silently corrupt a downstream database if unchecked),\n" +
      "while AFTER correctly said MISSING. The cloud model, being larger/better-instructed, may\n" +
      "already handle this gracefully even BEFORE the fix - which is itself the real lesson:\n" +
      "prompt engineering matters more, not less, on smaller/weaker/local models - you can't\n" +
      "assume behavior that holds on a large cloud model also holds on a small local one.\n"
  );
}

main();
