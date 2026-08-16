/**
 * Runs Exercise 2's extraction logic directly against the sample emails in config.ts,
 * without needing the Express server running - useful for a quick demo or for testing
 * the reject/retry logic in isolation. Run: npm run extraction-demo
 */

import * as config from "./config.js";
import * as extraction from "./extraction.js";
import { printPanel } from "./utils.js";

async function main(): Promise<void> {
  for (const email of config.SAMPLE_SUPPORT_EMAILS) {
    printPanel("Input email", email);
    const result = await extraction.extractSupportTicket(email);
    if (result.error) {
      printPanel("Extraction FAILED", result.error);
    } else {
      const fields = Object.entries(result.ticket!)
        .map(([key, value]) => `${key}: ${value}`)
        .join("\n");
      printPanel(`Extracted ticket (succeeded on attempt ${result.attempts})`, fields);
    }
  }
}

main();
