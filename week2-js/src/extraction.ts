import { SupportTicketSchema, type SupportTicket } from "./schemas.js";
import * as config from "./config.js";
import * as models from "./models.js";
import { extractionPromptTemplate } from "./prompts.js";

// Hand-written JSON-shape instructions, playing the same role as Python's
// PydanticOutputParser.get_format_instructions() (Level 1 of structured output:
// tell the model the exact shape you expect, in the prompt itself).
function getFormatInstructions(): string {
  return (
    "Respond with a single JSON object matching exactly this shape (no extra text, no markdown fences):\n" +
    "{\n" +
    '  "customer_name": string,        // full name, or "Unknown" if not stated\n' +
    '  "customer_email": string,       // the email address exactly as written in the message\n' +
    '  "priority": "low" | "medium" | "high" | "urgent",\n' +
    '  "issue_type": "billing" | "technical" | "account" | "feature_request" | "bug" | "other",\n' +
    '  "summary": string                // one to two sentence summary of the issue\n' +
    "}"
  );
}

export interface ExtractionResult {
  ticket: SupportTicket | null;
  attempts: number;
  error: string | null;
}

/** Reject-and-retry loop for Exercise 2. Each attempt: call the model in JSON mode,
 * then JSON.parse the raw text (syntax check) and run it through
 * SupportTicketSchema.parse (shape/enum check - the zod equivalent of Pydantic
 * validation). A malformed reply (bad JSON, a priority value outside the allowed
 * enum, a badly-formed email address, ...) throws instead of silently passing
 * through, and we retry rather than return garbage. */
export async function extractSupportTicket(
  emailText: string,
  requestId = "-",
  maxRetries: number = config.EXTRACTION_MAX_RETRIES
): Promise<ExtractionResult> {
  const chat = models.getJsonModeChatModel(config.EXTRACTION_MODEL);
  const messages = await extractionPromptTemplate.formatMessages({
    format_instructions: getFormatInstructions(),
    email_text: emailText,
  });

  let lastError: string | null = null;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    const start = performance.now();
    try {
      const response = await models.ainvokeWithRetry(chat, messages);
      const raw = JSON.parse(response.content as string); // throws SyntaxError on bad JSON
      const ticket = SupportTicketSchema.parse(raw); // throws ZodError on bad shape/enum
      const elapsed = Math.round((performance.now() - start) / 10) / 100;
      const usage = models.extractTokenUsage(response);
      console.log(
        `[${requestId}] extraction attempt ${attempt}/${maxRetries} succeeded in ${elapsed}s (tokens=${JSON.stringify(usage)})`
      );
      return { ticket, attempts: attempt, error: null };
    } catch (error) {
      const isMalformed = error instanceof SyntaxError || (error as { name?: string })?.name === "ZodError";
      if (isMalformed) {
        lastError = `malformed model output: ${error}`;
        console.warn(`[${requestId}] extraction attempt ${attempt}/${maxRetries} rejected - ${String(error).slice(0, 200)}`);
      } else {
        lastError = String(error);
        console.error(`[${requestId}] extraction attempt ${attempt}/${maxRetries} failed - ${lastError.slice(0, 200)}`);
      }
    }
  }

  return {
    ticket: null,
    attempts: maxRetries,
    error: `could not extract a valid ticket after ${maxRetries} attempts. Last error: ${lastError}`,
  };
}
