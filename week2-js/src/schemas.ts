import { z } from "zod";

// JS/zod equivalent of week2/schemas.py's SupportTicket (Pydantic). Field-level
// .describe() calls feed the format instructions in extraction.ts, the same way
// Pydantic's Field(description=...) feeds PydanticOutputParser in the Python version.
export const SupportTicketSchema = z.object({
  customer_name: z.string().describe("Full name of the customer, or 'Unknown' if not stated in the email"),
  customer_email: z.string().email().describe("The customer's email address exactly as written in the message"),
  priority: z
    .enum(["low", "medium", "high", "urgent"])
    .describe("Urgency of the issue, judged from tone and content"),
  issue_type: z
    .enum(["billing", "technical", "account", "feature_request", "bug", "other"])
    .describe("Category that best describes the issue"),
  summary: z.string().describe("A one to two sentence summary of the customer's issue"),
});

export type SupportTicket = z.infer<typeof SupportTicketSchema>;
