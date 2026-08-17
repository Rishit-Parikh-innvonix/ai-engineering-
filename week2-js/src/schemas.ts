import { z } from "zod";

// Exercise 3: zod equivalent of a Pydantic review-analysis model. Field-level
// .describe() calls double as the format instructions fed to StructuredOutputParser
// in the second pipeline implementation (see reviewAnalysisPipeline.ts).
export const ReviewAnalysisSchema = z.object({
  sentiment: z
    .enum(["positive", "negative", "neutral", "mixed"])
    .describe("Overall sentiment of the review"),
  key_issues: z
    .array(z.string())
    .describe("Specific problems, complaints, or standout concerns raised in the review; empty array if none"),
  summary: z.string().describe("A one to two sentence summary of what the reviewer said"),
});

export type ReviewAnalysis = z.infer<typeof ReviewAnalysisSchema>;
