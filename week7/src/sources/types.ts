import { z } from "zod";

export const SOURCE_KINDS = ["wikipedia", "arxiv", "hackernews"] as const;
export type SourceKind = (typeof SOURCE_KINDS)[number];

export const sourceDocumentSchema = z.object({
  kind: z.enum(SOURCE_KINDS),
  title: z.string(),
  url: z.string(),
  text: z.string(),
});
export type SourceDocument = z.infer<typeof sourceDocumentSchema>;

// One place a search can be run. fetch() throws on network/HTTP/format problems and returns []
// when the source simply has nothing - the retrieval agent treats those two cases differently.
export interface SourceFetcher {
  kind: SourceKind;
  label: string;
  fetch(query: string): Promise<SourceDocument[]>;
}
