import { z } from "zod";

// "weather" is not searched: it comes from a live-data tool (src/tools/weather.ts), not a search box.
export const SOURCE_KINDS = ["wikipedia", "arxiv", "hackernews", "weather"] as const;
export type SourceKind = (typeof SOURCE_KINDS)[number];
export type SearchKind = Exclude<SourceKind, "weather">;

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
  kind: SearchKind;
  label: string;
  fetch(query: string): Promise<SourceDocument[]>;
}
