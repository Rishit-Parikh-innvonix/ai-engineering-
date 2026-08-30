/**
 * A minimal LangChain-style Document Loader for PDFs.
 *
 * LangChain's real PDFLoader lives in @langchain/community, which we can't
 * install cleanly here - its peer @browserbasehq/stagehand pins dotenv@^16,
 * conflicting with this project's dotenv@^17 (same conflict documented for
 * vectorSearchDemo.ts's Chroma usage). Rather than force it with
 * --legacy-peer-deps, this reimplements the same *interface* LangChain
 * loaders expose - extend BaseDocumentLoader, implement load(), return
 * Document[] - using pdf-parse directly, which is already a real dependency.
 *
 * The payoff over the old approach (one flattened string for the whole PDF)
 * is real, not just structural: pdf-parse's getText() already extracts text
 * per page, so each Document below carries its own page number in metadata.
 * That page number survives into every chunk made from it, so retrieved
 * chunks can now be traced back to an actual page in the source PDF.
 */

import fs from "node:fs";
import { PDFParse } from "pdf-parse";
import { BaseDocumentLoader } from "@langchain/core/document_loaders/base";
import { Document } from "@langchain/core/documents";

export class SimplePdfLoader extends BaseDocumentLoader {
  constructor(private readonly filePath: string) {
    super();
  }

  async load(): Promise<Document[]> {
    const buffer = fs.readFileSync(this.filePath);
    const parser = new PDFParse({ data: buffer });
    const { pages, total } = await parser.getText();
    await parser.destroy();

    return pages.map(
      (page) =>
        new Document({
          pageContent: page.text,
          metadata: {
            source: this.filePath,
            pageNumber: page.num,
            totalPages: total,
          },
        })
    );
  }
}
