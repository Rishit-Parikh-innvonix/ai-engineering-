import fs from "node:fs/promises";
import path from "node:path";

import { REPORTS_DIR } from "./config.js";
import { slugify } from "./logic.js";

// A timestamp in the name means running the same topic twice never silently overwrites the first brief.
export async function saveReportToDisk(topic: string, markdown: string): Promise<string> {
  const stamp = new Date().toISOString().slice(0, 16).replace(/[-:T]/g, "");
  await fs.mkdir(REPORTS_DIR, { recursive: true });
  const filePath = path.join(REPORTS_DIR, `${slugify(topic)}-${stamp}.md`);
  await fs.writeFile(filePath, markdown, "utf-8");
  return filePath;
}
