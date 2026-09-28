import type { z } from "zod";

// Wikipedia's API policy requires a descriptive User-Agent (a bare fetch gets throttled with 429 -
// confirmed in week5). Deliberately contains no personal contact details: the project name is
// enough for the policy, and nothing about the person running this belongs in a request header.
const USER_AGENT = "week7-research-brief/1.0 (educational project)";
const REQUEST_TIMEOUT_MS = 15_000;

async function httpGetText(url: string): Promise<string> {
  const response = await fetch(url, {
    headers: { "User-Agent": USER_AGENT },
    signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} from ${new URL(url).hostname}`);
  }
  return response.text();
}

// External APIs are the one place data enters this program untrusted, so every response is
// validated against a schema instead of being cast to a type and hoped for.
export async function httpGetJson<T>(url: string, schema: z.ZodType<T>): Promise<T> {
  const parsed = schema.safeParse(JSON.parse(await httpGetText(url)));
  if (!parsed.success) {
    throw new Error(`Unexpected response format from ${new URL(url).hostname}`);
  }
  return parsed.data;
}

export async function httpGetXml<T>(url: string, parse: (xml: string) => T): Promise<T> {
  return parse(await httpGetText(url));
}
