/**
 * Wikipedia's API enforces a User-Agent policy (https://meta.wikimedia.org/wiki/User-Agent_policy)
 * and returns intermittent 429s to requests that don't identify themselves - confirmed live:
 * a bare fetch() to en.wikipedia.org got 429, the identical request with a descriptive
 * User-Agent got 200.
 *
 * @langchain/community's WikipediaQueryRun has no constructor option for custom headers or a
 * custom fetch implementation, so there is no way to fix this through its public API. This
 * installs the header at the transport level instead - but scoped to requests whose hostname
 * is actually en.wikipedia.org, so every other outbound call in the app (OpenRouter, Open-Meteo)
 * is untouched and behaves exactly as if this module didn't exist.
 */

const WIKIPEDIA_HOSTNAME = "en.wikipedia.org";
const USER_AGENT = "week5-agents-demo/1.0 (educational project; contact: parikhrishit@gmail.com)";

let installed = false;

function resolveUrl(input: Parameters<typeof fetch>[0]): URL {
  if (typeof input === "string") return new URL(input);
  if (input instanceof URL) return input;
  return new URL(input.url);
}

export function ensureWikipediaUserAgent(): void {
  if (installed) return;
  installed = true;

  const originalFetch = globalThis.fetch;

  globalThis.fetch = (async (input: Parameters<typeof fetch>[0], init: Parameters<typeof fetch>[1] = {}) => {
    if (resolveUrl(input).hostname !== WIKIPEDIA_HOSTNAME) {
      return originalFetch(input, init);
    }

    const headers = new Headers(init.headers);
    headers.set("User-Agent", USER_AGENT);
    return originalFetch(input, { ...init, headers });
  }) as typeof fetch;
}
