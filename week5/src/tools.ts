/**
 * Single source of truth for every tool used across the 3 exercises, so each exercise file
 * just imports the tool(s) it needs instead of redefining them.
 *
 * @langchain/community is deprecated upstream as a whole package (npm install prints this,
 * and https://github.com/langchain-ai/langchainjs-community/issues/61 confirms it) - checked
 * directly and there is currently no dedicated, non-deprecated replacement package for a
 * Wikipedia tool (LangChain's own guidance for that case: keep using it, or implement directly).
 * WikipediaQueryRun's class itself carries no individual @deprecated tag, only the umbrella
 * package does, and it still works correctly - kept deliberately for that reason. Everything
 * else that package offered here (Calculator) IS trivially replaceable, so it's replaced below
 * instead of pulled in from the deprecated package for no reason.
 *
 * Installed with --legacy-peer-deps because @langchain/community lists @browserbasehq/stagehand
 * (used only by its unrelated WebBrowser tool, which we never import) as a peer wanting an
 * older dotenv than this project uses - a peer-resolution conflict only.
 */

import { z } from "zod";
import { tool } from "@langchain/core/tools";
import { WikipediaQueryRun } from "@langchain/community/tools/wikipedia_query_run";
import Mexp from "math-expression-evaluator";

import { ensureWikipediaUserAgent } from "./wikipediaFetch.js";

// See wikipediaFetch.ts: WikipediaQueryRun has no constructor option for custom headers,
// and Wikipedia's API rejects/throttles requests with none. Scoped to en.wikipedia.org only -
// every other outbound call (OpenRouter, Open-Meteo) is unaffected.
ensureWikipediaUserAgent();

// Exercise 1: a real pre-built LangChain tool - no custom code, just configuration.
export const wikipedia = new WikipediaQueryRun({ topKResults: 1, maxDocContentLength: 500 });

interface GeocodingResult {
  results?: { name: string; country: string; latitude: number; longitude: number }[];
}
interface ForecastResult {
  current: { temperature_2m: number; wind_speed_10m: number; weather_code: number };
}

// Exercise 2: a custom tool, from scratch, calling a real external API. Open-Meteo needs no
// API key - two plain fetches: city name -> lat/lon, then lat/lon -> current weather.
export const getCurrentWeather = tool(
  async ({ city }: { city: string }) => {
    const geoResponse = await fetch(`https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(city)}&count=1`);
    if (!geoResponse.ok) return `Weather lookup failed: geocoding service returned HTTP ${geoResponse.status}.`;
    const geo = (await geoResponse.json()) as GeocodingResult;
    const place = geo.results?.[0];
    if (!place) return `Could not find a location named "${city}".`;

    const weatherResponse = await fetch(
      `https://api.open-meteo.com/v1/forecast?latitude=${place.latitude}&longitude=${place.longitude}&current=temperature_2m,wind_speed_10m,weather_code`
    );
    if (!weatherResponse.ok) return `Weather lookup failed: forecast service returned HTTP ${weatherResponse.status}.`;
    const weather = (await weatherResponse.json()) as ForecastResult;
    const { temperature_2m, wind_speed_10m, weather_code } = weather.current;

    return `Current weather in ${place.name}, ${place.country}: ${temperature_2m}°C, wind ${wind_speed_10m} km/h (WMO weather code ${weather_code}).`;
  },
  {
    name: "get_current_weather",
    description: "Get the current temperature and wind speed for a named city, using live public weather data. Use this whenever a question needs today's actual weather.",
    schema: z.object({ city: z.string().describe("The city name, e.g. 'Mumbai' or 'London'.") }),
  }
);

// Third tool for exercise 3's ReAct agent (it asks for 2-4 tools; this gives us 3) - built from
// scratch the same way as get_current_weather, instead of pulling in the deprecated
// @langchain/community Calculator (which itself was just a thin wrapper around this same
// math-expression-evaluator package - so this uses the real logic directly, no wrapper).
const expressionParser = new Mexp();

export const calculator = tool(
  async ({ expression }: { expression: string }) => {
    try {
      return expressionParser.eval(expression).toString();
    } catch {
      return `Could not evaluate "${expression}" as a math expression.`;
    }
  },
  {
    name: "calculator",
    description: "Evaluate a mathematical expression, e.g. '47 * 89' or '(3 + 4) / 2'.",
    schema: z.object({ expression: z.string().describe("A valid arithmetic expression.") }),
  }
);
