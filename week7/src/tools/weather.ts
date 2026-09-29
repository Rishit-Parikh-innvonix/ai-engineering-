/**
 * The weather tool: live data that Wikipedia, arXiv and Hacker News cannot provide.
 *
 * Uses Open-Meteo (free, no API key): one call turns a city name into coordinates, a second
 * returns the current conditions. Plain code builds the text of the reading, so every number in
 * it comes straight from the API - the AI only ever phrases around it and never supplies a value.
 */

import { z } from "zod";

import { httpGetJson } from "../sources/http.js";
import type { SourceDocument } from "../sources/types.js";

export interface WeatherTool {
  /** Throws if the city cannot be found or the service is down; the graph then falls back to research. */
  lookup(city: string): Promise<SourceDocument>;
}

const geocodingSchema = z.object({
  results: z
    .array(
      z.object({
        name: z.string(),
        latitude: z.number(),
        longitude: z.number(),
        country: z.string().optional(),
        country_code: z.string().optional(),
        admin1: z.string().optional(),
        timezone: z.string().optional(),
      })
    )
    .optional(),
});
export type GeocodedPlace = NonNullable<z.infer<typeof geocodingSchema>["results"]>[number];

const forecastSchema = z.object({
  timezone: z.string().optional(),
  current: z.object({
    time: z.string(),
    temperature_2m: z.number(),
    apparent_temperature: z.number(),
    relative_humidity_2m: z.number(),
    precipitation: z.number(),
    wind_speed_10m: z.number(),
    weather_code: z.number(),
  }),
  current_units: z.object({
    temperature_2m: z.string(),
    relative_humidity_2m: z.string(),
    precipitation: z.string(),
    wind_speed_10m: z.string(),
  }),
});
export type CurrentWeather = z.infer<typeof forecastSchema>;

// WMO weather interpretation codes, as documented by Open-Meteo.
const WMO_CODES: Record<number, string> = {
  0: "clear sky",
  1: "mainly clear",
  2: "partly cloudy",
  3: "overcast",
  45: "fog",
  48: "depositing rime fog",
  51: "light drizzle",
  53: "moderate drizzle",
  55: "dense drizzle",
  56: "light freezing drizzle",
  57: "dense freezing drizzle",
  61: "slight rain",
  63: "moderate rain",
  65: "heavy rain",
  66: "light freezing rain",
  67: "heavy freezing rain",
  71: "slight snowfall",
  73: "moderate snowfall",
  75: "heavy snowfall",
  77: "snow grains",
  80: "slight rain showers",
  81: "moderate rain showers",
  82: "violent rain showers",
  85: "slight snow showers",
  86: "heavy snow showers",
  95: "thunderstorm",
  96: "thunderstorm with slight hail",
  99: "thunderstorm with heavy hail",
};
export const describeWeatherCode = (code: number): string => WMO_CODES[code] ?? `unrecognised condition (WMO code ${code})`;

/**
 * "Ahmedabad" -> the top match. "Ahmedabad, India" or "Springfield, Illinois" -> the first match
 * whose country or region fits the qualifier, else the top match. Open-Meteo already orders
 * results by relevance, and city names repeat across countries (there is an Ahmedabad in Pakistan too).
 */
export function pickPlace(results: GeocodedPlace[], qualifier: string): GeocodedPlace | undefined {
  const wanted = qualifier.trim().toLowerCase();
  if (wanted) {
    const fits = results.find((place) =>
      [place.country, place.country_code, place.admin1].some((field) => field?.toLowerCase() === wanted)
    );
    if (fits) return fits;
  }
  return results[0];
}

export const placeLabel = (place: GeocodedPlace): string =>
  [place.name, place.admin1, place.country].filter((part, i, all) => part && all.indexOf(part) === i).join(", ");

/** The fact block the AI is allowed to write from. Every value is copied from the API response. */
export function buildWeatherDocument(place: GeocodedPlace, weather: CurrentWeather, url: string): SourceDocument {
  const { current, current_units: units } = weather;
  const where = placeLabel(place);
  const zone = weather.timezone ? ` (${weather.timezone})` : "";
  return {
    kind: "weather",
    title: `Current weather in ${where}`,
    url,
    text: [
      `Location: ${where}.`,
      `Observed: ${current.time.replace("T", " ")} local time${zone}.`,
      `Conditions: ${describeWeatherCode(current.weather_code)}.`,
      `Temperature: ${current.temperature_2m} ${units.temperature_2m} (feels like ${current.apparent_temperature} ${units.temperature_2m}).`,
      `Relative humidity: ${current.relative_humidity_2m}${units.relative_humidity_2m}.`,
      `Wind speed: ${current.wind_speed_10m} ${units.wind_speed_10m}.`,
      `Precipitation: ${current.precipitation} ${units.precipitation}.`,
    ].join("\n"),
  };
}

export const openMeteoWeather: WeatherTool = {
  async lookup(city) {
    const [name, ...qualifier] = city.split(",");
    const geocodeUrl =
      "https://geocoding-api.open-meteo.com/v1/search?count=10&language=en&format=json&name=" +
      encodeURIComponent(name.trim());
    const geocoded = await httpGetJson(geocodeUrl, geocodingSchema);
    const place = pickPlace(geocoded.results ?? [], qualifier.join(","));
    if (!place) throw new Error(`No place called "${city}" was found.`);

    const forecastUrl =
      "https://api.open-meteo.com/v1/forecast" +
      `?latitude=${place.latitude}&longitude=${place.longitude}&timezone=auto` +
      "&current=temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,wind_speed_10m,weather_code";
    const weather = await httpGetJson(forecastUrl, forecastSchema);
    return buildWeatherDocument(place, weather, forecastUrl);
  },
};
