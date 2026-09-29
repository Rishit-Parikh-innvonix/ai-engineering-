"""The weather tool: live data that Wikipedia, arXiv and Hacker News cannot provide.

Uses Open-Meteo (free, no API key): one call turns a city name into coordinates, a second returns
the current conditions. Plain code builds the text of the reading, so every number in it comes
straight from the API - the AI only ever phrases around it and never supplies a value."""

from dataclasses import dataclass
from typing import Awaitable, Callable
from urllib.parse import urlencode

from pydantic import BaseModel

from brief.sources.http import http_get_json
from brief.sources.types import SourceDocument


class GeocodedPlace(BaseModel):
    name: str
    latitude: float
    longitude: float
    country: str | None = None
    country_code: str | None = None
    admin1: str | None = None
    timezone: str | None = None


class _Geocoding(BaseModel):
    results: list[GeocodedPlace] | None = None


class _Current(BaseModel):
    time: str
    temperature_2m: float
    apparent_temperature: float
    relative_humidity_2m: float
    precipitation: float
    wind_speed_10m: float
    weather_code: int


class _Units(BaseModel):
    temperature_2m: str
    relative_humidity_2m: str
    precipitation: str
    wind_speed_10m: str


class CurrentWeather(BaseModel):
    timezone: str | None = None
    current: _Current
    current_units: _Units


@dataclass(frozen=True)
class WeatherTool:
    """`lookup` raises if the city cannot be found or the service is down; the graph then falls back to research."""

    lookup: Callable[[str], Awaitable[SourceDocument]]


# WMO weather interpretation codes, as documented by Open-Meteo.
WMO_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    56: "light freezing drizzle", 57: "dense freezing drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "slight snowfall", 73: "moderate snowfall", 75: "heavy snowfall", 77: "snow grains",
    80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    85: "slight snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}  # fmt: skip


def describe_weather_code(code: int) -> str:
    return WMO_CODES.get(code, f"unrecognised condition (WMO code {code})")


def pick_place(results: list[GeocodedPlace], qualifier: str) -> GeocodedPlace | None:
    """"Ahmedabad" -> the top result. "Ahmedabad, India" or "Springfield, Illinois" -> the first
    match whose country or region fits the qualifier, else the top match. Open-Meteo already orders
    results by relevance, and city names repeat across countries (there is an Ahmedabad in Pakistan too)."""
    wanted = qualifier.strip().lower()
    if wanted:
        for place in results:
            if wanted in [(f or "").lower() for f in (place.country, place.country_code, place.admin1)]:
                return place
    return results[0] if results else None


def place_label(place: GeocodedPlace) -> str:
    parts = [place.name, place.admin1, place.country]
    return ", ".join(part for i, part in enumerate(parts) if part and part not in parts[:i])


def _num(value: float) -> str:
    return f"{value:g}"  # 34.0 -> "34", 38.7 -> "38.7", the same text the API's JSON shows


def build_weather_document(place: GeocodedPlace, weather: CurrentWeather, url: str) -> SourceDocument:
    """The fact block the AI is allowed to write from. Every value is copied from the API response."""
    current, units = weather.current, weather.current_units
    where = place_label(place)
    zone = f" ({weather.timezone})" if weather.timezone else ""
    return SourceDocument(
        kind="weather",
        title=f"Current weather in {where}",
        url=url,
        text="\n".join(
            [
                f"Location: {where}.",
                f"Observed: {current.time.replace('T', ' ')} local time{zone}.",
                f"Conditions: {describe_weather_code(current.weather_code)}.",
                f"Temperature: {_num(current.temperature_2m)} {units.temperature_2m} "
                f"(feels like {_num(current.apparent_temperature)} {units.temperature_2m}).",
                f"Relative humidity: {_num(current.relative_humidity_2m)}{units.relative_humidity_2m}.",
                f"Wind speed: {_num(current.wind_speed_10m)} {units.wind_speed_10m}.",
                f"Precipitation: {_num(current.precipitation)} {units.precipitation}.",
            ]
        ),
    )


async def _lookup(city: str) -> SourceDocument:
    name, _, qualifier = city.partition(",")
    geocode_url = "https://geocoding-api.open-meteo.com/v1/search?" + urlencode(
        {"count": 10, "language": "en", "format": "json", "name": name.strip()}
    )
    geocoded = await http_get_json(geocode_url, _Geocoding)
    place = pick_place(geocoded.results or [], qualifier)
    if place is None:
        raise RuntimeError(f'No place called "{city}" was found.')

    forecast_url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={place.latitude}&longitude={place.longitude}&timezone=auto"
        "&current=temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,wind_speed_10m,weather_code"
    )
    weather = await http_get_json(forecast_url, CurrentWeather)
    return build_weather_document(place, weather, forecast_url)


open_meteo_weather = WeatherTool(lookup=_lookup)
