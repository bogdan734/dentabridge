"""Airports with scheduled passenger service, from the OurAirports snapshot in data/."""

import csv
import gzip
import math
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class Airport:
    iata: str
    name: str
    municipality: str
    iso_country: str
    lat: float
    lon: float


def latest(pattern: str, directory: Path = DATA_DIR) -> Path:
    found = sorted(directory.glob(pattern))
    if not found:
        raise FileNotFoundError(f"no {pattern} in {directory}: run scripts/build_snapshots.py")
    return found[-1]  # the file names carry an ISO date, so the last one is the newest


def load_airports(path: Path | None = None) -> dict[str, Airport]:
    with gzip.open(path or latest("airports_*.csv.gz"), "rt", encoding="utf-8", newline="") as f:
        return {
            row["iata"]: Airport(
                iata=row["iata"],
                name=row["name"],
                municipality=row["municipality"],
                iso_country=row["iso_country"],
                lat=float(row["lat"]),
                lon=float(row["lon"]),
            )
            for row in csv.DictReader(f)
        }


def search_cities(q: str, limit: int, airports: dict[str, Airport]) -> list[Airport]:
    """Airports whose city, name or code starts with the query, city matches first."""
    needle = q.strip().casefold()
    if not needle:
        return []
    by_city = [a for a in airports.values() if a.municipality.casefold().startswith(needle)]
    seen = {a.iata for a in by_city}
    others = [
        a
        for a in airports.values()
        if a.iata not in seen
        and (a.iata.casefold() == needle or a.name.casefold().startswith(needle))
    ]
    ordered = sorted(by_city, key=lambda a: (a.municipality, a.iata)) + sorted(
        others, key=lambda a: (a.name, a.iata)
    )
    return ordered[:limit]


def haversine_km(a: Airport, b: Airport) -> float:
    lat1, lat2 = math.radians(a.lat), math.radians(b.lat)
    dlat, dlon = lat2 - lat1, math.radians(b.lon - a.lon)
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))
