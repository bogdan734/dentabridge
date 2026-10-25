"""Rebuild the committed snapshots in data/ (network; run by a maintainer, not by CI).

- OurAirports (public domain): airports that have scheduled passenger service and an IATA code.
- ECB reference rates through api.frankfurter.dev, quoted against USD.
"""

import csv
import gzip
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

DATA = Path(__file__).resolve().parents[1] / "data"
AIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
FX_URL = "https://api.frankfurter.dev/v1/latest?base=USD"
KINDS = {"large_airport", "medium_airport", "small_airport"}


def airports() -> str:
    text = httpx.get(AIRPORTS_URL, timeout=60.0, follow_redirects=True).raise_for_status().text
    rows = [
        {
            "iata": r["iata_code"],
            "name": r["name"],
            "municipality": r["municipality"],
            "iso_country": r["iso_country"],
            "lat": r["latitude_deg"],
            "lon": r["longitude_deg"],
        }
        for r in csv.DictReader(io.StringIO(text))
        if r["scheduled_service"] == "yes" and r["type"] in KINDS and len(r["iata_code"]) == 3
    ]
    rows.sort(key=lambda r: r["iata"])
    stamp = datetime.now(UTC).date().isoformat()
    out = DATA / f"airports_{stamp}.csv.gz"
    with gzip.open(out, "wt", encoding="utf-8", newline="", compresslevel=9) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return f"{out.name}: {len(rows)} airports"


def fx() -> str:
    body = httpx.get(FX_URL, timeout=30.0).raise_for_status().json()
    snapshot = {"as_of": body["date"], "base": body["base"], "rates": body["rates"]}
    out = DATA / f"fx_{body['date']}.json"
    out.write_text(json.dumps(snapshot, indent=1, sort_keys=True) + "\n")
    return f"{out.name}: {len(body['rates'])} currencies"


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    print(airports())
    print(fx())
