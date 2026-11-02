"""`dentabridge-api serve`: the HTTP API on uvicorn, configured by DENTABRIDGE_* (see Settings).

Demo mode replays the precomputed answers for the bundled samples on a frozen clock, so it needs
no model and no LiveKit, and every visitor sees the same thing. Live mode runs the vision model
DENTABRIDGE_VISION_BACKEND names, limited to what the release gate's config lets it claim, on the
real clock. A file either mode needs and cannot find stops the server at startup.
"""

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime

import uvicorn
from fastapi import FastAPI

from dentabridge_api.app import create_app
from dentabridge_api.clock import FrozenClock, SystemClock
from dentabridge_api.demo.fixture_vision import FixtureVisionBackend
from dentabridge_api.settings import Settings
from dentabridge_ml.serving import load_vision_backend
from dentabridge_ml.serving.parse import ReleaseConfig
from dentabridge_pricing.airports import load_airports
from dentabridge_pricing.fares import OfflineFareEstimator, load_fare_params

DEMO_NOW = datetime(2026, 11, 2, 9, tzinfo=UTC)


def build_app(settings: Settings) -> FastAPI:
    fares = OfflineFareEstimator(load_fare_params(), load_airports())
    if settings.mode == "demo":
        vision = FixtureVisionBackend(settings.samples_dir)
        return create_app(settings, FrozenClock(DEMO_NOW), vision, fares)
    release = ReleaseConfig.read(settings.release_config)
    live = load_vision_backend(settings.vision_backend, release, settings.image_max_px)
    return create_app(settings, SystemClock(), live, fares)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dentabridge-api", description=__doc__)
    steps = parser.add_subparsers(dest="step", required=True)
    serve = steps.add_parser("serve", help="run the API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    uvicorn.run(build_app(Settings()), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
