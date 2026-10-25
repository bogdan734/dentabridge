"""The FastAPI application: wiring only. Behaviour lives in routes/ and the service modules."""

import logging
import time
from collections.abc import Awaitable, Callable
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from dentabridge_api.clock import Clock
from dentabridge_api.db.session import init_schema, make_engine, session_factory
from dentabridge_api.errors import install_error_handlers
from dentabridge_api.routes import meta
from dentabridge_api.settings import Settings
from dentabridge_core import CONTRACT_VERSION, FareEstimator, VisionBackend

EXPOSED_HEADERS = [
    "X-Request-Id",
    "Retry-After",
    "RateLimit-Limit",
    "RateLimit-Remaining",
    "RateLimit-Reset",
    "Idempotent-Replayed",
]
log = logging.getLogger("dentabridge_api.access")


def create_app(
    settings: Settings, clock: Clock, vision: VisionBackend, fares: FareEstimator
) -> FastAPI:
    app = FastAPI(title="DentaBridge API", version=CONTRACT_VERSION)
    engine = make_engine(settings.db_url)
    init_schema(engine)
    app.state.settings = settings
    app.state.clock = clock
    app.state.vision = vision
    app.state.fares = fares
    app.state.sessions = session_factory(engine)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Idempotency-Key", "Accept-Language"],
        expose_headers=EXPOSED_HEADERS,
        max_age=600,
    )

    @app.middleware("http")
    async def request_id_and_access_log(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request.state.request_id = request.headers.get("X-Request-Id") or uuid4().hex
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-Id"] = request.state.request_id
        # request id, route, status and duration: no body, no query string, no client address
        log.info(
            "%s %s %s %d %.0fms",
            request.state.request_id,
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
        )
        return response

    install_error_handlers(app)
    app.include_router(meta.router, prefix="/v1")
    return app
