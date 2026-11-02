"""The FastAPI application: wiring only. Behaviour lives in routes/ and the service modules."""

from datetime import timedelta

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dentabridge_api.bodylimit import BodyLimit
from dentabridge_api.booking.catalogue import SlotCatalogue
from dentabridge_api.clock import Clock
from dentabridge_api.context import RequestContext
from dentabridge_api.db.session import init_schema, make_engine, session_factory
from dentabridge_api.errors import install_error_handlers
from dentabridge_api.idempotency import IdempotencyStore
from dentabridge_api.ratelimit import DEFAULT_BUDGETS, RateLimiter
from dentabridge_api.routes import analyses, bookings, catalogue, consents, meta, quotes, samples
from dentabridge_api.samples import SampleCatalogue
from dentabridge_api.seed import ClinicSeed, load_clinics
from dentabridge_api.settings import Settings
from dentabridge_api.uploads import FORM_OVERHEAD, MAX_UPLOAD_BYTES
from dentabridge_core import CONTRACT_VERSION, FareEstimator, VisionBackend
from dentabridge_core.policy import DEFAULT_TABLE_PATH, load_table
from dentabridge_pricing.airports import Airport, load_airports
from dentabridge_pricing.fx import FxSnapshot, load_fx
from dentabridge_pricing.hotels import HotelTable, load_hotels

EXPOSED_HEADERS = [
    "X-Request-Id",
    "Retry-After",
    "RateLimit-Limit",
    "RateLimit-Remaining",
    "RateLimit-Reset",
    "Idempotent-Replayed",
]


def create_app(
    settings: Settings,
    clock: Clock,
    vision: VisionBackend,
    fares: FareEstimator,
    *,
    clinics: dict[str, ClinicSeed] | None = None,
    airports: dict[str, Airport] | None = None,
    fx: FxSnapshot | None = None,
    hotels: HotelTable | None = None,
) -> FastAPI:
    app = FastAPI(title="DentaBridge API", version=CONTRACT_VERSION)
    engine = make_engine(settings.db_url)
    init_schema(engine)
    app.state.settings = settings
    app.state.clock = clock
    app.state.vision = vision
    app.state.fares = fares
    app.state.sessions = session_factory(engine)
    app.state.limiter = RateLimiter(DEFAULT_BUDGETS, clock)
    app.state.idempotency = IdempotencyStore(app.state.sessions, clock, timedelta(hours=24))
    app.state.urgency_table = load_table(DEFAULT_TABLE_PATH)
    app.state.samples = SampleCatalogue(settings.samples_dir)
    app.state.clinics = clinics if clinics is not None else load_clinics()
    app.state.catalogue = SlotCatalogue(app.state.clinics)
    app.state.airports = airports if airports is not None else load_airports()
    app.state.fx = fx if fx is not None else load_fx()
    app.state.hotels = hotels if hotels is not None else load_hotels()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Idempotency-Key", "Accept-Language"],
        expose_headers=EXPOSED_HEADERS,
        max_age=600,
    )

    app.add_middleware(BodyLimit, limits={"/v1/analyses": MAX_UPLOAD_BYTES + FORM_OVERHEAD})
    app.add_middleware(RequestContext)  # last added = outermost: every response gets an id
    install_error_handlers(app)
    for router in (
        meta.router,
        consents.router,
        samples.router,
        analyses.reading,
        analyses.creating,
        catalogue.router,
        quotes.writing,
        quotes.reading,
        bookings.writing,
        bookings.idempotent,
        bookings.reading,
    ):
        app.include_router(router, prefix="/v1")
    return app
