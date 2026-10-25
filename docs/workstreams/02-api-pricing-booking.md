# WS02 API, pricing and booking (+ the contracts the site consumes)
Branch ws/02-api. Starts after tag `m0`.

## Owned paths
- packages/api/ (src/dentabridge_api/{app,settings,errors,clock,ratelimit,idempotency,uploads,analysis_service,export,call_tokens}.py, db/, routes/, booking/, demo/, alembic/, tests/)
- packages/pricing/ (src/dentabridge_pricing/{airports,fx,fares,calibrate,evaluate,quote,hotels}.py, data/, scripts/build_snapshots.py, LICENCES.yaml, tests/)
- seed/ (clinics.yaml, hotel_tiers.yaml, schema.json), contracts/ (generated only), docs/results/fares.{json,md}, mk/api.mk
- Gitignored: var/ (SQLite), data/raw/pricing/ (DB1B, Kaggle flight prices)

## Tasks (PLAN.md ids, in order)
1. A1 skeleton [L] → A2 rate limit + idempotency [S] → A6 contracts export [S]
2. P1 airports/cities/FX [S], in parallel with A2
3. A3 consents + analyses + uploads + demo vision [L] (needs F4)
4. P2 fare estimator + calibration [L] → P3 quote [S]
5. A4 clinics/slots/holds/bookings, with the share-with-clinic snapshot [L] → A5 calls + dentist context + replays [L]
6. At M4: `make precompute-samples` with the shipped backend (ML5 + ML9)
- Every endpoint task also extends `record_fixtures` and regenerates contracts/.

## Contracts consumed
- core: every model and the ports, `policy.evaluate`, `policy.render_patient_text`, `policy.guard_note`, `policy.texts`, `policy.lint`, VERIFIED_PAIRS, `testing` fakes
- WS01: `dentabridge_ml.serving.load_vision_backend` (from M2; the fixture backend until then)
- WS03: packages/voice/replays/<id>/manifest.json (read via settings.replays_dir; ReplayManifest schema from core)

## Contracts produced
- HTTP API /v1 exactly as in the PLAN.md "API surface" table (codes, idempotency, rate limits, CORS)
- contracts/openapi.json, contracts/ts (openapi-typescript 7.x types + openapi-fetch client), contracts/fixtures/{happy_path,errors}/
- LiveKit room metadata = CallRoomMetadata JSON; tokens with identity "<role>-<session_id>"; GET /v1/calls/{id}/context (Bearer call token, role dentist) returns CallContext
- `OfflineFareEstimator` (implements FareEstimator) and `build_quote`
- docs/results/fares.json (MdAPE, interval coverage, bias by band, dataset versions)

## Gate
- `scripts/check`
- `uv run pytest packages/api packages/pricing -q`
- `make contracts && git diff --exit-code contracts/`
- Maintainer only (network + raw data): `make eval-fares`

## Definition of done
- Demo mode (`DENTABRIDGE_MODE=demo`) serves every endpoint deterministically. The fixtures regenerate byte-identical.
- Privacy: EXIF is stripped; nothing is written to disk; no image bytes or raw IPs appear in logs or the DB; TTL 410 works. A shared-results snapshot lives only on the booking (purged 24 h after the appointment or on cancel); the context endpoint never returns image bytes and rejects patient tokens.
- The booking state machine never double-books (hypothesis), including across a DST transition.
- Quotes:
  - every line has a chip and a basis key; totals are ESTIMATE ranges;
  - fare_params_v1.json records datasets and metrics (or v0 ships with fare_mdape null);
  - the international interval is q10-q90.
- Every patient-facing string in a response comes from core.policy and passes the runtime lint.
- Seed clinics are fictional, in real cities on 3 continents, with no country or currency literals in code.

## Do NOT touch
- packages/core (file a change request below), packages/ml, packages/voice, deploy/, space/, tools/, README.md, web/
- Never hand-edit contracts/: regenerate it.
- No live travel adapters, payment providers or API keys in v1.

## Notes for the site (WS05), keep true
- CORS allow-list only, no credentials. Safety texts come from GET /v1/meta. Call tokens need `ai_disclosure_acknowledged: true`. Sharing results with the clinic needs its own consent (purpose share_with_clinic); the dentist panel reads GET /v1/calls/{id}/context. RUB is not offered (no ECB rate).

## Contract change requests
- (none yet)
