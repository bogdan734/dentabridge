# DentaBridge master plan (contracts only)
Draft v1.1, 2026-10-25 (owner answers applied: dentist context, chat, permissive default translator, RUB). Author: Bohdan Havryliuk. Scope: everything up to, NOT including, the website UI (web/ is reserved for WS05).
Task ids: F foundation, ML, A api, P pricing, V voice, D docs/packaging. Lane S = up to 3 files with a clear spec; L = subsystem, algorithm, cross-module or risky.
Every task: failing test first, one commit `<id>: <summary>` with the repo-local identity, `scripts/check` green.

## Goal
A local-first open-source research prototype. Flow: patient photo → image observations + demo urgency hint (decision support, not diagnosis) → fictional clinic → quote (treatment + round-trip flight + hotel, all estimates) → mock booking → video call with live voice-to-voice EN<->RU translation.
Star deliverables: (1) a fine-tuned open VLM of at most 4B parameters with a script-generated honest evaluation; (2) the translated call. The site (EN+RU, built later by another model) consumes only `contracts/` (OpenAPI, TS client, demo fixtures) and the CORS rules below.

## Architecture
- (a) Shipped vision path: tuned Qwen3-VL-4B end to end (image → constrained JSON `VisionOutput`) behind `VisionBackend`. Why:
  - one model fits mixed label granularity (AKU boxes, AlphaDent masks, Mendeley image-level gingivitis);
  - the quality gate and abstention come in the same pass, and it is the star artefact;
  - patient text comes from templates, so a separate "VLM explainer" stage adds risk and no value.
- Design A (D-FINE detector via Transformers, Apache-2.0) is trained on the same splits as the main ablation. Zero-shot Ollama qwen3-vl:4b/8b are the baselines; AlphaDent's YOLOv8x is a local-only row. The pre-registered ship rule (ML9) may switch to the detector, and a per-finding release gate decides which findings ship as assessed.
- Urgency never comes from the model: a deterministic rule table over validated findings (core policy) stays "illustrative" until a dentist review record matches its sha256.
- uv workspace, Python 3.13: `core` (contracts, ports, policy, i18n) ← `ml`, `pricing`, `voice`; `api` ← core, pricing, ml[serving]. Nothing imports api or voice.
- Runtime: FastAPI + SQLite (SQLAlchemy 2 + Alembic); Ollama (baselines, MT, and the tuned model if ML1 passes); self-hosted LiveKit server + livekit-agents worker; Piper HTTP server in its own container (keeps GPL-3.0 code out of our process).
- Call, one direction per speaker: Silero VAD → STT (Parakeet-v3 via MLX on Mac, faster-whisper large-v3-turbo in Docker) → Translator (default: a permissive Qwen3 4B/8B instruct on Ollama with the glossary in the prompt; TranslateGemma-4B, Opus-MT and MADLAD are opt-in rows; the V4 bake-off fixes the default by measured chrF++ and glossary accuracy) → glossary + digit checks → Piper → translated track + caption text stream. Consecutive rhythm by default. Text chat reuses the same Translator and checks (topics chat.in.v1 → chat.v1).
- Demo mode (`DENTABRIDGE_MODE=demo`): frozen clock 2026-11-02T09:00:00Z, uuid5 ids. `FixtureVisionBackend` serves precomputed outputs of the shipped model on 12 licence-clean samples, so no ML or LiveKit is needed. The site and the Space run on it.

| Path | Owner | Note |
|---|---|---|
| pyproject.toml, uv.lock, Makefile, mk/core.mk, scripts/, .github/, CLAUDE.md, LICENSE, .gitignore, .python-version, docs/PLAN.md, docs/workstreams/, docs/clinical-review/ | WS00 lead | the lead re-locks uv.lock after each merge |
| packages/core/ | WS00 lead | changes only via the protocol below |
| packages/ml/, docs/model-cards/, docs/results/ml/, mk/ml.mk | WS01 ml | |
| packages/api/, packages/pricing/, seed/, contracts/, docs/results/fares.*, mk/api.mk | WS02 api | contracts/ is generated |
| packages/voice/, docs/results/voice.*, mk/voice.mk | WS03 call | |
| deploy/, space/, tools/, README.md, docs/licences.md, docs/demo/, mk/deploy.mk, .env.example | WS04 docs+packaging | |
| web/ | WS05 site (reserved) | README placeholder only |

## Global constraints
- Python 3.13 everywhere, including Kaggle, Colab and AMD (`uv python install 3.13`), with one uv.lock. Torch comes from the conflicting uv extras `cpu | cu128 | rocm` (prior art: uv's "Using uv with PyTorch" guide).
- MIT code. No AGPL/GPL code imported in-process (Ultralytics never in packages/; Piper over HTTP only). NC models (XTTS-v2, F5-TTS-RU, NLLB) and DENTEX are opt-in downloads with licence rows.
- Datasets are never committed; data/, private/ and var/ are gitignored. Kaggle "Oral Diseases" (licence Unknown) lives only under private/ and never reaches manifests, results, weights, the README or the Space (a test enforces this).
- Patient-facing strings come only from the core i18n templates (EN, RU), linted at test time and at runtime; a violation raises, never passes silently. The VLM note is EN only, optional and linted.
- No paid APIs and no keys on the default path. Network is used only by `make data | models` and by maintainer scripts marked `network`.
- Every README number is rendered from docs/results/*.json, which repo scripts produce. The test split is touched only via `make eval-test`, and each access is logged.
- `scripts/check` = ruff check, ruff format --check, mypy --strict, pytest without `slow|gpu|network`, `dentabridge-lint`. It stays under 2 min and quiet on success. Never weaken it.
- No literal country, city or currency in code: they live in seed/ and config/. Clinics are fictional (`is_fictional: Literal[True]`).
- The owner's private project names never appear in the repo; the banned list lives outside it (env `DENTABRIDGE_BANNED_NAMES_FILE`).
- Contract change protocol: a workstream writes the request in its WS file; the lead edits core, bumps `CONTRACT_VERSION`, refreshes the snapshots and merges; the others rebase.
- Branches `ws/NN-name`, merged by the lead. On a uv.lock conflict, take main's lock and run `uv lock`. The repo and the HF repos stay private until the owner says "publish".
- Schedule rule (owner, 2026-10-25): speed over calendar. The AMD credit expiry is unknown, so the ML stream starts first and its 50-step MI300X pilot runs as soon as ML4 and ML8 allow; ablations do not wait for M2.

## Rejected alternatives and why
- Detector-only shipped path: cannot learn gingivitis from image-level labels, has no quality gate, and the VLM is the star. Kept as the ablation and the ship-rule fallback.
- Detector + VLM explainer: text comes from linted templates, so the explainer only adds failure modes.
- MedGemma 1.5 4B: HAI-DEF terms (pass-through, termination, deletion). Qwen3.5-4B: Ollama vision disputed, no T4 bf16. Fallbacks only after ML1.
- Ultralytics YOLO: AGPL-3.0. RF-DETR: separate package, and the licence varies by size (PML 1.0 for Atto/Femto/Pico/XL/2XL); D-FINE in Transformers covers the need.
- Unsloth as a requirement: vendor-specific. TRL + PEFT run on CUDA and ROCm; Unsloth stays an optional accelerator.
- MetaDent: dataset licence unstated and images web-sourced, so deferred.
- LiveKit turn-detector plugin: models under the LiveKit Model License (not OSI). Silero VAD + silence endpointing instead.
- Open S2S (Seamless, Hibiki): NC licences and no EN<->RU.
- Live fare adapters: Amadeus Self-Service decommissioned, Kiwi self-serve closed, Duffel test prices fake. The port is defined; the offline estimator ships.
- Cal.diy for slots: an extra Node + Postgres service for one table. Postgres default: SQLite suffices locally (Postgres via `db_url`).
- Uploads stored on disk: in-memory processing is simpler and more private; the TTL applies to results.

## Verified facts (checked 2026-10-25; P = primary page, S = secondary)
- Qwen3-VL-4B-Instruct: HF tag apache-2.0, 4B, needs transformers >= 4.57 (P, HF card). Its Ollama vision import is UNVERIFIED → ML1.
- TranslateGemma-4B-it: Gemma Terms of Use (not Apache), gated, Russian not listed on the card (P). Owner rule 2026-10-25: the runtime stays open source, so it is opt-in only and the default translator is a permissive model.
- Parakeet-tdt-0.6b-v3: CC-BY-4.0, includes ru, released 2025-08-14, chunked streaming script (P). D-FINE: ustc-community checkpoints apache-2.0, `DFineForObjectDetection` in Transformers (S).
- RF-DETR: code + N/S/M/L detection + all Seg models Apache-2.0, other sizes PML 1.0 (P). piper1-gpl: GPL-3.0 with HTTP server mode (P). mlx-vlm: MIT, LoRA/QLoRA, OpenAI-compatible server (P).
- livekit-agents 1.8.9 (2026-10-24), Apache-2.0, Python >=3.10,<3.15 (P, PyPI); 2.0.0b1 is a pre-release, so pin 1.8.x. LiveKit server Apache-2.0, v1.13.2 2026-06-27 (S).
- openapi-typescript 7.13.0 MIT, openapi-fetch 0.17.0 MIT (S). On this machine: uv 0.11.19, Ollama 0.40.10, Python 3.13 (P).
- AMD Developer Cloud MI300X: $1.99/GPU-h (S, DigitalOcean doc last verified 2025-06-03), so $100 buys about 50 GPU-h. Re-check at sign-up.
- llama.cpp has loaded Qwen3-VL GGUF with a separate mmproj since 2025-10-30 (S, Unsloth docs). No Ollama recipe found for a tuned VLM → ML1.
- Still UNVERIFIED, each owned by a task: DINOv2 licence, AlphaDent 1,320 vs 1,455, the meaning of caries 1-6, Mendeley labels and caption language (ML2); Ollama logprobs (ML1); DB1B and Kaggle flight-price terms (P2); RUB missing from the ECB snapshot (memory, P1); Piper RU voice, faster-whisper, parakeet-mlx and FLEURS licences (V2, V4).

## Shared contracts (packages/core, frozen at M0, CONTRACT_VERSION 1.0.0)
Pydantic v2 with `frozen=True, extra="forbid"`; tz-aware UTC datetimes; money in minor units; BCP-47 language strings.
- `FindingCode(StrEnum)`: photo v1 = caries, filling, crown, abrasion, gingivitis; never assessed = calculus, ulcer, discolouration, hypodontia, malocclusion; xray opt-in = periapical_lesion, impacted_tooth. `ASSESSABLE: Mapping[Modality, frozenset[FindingCode]]`.
- `Finding`: code; status: Literal["possible","not_seen","not_assessed"]; score: float | None (0..1); subtype: str | None (source label, never shown to patients in v1); regions: tuple[Region(x0, y0, x1, y1: float 0..1), ...].
- `ModelInfo`: model_id; version; backend: Literal["ollama","mlx","transformers","fixture"]; card_url: str | None.
- `UrgencyHint`: level: Literal["routine","soon","prompt","retake_photo"]; rule_id; table_version; review_status: Literal["illustrative_not_clinically_reviewed","reviewed"]; triggered_by: tuple[FindingCode, ...].
- `PatientText`: locale: Literal["en","ru"]; observations: tuple[str, ...]; urgency_line, see_dentist_line, not_assessed_line, disclaimer: str.
- `AnalysisResult`: analysis_id: UUID; modality: Literal["intraoral_photo","panoramic_xray"]; model: ModelInfo; image_quality: Literal["ok","unusable"]; quality_reason: Literal["blurry","dark","no_teeth","not_dental"] | None; findings: tuple[Finding, ...] (every catalogue code exactly once); urgency; patient_text; vlm_note: str | None; vlm_note_status: Literal["absent","passed","rejected"]; created_at; expires_at.
- `ConsentRequest`: purpose: Literal["photo_analysis","share_with_clinic"]; health_data_explicit: Literal[True]; text_version; locale. `ConsentRecord` = those + consent_id, given_at, expires_at (no IP, no name). Sharing results with a clinic needs its own consent, separate from the photo consent.
- `Money`: amount_minor: int >= 0; currency (^[A-Z]{3}$). `PriceRange`: low, mid, high: Money (same currency, low <= mid <= high). `PriceChip`: Literal["ESTIMATE","SAMPLE","SIMULATED"].
- `ClinicSummary`: clinic_id (slug), name, city, country_code, airport_iata, timezone (IANA), languages, dentists: tuple[DentistSummary(dentist_id, display_name, languages)], treatments: tuple[TreatmentPrice(treatment_code: TreatmentCode, price: Money, chip: Literal["SAMPLE"], trips: int, nights_per_trip: int)], is_fictional: Literal[True]. TreatmentCode: consultation, filling, crown, root_canal, implant, scaling.
- `QuoteRequest`: clinic_id; treatment_codes (min 1); origin_iata (^[A-Z]{3}$); depart_date, return_date: date; travellers: int 1..6; currency.
  - `QuoteLine`: kind: Literal["treatment","flight","hotel"]; trips; amount: PriceRange; chip; basis_key.
  - `Quote`: quote_id (uuid5 of canonical request + as_of date); request; lines; total: PriceRange; total_chip: Literal["ESTIMATE"]; fx_as_of: date; fare_model_version; fare_mdape: float | None; valid_until.
- Booking: `Slot`(slot_id, clinic_id, dentist_id, start, end, status: Literal["available","held","booked"]); `Hold`(hold_id, slot_id, expires_at); `BookingRequest`(hold_id, quote_id, analysis_id: UUID | None, share_consent_id: UUID | None; both or neither); `MockTicket`(kind: Literal["flight","hotel"], reference "DEMO-…", chip: Literal["SIMULATED"]); `Booking`(booking_id, slot, quote_id, status: Literal["confirmed","cancelled"], tickets, shared_analysis_id: UUID | None, simulated: Literal[True]).
- Call objects:
  - `AiDisclosure`: synthetic_voice, machine_translation, not_medical_interpreter, must_acknowledge: Literal[True]; basis: Literal["EU AI Act Art. 50"]; text_key.
  - `CallSession`: session_id; booking_id; room; languages: dict[Literal["patient","dentist"], str]; pair_status: Literal["verified","unverified"]; mode: Literal["consecutive","captions_only"]; ai_disclosure; recording: Literal[False]; expires_at.
  - `CallTokenRequest`(role, ai_disclosure_acknowledged: Literal[True]); `CallToken`(token, livekit_url: str | None, identity, expires_at) (null URL = demo mode, no live room); `CallRoomMetadata`(session_id, languages, mode, glossary_id); `CallContext`(session_id, booking_id, slot: Slot, patient_locale: Literal["en","ru"], analysis: AnalysisResult | None), the dentist's panel data; it never holds an image.
  - `CaptionEvent`: utterance_id; speaker; src_lang, tgt_lang, src_text, tgt_text; final: bool; glossary_missing: tuple[str, ...]; digits_ok: bool; audio: Literal["spoken","captions_only"]; t_speech_end_ms: int; t_first_audio_ms: int | None.
  - `ReplayManifest`(replay_id, tracks: tuple[ReplayTrack(role, lang, url)], events: tuple[CaptionEvent], latency_p50_ms, latency_p95_ms, n).
  - `ChatEvent`: message_id: UUID; sender: Literal["patient","dentist"]; src_lang, tgt_lang; src_text (max 1000 chars); tgt_text: str | None; translation_status: Literal["ok","failed"]; glossary_missing: tuple[str, ...]; digits_ok: bool; sent_at. Clients send raw text on text-stream topic `chat.in.v1`; the agent answers on `chat.v1`, one ChatEvent per message. The source text is always kept; a failed translation shows the source only.
- `ErrorResponse`: code: ErrorCode (StrEnum of the codes in the API table); message_key; detail: str | None; request_id; retry_after_s: int | None.
- Ports (`typing.Protocol`, runtime_checkable, ports.py; fakes in `dentabridge_core.testing`):
  - `VisionBackend`: model: ModelInfo; `async def analyse(self, image: SanitisedImage, modality: Modality) -> VisionOutput`. SanitisedImage(data: bytes JPEG without metadata, width, height, sha256); VisionOutput(image_quality, quality_reason, findings, note_en: str | None, latency_ms).
  - `FareEstimator`: model_version; `def estimate(self, req: FareRequest) -> FareEstimate`. FareRequest(origin_iata, dest_iata, depart, return_, booked_on: date, travellers: int); FareEstimate(range_usd: PriceRange, interval: Literal["q20-q80","q10-q90"], international: bool, basis_key).
  - `Translator`: model_id; `async def translate(self, text: str, src: str, tgt: str, glossary: Sequence[GlossaryEntry]) -> Translation`. GlossaryEntry(src, tgt); Translation(text, latency_ms).
- `VERIFIED_PAIRS: frozenset[tuple[str, str]] = {("en","ru"), ("ru","en")}`.

## API surface (FastAPI, base /v1, JSON; errors are always ErrorResponse + X-Request-Id)
| Endpoint | 2xx | Errors | Idem | Rate |
|---|---|---|---|---|
| GET /health | 200 Health(status, contract_version, mode, vision_backend) | - | - | read |
| GET /meta?locale | 200 Meta(uploads_enabled, assessable, texts: dict[str,str], verified_pairs, currencies, fx_as_of) | 422 | - | read |
| GET /samples, /samples/{id}/image | 200 list[SampleImage(id, attribution, licence)], image/jpeg | 404 | - | read |
| POST /consents (purpose photo_analysis or share_with_clinic) | 201 ConsentRecord | 422 | - | write |
| POST /analyses (multipart file, consent_id, modality, locale) | 201 AnalysisResult | 400 image_unreadable, 403 consent_required / uploads_disabled, 410 consent_expired, 413 payload_too_large, 415 unsupported_media_type, 503 backend_unavailable | yes | analysis |
| POST /analyses/sample {sample_id, locale} | 201 AnalysisResult | 404 | yes | analysis |
| GET, DELETE /analyses/{id} | 200, 204 | 404, 410 gone | - | read |
| GET /treatments, /cities?q&limit, /clinics?treatment, /clinics/{id} | 200 | 404, 422 | - | read |
| POST /quotes | 201 Quote (same id for the same request and day) | 404, 422 trip_too_short / currency_unsupported / unknown_airport | by construction | write |
| GET /quotes/{id} | 200 | 404, 410 | - | read |
| GET /clinics/{id}/slots?from&to (max 31 days) | 200 list[Slot] | 404, 422 | - | read |
| POST /holds {slot_id}; DELETE /holds/{id} | 201 Hold; 204 | 404, 409 slot_unavailable | yes | write |
| POST /bookings {hold_id, quote_id, analysis_id?, share_consent_id?} | 201 Booking | 403 share_consent_required, 404, 409 hold_expired / quote_mismatch / outside_trip, 410 quote expired / analysis_expired | yes | write |
| GET /bookings/{id}; POST /bookings/{id}/cancel | 200 | 404 | by state | write |
| POST /bookings/{id}/call | 201 CallSession (200 if it exists) | 404, 409 booking_cancelled, 422 language_pair_unsupported | yes | write |
| POST /calls/{session_id}/tokens | 201 CallToken (livekit_url is null when no LiveKit is configured) | 404, 422 disclosure_not_acknowledged | - | write |
| GET /calls/{session_id}/context (Bearer call token, role dentist) | 200 CallContext | 401 invalid_call_token, 403 wrong_role, 404, 410 gone | - | read |
| GET /replays/{id}, /replays/{id}/audio/{file} | 200 ReplayManifest, audio/ogg | 404 | - | read |
- Common errors on any route: 400 idempotency_key_required, 409 idempotency_in_progress, 422 validation_error / idempotency_key_reuse, 429 rate_limited, 500 internal_error.
- Idempotency: `Idempotency-Key` header (8-64 chars of [A-Za-z0-9-]) where Idem = yes. Stored in SQLite (key, route, body_sha256, status, body) for 24 h. A replay returns the stored status and body plus `Idempotent-Replayed: true`; another body under the same key → 422.
- Rate limits: in-memory per client, key = sha256(ip + daily salt), never persisted. analysis 10/min, write 30/min, read 120/min. 429 sets `Retry-After` and the `RateLimit-*` headers.
- CORS:
  - allow-list `DENTABRIDGE_CORS_ORIGINS` (default http://localhost:3000, plus the Space origin by config); `*` refuses startup;
  - no credentials; methods GET, POST, DELETE; headers Content-Type, Idempotency-Key, Accept-Language;
  - exposes X-Request-Id, Retry-After, RateLimit-*, Idempotent-Replayed; max_age 600.
- Site deliverables in contracts/: openapi.json, ts/src/schema.d.ts (openapi-typescript 7.x), ts/src/index.ts (openapi-fetch client), fixtures/<scenario>/<nn-step>.json recorded in demo mode.

## Rules
- Uploads: JPEG/PNG/WebP up to 8 MB and 40 MP, decoded in memory. EXIF orientation is applied, then the image is re-encoded to JPEG q90 at a 1536 px long side with no metadata. Never written to disk or logged; logs hold request_id, route, status and ms only. Consent must be explicit, unexpired and on the current text_version. Results and consent expire after 24 h (purged on access and hourly). `DENTABRIDGE_UPLOADS=off` on the Space.
- Slots and booking:
  - 60-min slots come from the weekly local schedule in seed/clinics.yaml (zoneinfo) and are stored in UTC, with unique(dentist_id, start); 90-day horizon, lead time >= 48 h.
  - A hold lasts 10 min, with at most one active hold per slot (partial unique index); expiry is checked lazily against the Clock.
  - Confirm requires an active hold, an unexpired quote for the same clinic, and an appointment local date strictly between depart_date and return_date. Cancel frees the slot and is idempotent. A booking issues SIMULATED MockTickets.
- Fare estimator v1 (pure, offline, deterministic):
  - Distance: haversine km between OurAirports coordinates (R 6371.0088).
  - mid_usd = travellers × (fixed + Σ concave per-km band rates over [0, 800, 2500, 6000, ∞) km) × season[quarter(depart)] × advance[bin(depart − booked_on)]. Advance bins: 0-6, 7-13, 14-20, 21-44, 45-90, 91-180, 181+ days.
  - low/high = mid × residual quantiles: q20/q80 when both airports are in one country, q10/q90 otherwise.
  - The fit uses BTS DB1B (distance taper, quarterly season) and Kaggle Flight Prices (advance curve), holding out whole city pairs in 5 grouped folds.
  - Committed: data/fare_params_v1.json with the coefficients, dataset versions, fit date, MdAPE, interval coverage and bias per band. Raw data is never committed. Until the fit exists, v0 hand-set priors ship with fare_mdape = null.
  - Basis line: "typical economy round trip, calibrated on US domestic fares; international error not measured".
  - Hotels: hand-set per-night ranges by city tier (seed/hotel_tiers.yaml), chip ESTIMATE. Treatment prices: chip SAMPLE. FX: a dated ECB snapshot via Frankfurter; a currency missing from it → 422 currency_unsupported, with no fallback. RUB is not in the ECB set (no reference rate since 2022), so it is unsupported by design, and /meta lists only the currencies the snapshot holds.
- Urgency table v1 (illustrative, first match wins):
  - R0 unusable → retake_photo;
  - R1 caries possible with a severity subtype → prompt (only if ML2 confirms that the subtype means severity);
  - R2 caries possible → soon; R3 gingivitis possible → soon; R9 otherwise → routine.
  - Every text says "cannot rule out" and "see a dentist".

## Tasks
### F0 Repo skeleton [L] WS00 M0 (deps none)
- Files: pyproject.toml (workspace); packages/{core,ml,api,pricing,voice}/pyproject.toml + src/dentabridge_<pkg>/__init__.py; scripts/check (start from `~/.claude/bin/new-repo --strict`, then add uv run + mypy; F2 adds the lint step); Makefile + mk/{core,ml,api,voice,deploy}.mk stubs; .github/workflows/ci.yml; .gitignore; LICENSE (MIT); CLAUDE.md; .python-version; web/README.md.
- Tests: test_every_package_imports (each dentabridge_* imports); test_data_dirs_ignored (`git check-ignore` holds for data/x, private/x, var/x, *.gguf, .env).
- Verify: `scripts/check && make help`
### F1 Contracts and ports [L] WS00 M0 (deps F0)
- Files: packages/core/src/dentabridge_core/{findings,analysis,urgency,money,clinic,quote,booking,call,consent,errors,ports,testing,version}.py; packages/core/schemas/*.json (snapshots).
- Signatures exactly as in Shared contracts. testing.py holds FakeVisionBackend(outputs: Mapping[str, VisionOutput]), FakeFareEstimator(), FakeTranslator(prefix: str).
- Tests: test_schema_snapshots_match (model_json_schema == committed file); test_price_range_rejects_disorder; test_analysis_requires_every_catalog_code; test_disclosure_fields_literal_true; test_fakes_satisfy_ports (isinstance on the protocols).
- Verify: `uv run pytest packages/core -q`
### F2 Claims linter [L] WS00 M1 (deps F1)
- Files: dentabridge_core/policy/{claims.py, claims_rules.yaml, lint_cli.py}; tests/policy/corpus/{en,ru}_{pass,fail}.txt.
- Signatures: `lint(text: str, locale: Literal["en","ru"], profile: Literal["patient","docs"]) -> tuple[Violation, ...]`; `Violation(rule_id: str, start: int, end: int, excerpt: str)`. CLI `dentabridge-lint --profile P --locale L PATH...` lints .json values and .md text and exits 1 on any violation.
- Rules:
  - banned claims: diagnos*, detects disease, clinical-grade, FDA/CE/MHRA/HIPAA approved/cleared/compliant, replaces your dentist, guarantee, cure (plus RU equivalents);
  - verdict sentences: second person + finding without a hedge ("you have caries", "у вас кариес");
  - the docs profile allowlists exact disclaimer sentences only.
- Tests: test_fail_corpus_every_line_flagged; test_pass_corpus_clean; test_disclaimer_allowed_only_in_docs_profile; test_cli_exit_codes.
### F3 Urgency rule table and review record [S] WS00 M1 (deps F1)
- Files: policy/{urgency.py, urgency_v1.yaml}; docs/clinical-review/README.md. Review record schema: reviewer_name, qualification, registration, reviewed_on, table_sha256, verdicts{rule_id: agree|change|remove, comment}, statement.
- Signatures: `load_table(path: Path) -> UrgencyTable`; `evaluate(findings: Sequence[Finding], quality: Literal["ok","unusable"], table: UrgencyTable) -> UrgencyHint`; `review_status(table_path: Path, reviews_dir: Path) -> Literal["illustrative_not_clinically_reviewed","reviewed"]`.
- Tests: test_every_status_combination_hits_one_rule (exhaustive 3^5 × 2); test_adding_possible_never_lowers_level; test_reviewed_only_when_sha_matches; test_illustrative_by_default.
### F4 Patient text renderer [S] WS00 M1 (deps F2, F3)
- Files: policy/render.py; i18n/{en,ru}.json (finding × status, urgency levels, banner, disclaimers, ai_disclosure, estimate footer, chips, consent text v1).
- Signatures: `render_patient_text(findings: Sequence[Finding], urgency: UrgencyHint, locale: Literal["en","ru"]) -> PatientText`; `guard_note(note: str | None) -> tuple[str | None, Literal["absent","passed","rejected"]]`; `texts(locale: str) -> Mapping[str, str]` (feeds /meta).
- Tests: test_keys_identical_across_locales; test_every_rendering_passes_patient_lint (hypothesis over finding combinations × locale); test_see_dentist_always_present; test_note_with_verdict_rejected.
- Verify (F2-F4): `uv run pytest packages/core/tests/policy -q && uv run dentabridge-lint --profile patient --locale ru packages/core/src/dentabridge_core/i18n/ru.json`
### ML1 Ollama round-trip pilot [L] WS01 M1 (deps F0; first ML task)
- Files: dentabridge_ml/pilot/roundtrip.py; packages/ml/configs/pilot_roundtrip.yaml; docs/results/ml/roundtrip.{json,md}.
- Procedure: untuned Qwen3-VL-4B + 20-step PEFT LoRA on 20 AlphaDent HF rows (MPS), then merge. Path A: `ollama create` from the merged safetensors dir. Path B: llama.cpp `convert_hf_to_gguf.py` text + mmproj, a Modelfile with the projector, then `ollama create`. Compare both with Transformers greedy output on 20 images.
- Signatures: `run_roundtrip(cfg: RoundtripConfig) -> RoundtripReport(path: Literal["safetensors","gguf_mmproj"], vision_ok: bool, json_valid_rate: float, agreement: float, logprobs_available: bool, ollama_version: str)`; `serving_decision(reports: Sequence[RoundtripReport]) -> Literal["ollama","mlx_or_transformers"]`.
- Pass = vision_ok, agreement >= 0.9 and json_valid_rate >= 0.95. On fail, the tuned model is served by MLX (Mac) and Transformers (CUDA), and Ollama keeps the baselines.
- Tests: test_decision_rule_table; test_report_roundtrips_json. The pilot itself is marked `gpu` and run by hand.
- Verify: `uv run python -m dentabridge_ml.pilot.roundtrip --config packages/ml/configs/pilot_roundtrip.yaml`
### ML2 Data download, lock, canonical records [L] WS01 M1 (deps F1)
- Files: dentabridge_ml/data/{sources.py, download.py, lock.py, records.py}; packages/ml/{data.lock.json, LICENCES.yaml, docs/data-card.md}.
- Sources: `SOURCES: Mapping[SourceId, SourceSpec(url, licence, published: bool, coverage: frozenset[FindingCode], camera: Literal["phone","clinic","xray"])]` for aku_caries, alphadent, mendeley_gingivitis, dentex (opt-in) and kaggle_oral_private (private).
- Signatures: `download(source: SourceId, root: Path) -> list[Path]`; `verify_lock(root: Path, lock: Path) -> None` (raises LockMismatch); `load_records(source: SourceId, root: Path) -> list[ImageRecord(record_id, source, path, width, height, patient_id: str | None, boxes: tuple[LabelBox, ...], image_labels: frozenset[FindingCode], caption: str | None)]`.
- Resolve and date in the data card: AlphaDent 1,320 vs 1,455 (count + hash, hidden-test rows?); caries 1-6 = severity or Black class; whether patient ids exist; Mendeley classes (healthy controls?) and caption language; DINOv2 licence.
- Tests: test_lock_detects_tamper; test_private_source_only_under_private_dir; test_coverage_declared_for_published_sources; test_loader_golden_per_format (tiny synthetic fixtures).
- Verify: `make data && uv run pytest packages/ml/tests/data -q`
### ML3 Pseudo-patient grouping, splits, leakage tests [L] WS01 M1 (deps ML2)
- Files: dentabridge_ml/splits/{embed.py, group.py, manifest.py}; packages/ml/manifests/{train,val,test}.jsonl + manifests.sha256.
- Signatures: `embed(paths: Sequence[Path], model_id: str) -> np.ndarray`; `phash(path: Path) -> int`; `group(records: Sequence[ImageRecord], emb: np.ndarray, hashes: Sequence[int], max_cos_dist: float, max_hamming: int) -> list[str]` (a real patient_id wins; group_method recorded per record); `split(records, groups, seed: int, ratios: tuple[float, float, float]) -> SplitManifest` (stratified by source × finding).
- Tests: test_no_group_in_two_splits; test_no_near_duplicate_across_splits; test_no_high_cosine_across_splits; test_manifest_deterministic (same seed → same sha256); test_private_and_optin_sources_absent_from_published_manifests.
- Verify: `make splits && uv run pytest packages/ml/tests/splits -q`
### ML4 Training targets [S] WS01 M2 (deps ML3)
- Files: dentabridge_ml/targets/{prompt.py, sft.py, degrade.py}.
- Signatures: `build_prompt(assess: frozenset[FindingCode], modality: Modality) -> str`; `target_json(rec: ImageRecord, assess: frozenset[FindingCode]) -> str`; `degrade(img: Image.Image, kind: Literal["blur","dark","overexposed","crop_no_teeth"], seed: int) -> Image.Image`; `write_sft(manifest: SplitManifest, split: str, out: Path, negatives_frac: float) -> int`.
- assess = the source's coverage, so findings without labels are never trained as not_seen.
- Tests: test_target_limited_to_source_coverage; test_target_parses_as_vision_output; test_negative_fraction_within_1pct; test_degrade_deterministic.
### ML5 VisionBackend adapters [L] WS01 M2 (deps F1, ML1)
- Files: dentabridge_ml/serving/{parse.py, ollama.py, transformers_backend.py, mlx_backend.py, factory.py}.
- Signatures:
  - `parse_output(raw: str, token_scores: Mapping[FindingCode, float] | None, release: ReleaseConfig) -> VisionOutput` (raises ParseError, no repair);
  - `OllamaVisionBackend(base_url: str, model: str, release: ReleaseConfig)`; `TransformersVisionBackend(model_dir: Path, device: str, release: ReleaseConfig)`; `MlxVisionBackend(model_dir: Path, release: ReleaseConfig)`;
  - `load_vision_backend(spec: str) -> VisionBackend` for "ollama:<tag>" | "transformers:<dir>" | "mlx:<dir>";
  - `ReleaseConfig(assessed: frozenset[FindingCode], thresholds: Mapping[FindingCode, float], needs_scores: bool)`, read from release.json next to the weights.
- Score = probability of "possible" at the status token (logprobs). An adapter without logprobs raises at construction if needs_scores is set.
- Tests: test_parse_golden (10 raw outputs, 3 invalid → ParseError); test_ollama_request_shape (httpx.MockTransport: images, JSON-schema format, temperature 0); test_release_marks_unshipped_codes_not_assessed; test_factory_rejects_unknown_scheme.
### ML6 Evaluation harness [L] WS01 M2 (deps ML3, ML5)
- Files: dentabridge_ml/eval/{predict.py, metrics.py, calibration.py, report.py, gallery.py}; packages/ml/configs/release_gate.yaml (pre-registered); docs/results/ml/test-access.log.
- Signatures:
  - `predict(backend: VisionBackend, manifest: SplitManifest, split: Literal["val","test"], out: Path, final: bool = False) -> Path` (test requires final=True and appends to the log);
  - `binary_stats(preds, labels, code: FindingCode, groups: Sequence[str], seed: int) -> BinaryStats(tp, fp, tn, fn, sens, spec, sens_ci, spec_ci)` (cluster bootstrap over groups, 2,000 reps);
  - `ece(scores: np.ndarray, labels: np.ndarray, bins: int) -> float`; `risk_coverage(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]` (curve + AURC);
  - `shift(preds, records) -> ShiftReport` (caries: AKU phone vs AlphaDent clinic); `gallery(preds, records, k: int, out: Path) -> None` (top FP/FN per code, published sources only, with attribution);
  - `report(pred_files: Sequence[Path], gate: Path, out_json: Path, out_md: Path) -> None` (adds json_valid_rate and the FP rate on images labelled absent).
- Tests: test_binary_stats_known_counts; test_ci_deterministic_and_contains_point; test_ece_zero_when_perfect; test_risk_coverage_monotone; test_uncovered_source_codes_not_scored (AKU never scored for filling); test_test_split_requires_final_and_logs; test_gallery_excludes_private.
- Verify: `uv run pytest packages/ml/tests/eval -q && make eval-val`
### ML7 Baselines [L] WS01 M2 (deps ML5, ML6)
- Files: dentabridge_ml/baselines/{zero_shot.py, detector.py}; configs/detector.yaml.
- Rows: zero-shot Ollama qwen3-vl:4b and :8b with the tuned prompt; D-FINE (ustc-community, Transformers) fine-tuned on caries/filling/crown/abrasion boxes, with gingivitis reported as "not covered"; AlphaDent YOLOv8x evaluated under private/ only.
- Signatures: `train_detector(cfg: DetectorConfig) -> Path`; `detector_predict(ckpt: Path, manifest: SplitManifest, split: str, out: Path) -> Path` (image score = max box score per code).
- Tests: test_image_score_is_max_box; test_no_ultralytics_import_in_packages (AST scan); test_zero_shot_prompt_equals_tuned_prompt.
- Verify: `make baselines` (writes docs/results/ml/val.json)
### ML8 Training recipe, portable CUDA/ROCm/MLX [L] WS01 M2 (deps ML4)
- Files: dentabridge_ml/train/{config.py, collate.py, sft.py}; configs/{lean-t4.yaml, mi300x-*.yaml, mlx-2b.yaml}; notebooks/kaggle_lean.ipynb (thin: install uv, run the CLI).
- Signatures:
  - `TrainConfig(base_id: str, image_max_px: int, lora_r: int, lora_alpha: int, train_vision: bool, quant: Literal["none","nf4"], dtype: Literal["fp16","bf16"], lr: float, epochs: int, grad_accum: int, save_every: int, push_to: str | None, seed: int)`;
  - `train(cfg: TrainConfig) -> Path` (TRL SFTTrainer + PEFT, loss on answer tokens only); `mlx_train(cfg: TrainConfig) -> Path` (mlx-vlm LoRA CLI).
- Lean T4 recipe: 4B, nf4, fp16, r 16, alpha 16, 448 px, 2 epochs, lr 1e-4, checkpoint every 500 steps. P100 is not used.
- Tests: test_collate_masks_prompt_and_image_tokens; test_lean_config_is_t4_safe; test_rocm_config_has_no_bitsandbytes; test_two_steps_tiny_model [slow].
- Verify: `uv run pytest packages/ml/tests/train -q`; a 50-step pilot writes docs/results/ml/throughput.json (T4, MI300X, M3 Max).
### ML9 Runs, release gate, ship decision [L] WS01 M3 (deps ML6, ML7, ML8)
- Runs: Kaggle lean; MI300X ablations, one change each vs lean in bf16 (train_vision, 672 px, no quality negatives, r 64); one larger run (best combination, 3 epochs); MLX 2B.
- MI300X cap 40 GPU-h (about $80), with a $20 reserve; the first hour is the 50-step pilot. Selection happens on val; test is evaluated once for the chosen model and every baseline.
- Gate (committed before the first test access): a finding ships as assessed if n_pos >= 30, the lower CI bounds of sens and spec are both >= 0.60, and ECE <= 0.10; otherwise not_assessed.
- Ship rule: the detector replaces the VLM only if it beats the VLM's sensitivity at the VLM's specificity beyond the CI on >= 3 of 4 shared codes. That would add a DetectorVisionBackend.
- Files: dentabridge_ml/release.py; docs/results/ml/{runs.json, test.json, test.md, gallery/}; release.json next to the weights.
- Signatures: `apply_gate(test: Path, gate: Path) -> ReleaseConfig`; `ship_decision(test: Path) -> Literal["vlm","detector"]`.
- Tests: test_gate_thresholds; test_gate_committed_before_first_test_access (git log time < first log line); test_runs_json_schema.
- Verify: `make eval-test && uv run pytest packages/ml/tests/release -q`
### ML10 Model card and publishing [S] WS01 M3 (deps ML9)
- Files: docs/model-cards/dentabridge-photo-v1.md (generated); dentabridge_ml/publish.py.
- Signatures: `render_card(results: Path, template: Path, out: Path) -> None`; `publish(weights_dir: Path, repo_id: str, private: bool) -> str` (LoRA, merged weights, GGUF + mmproj if ML1 passed, release.json).
- Card: intended use (research); out of scope (diagnosis, emergencies, minors); data, licences and attribution; metrics with CIs; shift; gallery; limitations; "not a medical device".
- Tests: test_card_numbers_equal_results; test_card_passes_docs_lint; test_publish_called_private_until_owner_flag.
### ML11 DENTEX X-ray model, opt-in NC-SA [L] WS01 M5 (deps ML9)
- Separate configs, release and HF repo with a CC BY-NC-SA 4.0 card. Modality panoramic_xray; codes caries, periapical_lesion, impacted_tooth. Data via `make data SOURCES=dentex`. The API serves it only when `DENTABRIDGE_XRAY_BACKEND` is set.
- Tests: test_photo_release_excludes_dentex; test_xray_card_declares_nc_sa; test_xray_endpoint_403_when_unset.
### A1 API skeleton [L] WS02 M1 (deps F1)
- Files: dentabridge_api/{app.py, settings.py, errors.py, clock.py, db/{models.py, session.py}, routes/meta.py}; packages/api/alembic/.
- Signatures:
  - `create_app(settings: Settings, clock: Clock, vision: VisionBackend, fares: FareEstimator) -> FastAPI`;
  - `Settings` (pydantic-settings, prefix DENTABRIDGE_): mode, uploads, cors_origins, db_url, vision_backend, xray_backend, livekit_url/key/secret, result_ttl_h, replays_dir;
  - `class Clock(Protocol): def now(self) -> datetime`, implemented by FrozenClock(at) and SystemClock();
  - CLI `dentabridge-api serve|seed|export`: serve picks FixtureVisionBackend in demo mode, else `dentabridge_ml.serving.load_vision_backend(settings.vision_backend)` (ML5). Until ML5 lands, non-demo mode fails at startup.
- Tests: test_health_and_meta_200; test_errors_use_error_response_and_request_id; test_cors_preflight_only_allowlisted; test_wildcard_origin_refused; test_alembic_upgrade_head_on_empty_sqlite.
- Verify: `uv run pytest packages/api/tests/test_app.py -q`
### A2 Rate limit and idempotency [S] WS02 M1 (deps A1)
- Files: dentabridge_api/{ratelimit.py, idempotency.py}.
- Signatures: `RateLimiter(budgets: Mapping[RouteClass, int], clock: Clock).check(client_key: str, cls: RouteClass) -> RateDecision`; `IdempotencyStore(session_factory, clock: Clock, ttl: timedelta)` with `.begin(key: str, route: str, body_sha256: str) -> StoredResponse | None` and `.complete(key: str, status: int, body: bytes) -> None`.
- Tests: test_429_after_budget_with_retry_after; test_ratelimit_headers; test_replay_same_status_and_body; test_reuse_with_other_body_422; test_missing_key_400; test_in_progress_409; test_raw_ip_never_stored.
### A6 Contracts export: OpenAPI, TS client, fixtures [S] WS02 M1 (deps A1; fixtures grow with each endpoint)
- Files: dentabridge_api/export.py; contracts/{openapi.json, ts/package.json, ts/src/schema.d.ts, ts/src/index.ts, fixtures/}; target `contracts` in mk/api.mk.
- Signatures: `export_openapi(out: Path) -> None`; `record_fixtures(scenario: Literal["happy_path","errors"], out: Path) -> None` (demo mode, TestClient, frozen clock).
- Tests: test_openapi_matches_committed; test_fixtures_regenerate_identically; test_every_operation_has_operation_id_and_error_schema. CI runs `make contracts && git diff --exit-code contracts/`.
### A3 Consents, analyses, uploads, demo vision [L] WS02 M1 (deps A2, F4)
- Files: dentabridge_api/{routes/consents.py, routes/analyses.py, uploads.py, analysis_service.py, demo/fixture_vision.py, demo/samples/}.
- Signatures: `sanitise_upload(data: bytes, content_type: str, max_bytes: int, max_pixels: int) -> SanitisedImage` (raises UploadRejected(code)); `run_analysis(image: SanitisedImage, modality: Modality, locale: Literal["en","ru"], backend: VisionBackend, clock: Clock) -> AnalysisResult` (validate → urgency → render → guard_note); `FixtureVisionBackend(samples_dir: Path)`; `precompute_samples(backend: VisionBackend, samples_dir: Path) -> None` (make target `precompute-samples`, run at M4 with the shipped model).
- Samples: 12 test-split images from AlphaDent and Mendeley with CC BY attribution. AKU is excluded because its participants include minors.
- Tests: test_exif_gps_removed; test_orientation_applied; test_400_413_415; test_consent_required_403; test_uploads_off_403; test_no_file_written; test_no_image_bytes_in_logs; test_result_410_after_ttl; test_fixture_backend_deterministic; test_runtime_lint_violation_raises; test_meta_texts_pass_patient_lint.
### P1 Airports, cities, FX snapshots [S] WS02 M1 (deps F1)
- Files: dentabridge_pricing/{airports.py, fx.py}; packages/pricing/data/{airports_<date>.csv.gz, fx_<date>.json}; packages/pricing/scripts/build_snapshots.py (network); packages/pricing/LICENCES.yaml.
- Signatures: `load_airports() -> Mapping[str, Airport(iata, name, municipality, iso_country, lat, lon)]`; `search_cities(q: str, limit: int) -> list[Airport]`; `haversine_km(a: Airport, b: Airport) -> float`; `convert(m: Money, to: str, fx: FxSnapshot) -> Money` (raises CurrencyUnsupported).
- Tests: test_haversine_lhr_jfk (5,540 ± 10 km); test_search_case_insensitive_prefix; test_convert_roundtrip_within_one_minor; test_unsupported_currency_raises; test_only_scheduled_iata_airports.
### P2 Fare estimator, calibration, offline evaluation [L] WS02 M2 (deps P1)
- Files: dentabridge_pricing/{fares.py, calibrate.py, evaluate.py}; data/fare_params_v{0,1}.json; docs/results/fares.{json,md}.
- Signatures: `OfflineFareEstimator(params: FareParams, airports: Mapping[str, Airport])` implements FareEstimator; `fit(db1b_dir: Path, flightprices_dir: Path, seed: int) -> FareParams`; `evaluate(params: FareParams, db1b_dir: Path, folds: int) -> FareEval(mdape, interval_coverage, bias_by_band)`.
- Tests: test_symmetric; test_monotone_in_distance (hypothesis); test_low_le_mid_le_high; test_linear_in_travellers; test_international_uses_q10_q90; test_deterministic; test_params_record_datasets_and_metrics.
- Verify: `uv run pytest packages/pricing -q`; maintainer: `make eval-fares`.
### P3 Quote assembly [S] WS02 M2 (deps P2)
- Files: dentabridge_pricing/{quote.py, hotels.py}; seed/hotel_tiers.yaml.
- Signature: `build_quote(req: QuoteRequest, clinic: ClinicSummary, fares: FareEstimator, hotels: HotelTable, fx: FxSnapshot, as_of: datetime) -> Quote`.
- Rules: trips = max over the treatments; flights × trips; hotel = nights_per_trip × trips × ceil(travellers / 2) rooms; valid_until = as_of + 24 h; return − depart >= nights_per_trip.
- Tests: test_total_bounds_are_sums; test_two_trip_treatment_doubles_flights; test_quote_id_stable; test_trip_too_short; test_every_line_has_chip_and_basis.
### A4 Clinics, slots, holds, bookings [L] WS02 M2 (deps A2, A3, P3)
- Files: dentabridge_api/{booking/slots.py, booking/holds.py, booking/bookings.py, routes/clinics.py, routes/quotes.py, routes/booking.py}; seed/{clinics.yaml, schema.json}. Seed: 4 fictional clinics in real cities on 3 continents, covering DST and non-DST zones.
- Signatures: `generate_slots(clinic: ClinicSeed, start: date, days: int) -> list[Slot]`; `create_hold(slot_id: UUID, clock: Clock) -> Hold`; `confirm(req: BookingRequest, clock: Clock) -> Booking`; `cancel(booking_id: UUID) -> Booking`. Sharing: when analysis_id is set, share_consent_id must be an unexpired share_with_clinic consent (else 403 share_consent_required) and the AnalysisResult snapshot is copied onto the booking; it is purged 24 h after the appointment end or on cancel.
- Tests: test_no_double_booking (hypothesis RuleBasedStateMachine over hold/confirm/cancel/expire); test_dst_spring_forward_slots; test_hold_expires_after_10_min; test_quote_other_clinic_409; test_outside_trip_409; test_cancel_idempotent; test_lead_time_48h; test_seed_all_fictional; test_share_requires_own_consent_403; test_snapshot_purged_after_appointment_plus_24h_or_on_cancel.
### A5 Calls, dentist context and replays endpoints [L] WS02 M2 (deps A4)
- Files: dentabridge_api/{routes/calls.py, routes/call_context.py, call_tokens.py, routes/replays.py}.
- Signatures: `create_session(booking: Booking, languages: Mapping[str, str], clock: Clock) -> CallSession` (room "db-<booking_id>", identity "<role>-<session_id>", metadata = CallRoomMetadata); `mint_token(session: CallSession, role: str, settings: Settings) -> CallToken` (livekit-api; TTL 2 h; join, mic publish, subscribe, data; signed with settings.call_secret generated into .env; livekit_url null when unset); `read_call_token(token: str, settings: Settings) -> CallClaims(session_id, role, identity)` (raises InvalidCallToken); `build_context(booking: Booking, session: CallSession, snapshot: AnalysisResult | None, patient_locale: Literal["en","ru"]) -> CallContext`.
- Tests: test_token_requires_ack_422; test_token_grants_room_and_identity (decode JWT); test_call_on_cancelled_booking_409; test_no_livekit_url_gives_token_with_null_url; test_replay_manifest_validates_against_core; test_context_requires_dentist_token (patient token → 403); test_context_without_share_has_no_analysis; test_context_never_contains_image_bytes; test_context_works_in_demo_mode.
### V1 Pipeline core and safety checks [L] WS03 M1 (deps F1)
- Files: dentabridge_voice/{pipeline.py, segment.py, checks.py, ports.py}; packages/voice/glossary/en-ru.tsv (about 200 terms); packages/voice/config/pairs.yaml.
- Signatures:
  - `SpeechToText.transcribe(pcm: bytes, sample_rate: int, lang: str) -> Awaitable[str]`; `TextToSpeech.synthesise(text: str, lang: str) -> AsyncIterator[bytes]`;
  - `Pipeline(stt, translator, tts, glossary, clock).handle(utt: Utterance, src: str, tgt: str) -> AsyncIterator[CaptionEvent | AudioChunk]`;
  - `missing_terms(src: str, tgt: str, entries: Sequence[GlossaryEntry]) -> tuple[str, ...]`; `digits_preserved(src: str, tgt: str) -> bool`.
- Tests: test_caption_before_audio; test_tts_failure_emits_captions_only; test_missing_glossary_term_reported; test_digit_swap_flagged ("tooth 36" → "зуб 63"); test_segments_capped_at_10s; test_timings_monotonic.
### V2 Model adapters [L] WS03 M1 (deps V1)
- Files: dentabridge_voice/adapters/{parakeet_mlx.py, faster_whisper.py, ollama_translator.py, piper_http.py}; packages/voice/LICENCES.yaml.
- Signatures: `ParakeetMlxSTT(model_id: str)`; `FasterWhisperSTT(model: str, device: Literal["cpu","cuda"], compute_type: str)`; `OllamaTranslator(base_url: str, model: str)` (prompt carries only the matched glossary terms); `PiperHttpTTS(base_url: str, voices: Mapping[str, str])`. Default voices must have CC0 or permissive data; anything else is opt-in.
- Tests: test_prompt_contains_only_matched_terms; test_translator_mock_transport; test_piper_mock_transport; real-model tests [slow].
### V3 LiveKit worker [L] WS03 M2 (deps V2; uses CallRoomMetadata from F1)
- Files: dentabridge_voice/{worker.py, routing.py}.
- Signatures: `entrypoint(ctx: JobContext) -> Awaitable[None]` (livekit-agents 1.8.x); `route(participants: Sequence[ParticipantInfo], meta: CallRoomMetadata) -> list[Direction(speaker_identity, src, tgt)]`. Publishes track `translated-<role>` and text stream topic `captions.v1` (CaptionEvent JSON). CLI `dentabridge-voice worker` (livekit-agents `start`).
- Tests: test_never_subscribes_agent_or_translated_tracks; test_two_participants_two_directions; test_unverified_pair_passes_flag; e2e [slow]: the `lk` CLI publishes a wav into a local room, and at least one caption and one audio frame arrive.
### V4 Latency and quality harness, replay [L] WS03 M4 (deps V3)
- Files: dentabridge_voice/harness/{run.py, metrics.py, script_en_ru.yaml (50 utterances)}; packages/voice/replays/demo/; docs/results/voice.{json,md}.
- Signatures: `run_harness(pipeline: Pipeline, clips: Sequence[Clip]) -> HarnessReport(stage_ms, p50_ms, p95_ms, wer, chrf, glossary_acc, digit_acc)`; `record_replay(events: Sequence[CaptionEvent], audio_dir: Path, out: Path) -> ReplayManifest`.
- Audio: a scripted dialogue (owner-read or synthetic) plus FLEURS en/ru, never patient audio.
- Tests: test_percentiles_known; test_report_schema; test_replay_matches_core_schema; test_deterministic_with_fakes.
- Verify: `make voice-bench` (Mac native, then the cpu profile)
### V5 Chat translation [S] WS03 M2 (deps V2)
- Files: dentabridge_voice/chat.py; packages/voice/tests/test_chat.py.
- Signature: `ChatTranslator(translator: Translator, glossary: Sequence[GlossaryEntry], clock: Clock).handle(raw: str, sender: Literal["patient","dentist"], src: str, tgt: str) -> Awaitable[ChatEvent]`. V3's worker wires text-stream topic `chat.in.v1` → `chat.v1`.
- Rules: reject empty or over 1000 chars (ValueError); a translator failure returns translation_status "failed" with tgt_text null; glossary and digit checks reuse V1 `checks.py`.
- Tests: test_chat_event_matches_core_schema; test_digit_swap_flagged_in_chat; test_translator_failure_keeps_source; test_message_length_capped; test_worker_ignores_its_own_chat_events.
### D1 Compose profiles and make path [L] WS04 M4 (deps A3, A5, V3)
- Files: deploy/compose.yaml; deploy/docker/{api,voice}.Dockerfile; deploy/livekit.yaml; mk/deploy.mk; .env.example.
- Profiles: demo (api only); cpu (api, ollama, livekit, piper, voice-cpu); cuda (the same with GPU reservations); mac (api, livekit and piper in Docker; voice worker and Ollama native).
- Targets: `make demo | up-cpu | up-cuda | mac | down | models`; they start `dentabridge-api serve` and `dentabridge-voice worker`. `models` prints each gated or opt-in licence and asks before downloading it.
- Tests: test_compose_config_valid_per_profile (`docker compose --profile X config -q`); test_no_committed_secrets (livekit keys generated into .env); test_demo_smoke [slow] (health 200 within 60 s).
### D2 README, licences, docs lint [S] WS04 M5 (deps ML10, P2, V4)
- Files: README.md; docs/licences.md; tools/{render_readme.py, render_licences.py}.
- Signatures: `render_tables(results_dir: Path, readme: Path) -> None` (between markers); `render_licences(tables: Sequence[Path], out: Path) -> None`.
- Tests: test_readme_tables_match_results; test_every_model_and_dataset_id_has_licence_row; test_readme_and_cards_pass_docs_lint; test_banned_names_absent (list read from outside the repo).
### D3 HF Space and demo kit [S] WS04 M5 (deps D1, ML10)
- Files: space/{README.md, Dockerfile}; docs/demo/{script.md, shotlist.md}.
- The Space runs the API in demo mode with uploads off, precomputed samples, the live estimator and replay; it later mounts the static site at "/". It stays private.
- Tests: test_space_env_uploads_off; test_space_samples_have_attribution.

## Dependency graph
F0 → F1 → {F2, F3, A1, P1, ML2, V1}; F0 → ML1; {F2, F3} → F4
A1 → {A2, A6}; {A2, F4} → A3; P1 → P2 → P3; {A2, A3, P3} → A4 → A5
ML2 → ML3 → ML4 → ML8; {F1, ML1} → ML5; {ML3, ML5} → ML6; {ML5, ML6} → ML7; {ML6, ML7, ML8} → ML9 → {ML10, ML11}
V1 → V2 → {V3, V5}; V3 → V4. D1 needs {A3, A5, V3}; D2 needs {ML10, P2, V4}; D3 needs {D1, ML10}. WS05 (site) starts when A6 fixtures cover A3-A5 (end of M2).

## Milestones (each requires `scripts/check` green on main)
- M0 Contracts frozen (F0, F1): schema snapshots committed, CONTRACT_VERSION 1.0.0, ws branches created, tag `m0`. Parallel work starts only after this.
- M1 Foundations (F2-F4, ML1-ML3, A1-A3, A6, P1, V1-V2): roundtrip.md decides serving; manifests and leakage tests green; demo-mode analyses return linted EN/RU text; the voice pipeline translates wav files EN<->RU offline.
- M2 Baselines and flows (ML4-ML8, P2, P3, A4, A5, V3, V5): val table zero-shot vs detector; fare error published (or v0 flagged); happy_path fixtures cover photo → shared results → quote → booking → call + dentist context; a two-tab local EN<->RU call works, chat included; WS05 unblocked (the owner confirms the start and the Opus 5.5 switch then).
- M3 Tuned model (ML9, ML10): the lean recipe, re-run from a clean Kaggle session, lands within the recorded CI; MI300X spend <= $80; test evaluated once, gate applied, card rendered, weights pushed private.
- M4 Integration (V4, D1): the demo, cpu, cuda and mac paths start from a clean clone; voice p50/p95 measured; replay recorded; samples precomputed by the shipped model.
- M5 Release candidate (D2, D3, ML11 optional): README tables generated, licences complete, Space live (private), owner review. M6 (reserved): site + demo video.

## Test strategy and CI
- core: schema snapshots, linter corpora, exhaustive urgency combinations, hypothesis over renderings.
- ml: leakage and determinism, metric goldens, parse goldens, mock-transport adapters, gate timing. Training and pilots are `gpu`/`slow`.
- api/pricing: TestClient contract tests, OpenAPI and fixture drift, idempotency, rate limit, CORS, privacy (EXIF, logs, no disk, TTL), booking state machine, fare properties.
- voice: fakes-driven order and fallbacks, check unit tests, routing; the harness produces the numbers; `lk` e2e [slow].
- packaging/docs: compose config per profile, README vs results equality, licence completeness, docs lint, banned names.
- CI (GitHub Actions, ubuntu, Python 3.13 via uv, CPU torch): `check` = scripts/check; `contracts` = drift diff; `compose` = config -q per profile. No model downloads and no GPU in CI.

## Risk register (top 8)
| # | Risk | Mitigation |
|---|---|---|
| 1 | The tuned VLM fails the gate or loses to the detector | Per-finding gate, pre-registered ship rule, publish the loss honestly: the evaluation is the product |
| 2 | Tuned VLM not importable into Ollama (GGUF + mmproj) | ML1 runs first; three adapters behind VisionBackend; Ollama keeps the baselines |
| 3 | Leakage or label gaps (AKU split by image, no gingivitis negatives, partial labels) | Grouped split + leakage tests; coverage matrix in targets and metrics; specificity reported "not estimable" when no negatives exist |
| 4 | Free or credit compute fails (quota, disconnects, ROCm quirks, price) | 500-step checkpoints, lean yardstick, 50-step pilots, one TRL script for CUDA and ROCm, $80 cap + $20 reserve |
| 5 | Licence contamination (private set, AGPL, NC, Gemma Terms) | Manifest source allowlist, AST/grep tests, out-of-process GPL, dated licence tables, opt-in prompts |
| 6 | Call latency over 2.5 s, or RU errors on digits and negation | Harness before any claim, captions first, digit and glossary checks, captions_only mode, p50/p95 published |
| 7 | Claims drift (VLM note, site strings, README) | Templates first, runtime + CI lint, /meta serves the safety texts, docs profile on README and cards, Literal[True] disclosure |
| 8 | Contract churn breaks the parallel streams | M0 freeze, schema snapshots, lead-only change protocol, TS and fixture drift check |

## Demo scenario (3 min; recorded at M6 on the site, assets from D3)
- 0:00 Banner "research prototype, not a medical device"; EN/RU switch.
- 0:15 Sample photo: observations ("cannot rule out", not-assessed list), demo urgency hint tagged "illustrative", "see a dentist".
- 0:45 Model card: test table with CIs, zero-shot vs detector vs tuned, risk-coverage curve, one failure-gallery item.
- 1:05 Fictional clinic + origin city: quote with ESTIMATE/SAMPLE chips, ranges, basis line, measured fare error, a two-trip implant case.
- 1:35 Slot, hold, "Reserve (demo)": SIMULATED booking and tickets.
- 1:55 Call: AI disclosure acknowledged before joining. Patient RU, dentist EN: side-by-side captions, translated voice, a digit-check badge on "tooth 36", a glossary term, a latency readout, a translated chat message, and the dentist's panel showing the shared observations.
- 2:45 Captions-only toggle, replay mode, one-command start (`make demo`), licences page. End.
