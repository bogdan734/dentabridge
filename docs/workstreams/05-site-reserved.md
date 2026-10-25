# WS05 Site (RESERVED, built later by a different model)
Not in this plan's scope. This file fixes only what the site may rely on and must respect, so no backend work is needed later.
Start condition: end of M2 (contracts/ fixtures cover analyses, quotes, booking and call). Before starting, the lead asks the owner to switch the session model to Opus 5.5 (`/model`) and to confirm the start.

## Owned paths
- web/ only (Next.js, i18n EN+RU). Everything else is read-only.

## Contracts consumed (the only integration surface)
- contracts/openapi.json and contracts/ts (generated types + openapi-fetch client). Never hand-write API types.
- contracts/fixtures/{happy_path,errors}/: deterministic demo-mode responses for mocks and tests.
- `make demo`: the local API in demo mode at http://localhost:8000/v1. CORS allows http://localhost:3000 by default, without credentials.
- LiveKit: tokens from POST /v1/calls/{id}/tokens; audio track `translated-<role>`; text stream topic `captions.v1` (CaptionEvent).

## Must
- Safety texts: banner, disclaimers, AI disclosure, estimate footer, chips and consent text come from GET /v1/meta?locale=…, never from site copy.
- Lint: the site's own EN/RU message files must pass `uv run dentabridge-lint --profile patient --locale en|ru web/messages/*.json`.
- Analysis: show `patient_text` as returned, the not-assessed list, and the urgency hint with its review_status tag. Never rephrase the findings.
- Uploads:
  - ask for explicit consent (POST /v1/consents) before upload;
  - hide upload when Meta.uploads_enabled is false;
  - send an Idempotency-Key on every POST marked Idem in PLAN.md.
- Prices: every money figure shows its chip (ESTIMATE, SAMPLE, SIMULATED), ranges and basis line; the button reads "Reserve (demo)".
- Call:
  - show the AI disclosure and get acknowledgement before requesting a token;
  - show source and translation side by side;
  - duck the original remote audio to about 0.15 while translated audio plays;
  - offer a captions-only toggle, the replay page (GET /v1/replays/demo) and a "pair unverified" notice when pair_status is unverified;
  - chat: send raw text on `chat.in.v1`, render each ChatEvent from `chat.v1` with the source beside the translation;
  - dentist panel: GET /v1/calls/{id}/context with the dentist call token (shared observations only, never an image); ask for the separate share_with_clinic consent before booking.
- Errors: map ErrorResponse.code / message_key to localized text and honour Retry-After on 429.

## Must not
- No analytics, accounts, cookies or third-party scripts. Never store images client-side beyond the upload request.
- No verdict wording ("you have …"), and no claims of diagnosis, clearance or compliance.

## Gate (to be added to scripts/check when web/ exists)
- `npm run lint && npm run typecheck && npm test` in web/, plus the dentabridge-lint command above.

## Definition of done (later, M6)
- Every flow runs against `make demo` and against the fixtures. EN/RU parity.
- The demo video follows docs/demo/script.md (3-minute scenario in PLAN.md).
