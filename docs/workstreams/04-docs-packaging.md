# WS04 Docs and packaging (one command path, README, licences, Space)
Branch ws/04-docs. D1 starts when A3, A5 and V3 are merged (M2-M4). D2 and D3 run at M5.

## Owned paths
- deploy/ (compose.yaml, docker/{api,voice,piper}.Dockerfile, livekit.yaml, release.json), mk/deploy.mk, .env.example, .dockerignore (root file, agreed with the lead)
- tools/ (make_env.py, models.py, render_readme.py, render_licences.py); their tests live in tests/test_deploy.py, which the gate runs
- README.md, docs/licences.md, docs/demo/ (script.md, shotlist.md)
- space/ (README.md Space card, Dockerfile)

## Tasks (PLAN.md ids, in order)
1. D1 compose profiles demo | cpu | cuda | mac + make targets [L]
2. D2 README with generated tables, licences page, docs lint [S]
3. D3 HF Space (samples only, uploads off, private) + demo kit [S]

## Contracts consumed
- CLIs: `dentabridge-api serve|seed|export` (WS02), `dentabridge-voice worker` (WS03), `make data | models` (WS01)
- Settings env vars, prefix DENTABRIDGE_: MODE, UPLOADS, CORS_ORIGINS, DB_URL, VISION_BACKEND, LIVEKIT_URL/KEY/SECRET, REPLAYS_DIR, XRAY_BACKEND
- Results JSON: docs/results/ml/*.json, docs/results/fares.json, docs/results/voice.json
- Licence tables: packages/{ml,pricing,voice}/LICENCES.toml
- `dentabridge-lint --profile docs`

## Contracts produced
- `make demo` (API demo mode only, no models; the site's backend), `make up-cpu`, `make up-cuda`, `make mac` (Docker: api, livekit, piper; native: Ollama, voice worker), `make down`, `make models`
- `make models` lists each gated or opt-in item with its licence and asks before downloading it (TranslateGemma: Gemma Terms; NC voices; DENTEX)
- README sections: badges (CI, licence, Python, test count), research-prototype banner, quickstart (one command), measured results tables (rendered between markers), limitations, licences, data attribution, author Bohdan Havryliuk, MIT
- Space: the API in demo mode with UPLOADS=off, precomputed samples, the live estimator and replay; the site's static export is mounted at "/" once WS05 delivers

## Gate
- `scripts/check`
- `uv run pytest tools -q`
- `docker compose -f deploy/compose.yaml --profile {demo,cpu,cuda,mac} config -q`
- Manual: `make demo`, then `curl -fsS localhost:8000/v1/health`

## Definition of done
- Clean clone → `make demo` → health 200 within 60 s, with no network beyond the image pulls.
- `make up-cpu` and `make mac` reach a working two-tab call. The cuda profile validates, and is run if a GPU host is available.
- README:
  - every number comes from results JSON (test proves equality);
  - limitations name the data, n, the camera shift and the US-domestic-only fare calibration;
  - it says "not a medical device".
- docs/licences.md has a row for every model, dataset and voice id found in configs, each with a verified-on date.
- No secrets committed: LiveKit dev keys are generated into .env by make.
- The Space is private, uploads are off, and every sample has attribution.

## Do NOT touch
- packages/* (ask the owning workstream for CLI or setting changes), contracts/, seed/, docs/results/ (read only), docs/model-cards/ (WS01), web/
- Never write a number into README.md by hand.

## Notes from D1 (2026-11-02)
- `make demo|mac|up-cpu|up-cuda|down|models`; every target first writes .env with fresh LiveKit and call keys (tools/make_env.py, mode 600, never overwritten). Every port is published on 127.0.0.1 only.
- The API runs live mode in mac, cpu and cuda (LiveKit checks token times against the real clock, so the frozen demo clock cannot open a room). `deploy/release.json` is what the deployed weights may claim: `{"assessed": []}` until the release gate (ML9) replaces it.
- LiveKit advertises 127.0.0.1 (browser, native worker) and its container address (the dockerised worker of the cpu profile).
- Verified on the Mac: `make demo` (health 200), `make mac` with a call opened through the API and interpreted (tests/test_deploy.py, slow). The cpu and cuda profiles validate only: the ollama/ollama image (2.8 GB) was not pulled, and there is no GPU here.

## Contract change requests
- (none yet)
