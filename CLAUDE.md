# dentabridge

Global rules apply (`~/.claude/CLAUDE.md`: routing and code style). This file holds only what is specific to this repo.

## Commands
- Verify everything: `scripts/check` (the Stop hook runs it when files changed; red means not done)
- Syntax check of one file: `scripts/check-fast <file>`
- Setup: `uv sync` (Python 3.13, one uv.lock)
- Run: `make help` lists every target; `check`, `test`, `fmt` live in mk/core.mk, each other mk/*.mk belongs to its workstream

## Layout
Contracts, the path-ownership table and the task list: `docs/PLAN.md`. Per-workstream scope: `docs/workstreams/NN-*.md`.
- `packages/{core,ml,api,pricing,voice}/`: uv workspace members (`src/dentabridge_<pkg>/`); core is imported by all, nothing imports api or voice
- `scripts/`: the gate (`check`, `check-fast`). `mk/`: make targets, one file per workstream. `tests/`: repo-level tests. `.github/workflows/`: CI
- `docs/`: plan, workstreams, research. `web/`: reserved for WS05 (README only)
- Owners: WS00 lead = root files, scripts, core; WS01 = ml; WS02 = api, pricing, seed, contracts; WS03 = voice; WS04 = deploy, space, tools, README; WS05 = web. Stay inside your own paths.

## Conventions
- Contract changes only through the lead (see docs/workstreams/00-foundation.md).
- No AGPL/GPL imports in-process; GPL tools such as Piper run as separate services over HTTP.
- Datasets are never committed (`/data/`, `private/`, `var/` are gitignored).
- Private project names never appear in the repo.
- Every README number comes from `docs/results/*.json`, rendered by repo scripts.
