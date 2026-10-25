.PHONY: data data-lock data-verify

# DATA_SOURCES limits the run, e.g. `make data DATA_SOURCES=aku_caries`; empty means every published source.
DATA_SOURCES ?=

data: ## Download and unpack the published datasets (network, large; see docs/research)
	uv run python -m dentabridge_ml.data fetch $(DATA_SOURCES)
	uv run python -m dentabridge_ml.data extract $(DATA_SOURCES)

data-lock: ## Write packages/ml/data.lock.json from the unpacked data (maintainer, once)
	uv run python -m dentabridge_ml.data lock $(DATA_SOURCES)

data-verify: ## Check the unpacked data against packages/ml/data.lock.json
	uv run python -m dentabridge_ml.data verify
