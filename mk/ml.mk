.PHONY: data data-lock data-verify splits eval-val eval-test kaggle-bundle

# DATA_SOURCES limits the run, e.g. `make data DATA_SOURCES=aku_caries`; empty means every published source.
DATA_SOURCES ?=

data: ## Download and unpack the published datasets (network, large; see docs/research)
	uv run python -m dentabridge_ml.data fetch $(DATA_SOURCES)
	uv run python -m dentabridge_ml.data extract $(DATA_SOURCES)

data-lock: ## Write packages/ml/data.lock.json from the unpacked data (maintainer, once)
	uv run python -m dentabridge_ml.data lock $(DATA_SOURCES)

data-verify: ## Check the unpacked data against packages/ml/data.lock.json
	uv run python -m dentabridge_ml.data verify

splits: ## Rebuild packages/ml/manifests from the unpacked data (seed 0); commit the result
	uv run python -m dentabridge_ml.splits.build --seed 0

# One run = one backend on one split: make eval-val BACKEND=ollama:qwen3-vl:4b RUN=zero-shot-4b
BACKEND ?= ollama:qwen3-vl:4b
IMAGE_PX ?= 448
RUN ?= zero-shot-4b

eval-val: ## Score the validation split with BACKEND; results land in docs/results/ml/runs/RUN
	uv run python -m dentabridge_ml.eval run --backend $(BACKEND) --split val --image-max-px $(IMAGE_PX) --out-dir docs/results/ml/runs/$(RUN)

eval-test: ## Score the test split ONCE per chosen model (logged in docs/results/ml/test-access.log)
	uv run python -m dentabridge_ml.eval run --backend $(BACKEND) --split test --final --access-log docs/results/ml/test-access.log --image-max-px $(IMAGE_PX) --out-dir docs/results/ml/runs/$(RUN)

kaggle-bundle: ## Pack HEAD into var/kaggle/dentabridge-code.tar.gz (upload it as a private Kaggle dataset)
	scripts/kaggle/bundle.sh
