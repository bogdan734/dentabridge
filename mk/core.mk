.PHONY: help check test fmt

help: ## List the documented targets
	@grep -hE '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "%-20s %s\n", $$1, $$2}'

check: ## Run the full gate (scripts/check)
	scripts/check

test: ## Run the fast tests (no slow, gpu or network)
	uv run pytest -q -m "not slow and not gpu and not network"

fmt: ## Format code and apply safe lint fixes
	uv run ruff format .
	uv run ruff check --fix .
