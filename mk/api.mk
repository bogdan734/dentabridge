.PHONY: contracts

contracts: ## Regenerate contracts/ (OpenAPI, recorded fixtures, TypeScript types and client); a diff fails the tests
	uv run python -m dentabridge_api.export contracts
	cd contracts/ts && { [ -d node_modules ] || npm ci --no-audit --no-fund; } && npm run --silent generate && npm run --silent check
