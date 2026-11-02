.PHONY: contracts

contracts: ## Regenerate contracts/ (OpenAPI document and recorded fixtures); a diff fails the tests
	uv run python -m dentabridge_api.export contracts
