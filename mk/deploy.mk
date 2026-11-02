# The one-command path (WS04, deploy/compose.yaml). Every target writes .env first.
COMPOSE := docker compose -f deploy/compose.yaml --env-file .env
WORKER := uv run --package dentabridge-voice --extra worker dentabridge-voice worker

.PHONY: demo mac up-cpu up-cuda down models docs

.env:
	uv run --no-project python tools/make_env.py

demo: .env ## The API in demo mode on http://localhost:8000 (Docker; no models, no LiveKit)
	$(COMPOSE) --profile demo up --build --detach --wait

mac: .env ## Docker: api, LiveKit, Piper; native: Ollama (start it first) and the voice worker in the foreground
	DENTABRIDGE_OLLAMA_URL=http://host.docker.internal:11434 $(COMPOSE) --profile mac up --build --detach --wait
	set -a && . ./.env && set +a && $(WORKER)

up-cpu: .env ## Everything in Docker on the CPU: api, LiveKit, Piper, Ollama, voice worker
	$(COMPOSE) --profile cpu up --build --detach --wait

up-cuda: .env ## The cpu profile with Ollama and the voice worker on an NVIDIA GPU
	$(COMPOSE) --profile cuda up --build --detach --wait

down: .env ## Stop and remove the containers of every profile (the volumes stay)
	$(COMPOSE) --profile demo --profile mac --profile cpu --profile cuda down

models: .env ## List the models and voices the profiles use, with their licences; download each one you accept
	uv run --no-project python tools/models.py

docs: ## Regenerate the tables in README.md and docs/licences.md from docs/results and the LICENCES.toml files
	uv run --no-project python tools/render_readme.py
	uv run --no-project python tools/render_licences.py
