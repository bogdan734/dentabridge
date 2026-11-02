.PHONY: voice-bench voice-worker voice-e2e

voice-bench: ## Measure the voice pipeline on this machine (needs Ollama, a Piper server on :5100, the stt and harness extras)
	uv run python -m dentabridge_voice.harness run --out docs/results --replay packages/voice/replays/demo --hardware "$(HARDWARE)"

voice-worker: ## Run the call interpreter (worker extra; DENTABRIDGE_LIVEKIT_URL, _KEY and _SECRET set; Ollama and Piper up)
	uv run dentabridge-voice worker

voice-e2e: ## The worker in a real LiveKit room (Docker): scripted models, plus local models when Ollama and Piper are up
	uv run pytest packages/voice/tests/test_worker_e2e.py -m slow -q
