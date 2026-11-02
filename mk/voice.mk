.PHONY: voice-bench

voice-bench: ## Measure the voice pipeline on this machine (needs Ollama, a Piper server on :5100, the stt and harness extras)
	uv run python -m dentabridge_voice.harness run --out docs/results --replay packages/voice/replays/demo --hardware "$(HARDWARE)"
