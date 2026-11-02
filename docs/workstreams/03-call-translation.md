# WS03 Call with live voice-to-voice translation
Branch ws/03-call. Starts after tag `m0`. EN<->RU is verified; other pairs are config only and flagged "unverified".

## Owned paths
- packages/voice/ (src/dentabridge_voice/{pipeline,segment,checks,ports,worker,routing}.py, adapters/, harness/, glossary/en-ru.tsv, config/pairs.yaml, replays/, LICENCES.toml, tests/)
- docs/results/voice.{json,md}, mk/voice.mk

## Tasks (PLAN.md ids, in order)
1. V1 pipeline core + glossary/digit checks [L] (fakes only, no models)
2. V2 adapters: Parakeet-MLX, faster-whisper, Ollama translator (permissive Qwen3 default, TranslateGemma opt-in), Piper HTTP [L]
3. V3 LiveKit worker, one direction per speaker [L]
4. V4 latency/quality harness + replay recording [L]. It also runs the translator bake-off (Qwen3 sizes, TranslateGemma, Opus-MT, MADLAD) that fixes the default.
5. V5 chat translation over text streams [S], parallel with V3; same checks as V1

## Contracts consumed
- core: Translator port, GlossaryEntry, Translation, CallRoomMetadata, CaptionEvent, ReplayManifest, ReplayTrack, VERIFIED_PAIRS
- WS02: room name "db-<booking_id>", participant identity "<role>-<session_id>", room metadata = CallRoomMetadata JSON

## Contracts produced (the site and WS02 rely on these)
- Track names: `translated-<role>` (audio, published by the agent). The original mic tracks stay untouched; the listener's client ducks them to 0.15 while translated audio plays.
- Text stream topic `captions.v1`: one CaptionEvent JSON per update (partial then final), with source and translation side by side.
- When TTS fails, CaptionEvent.audio = "captions_only". Explicit, never silent.
- Text stream topics `chat.in.v1` (clients → agent, raw text) and `chat.v1` (agent → room, one ChatEvent per message, source always kept).
- packages/voice/replays/demo/manifest.json + audio files (ReplayManifest), served by WS02.
- docs/results/voice.json: per-stage ms, p50/p95 speech-end → first translated audio, WER, chrF++, glossary and digit accuracy, per profile (mac native, cpu, cuda if available).
- CLI `dentabridge-voice worker` (livekit-agents 1.8.x `start`) for WS04 compose.

## Runtime rules
- Pin livekit-agents 1.8.x (2.0 is a beta). No LiveKit turn-detector plugin (LiveKit Model License); use Silero VAD.
- The worker subscribes only to human participants' microphone tracks, never to the agent's or translated tracks (prevents echo loops).
- Segments are cut at pauses and capped at 10 s. Consecutive mode is the default.
- Piper runs as a separate HTTP container (GPL-3.0); it is never imported. Default voices need CC0 or permissive data. XTTS-v2, F5-TTS-RU and NLLB are opt-in rows only.
- Default translator: a permissive-licence model (Qwen3 instruct via Ollama; the V4 bake-off picks the size). TranslateGemma (Gemma Terms, gated) and NLLB (NC) are opt-in rows that the user downloads via `make models`, never redistributed. The model name is config behind the Translator port.
- Audio: a scripted dialogue (owner-read or synthetic) and FLEURS. Never patient audio, never recording live calls.

## Gate
- `scripts/check`
- `uv run pytest packages/voice -q -m "not slow"`
- Manual: `make voice-bench` (Mac native first, then the cpu profile), which writes docs/results/voice.json

## Definition of done
- Pipeline:
  - the fake-driven tests prove caption-before-audio, the captions_only fallback, glossary misses, digit swaps and the 10 s cap;
  - routing tests prove no self-subscription.
- Live:
  - the `lk` CLI e2e [slow] passes against a local LiveKit server;
  - a two-tab EN<->RU call works on the Mac native path (M2).
- Numbers and licences:
  - voice.json holds measured p50/p95 (the 1.3-2.5 s figure stays labelled "estimate" until then);
  - the replay is recorded;
  - LICENCES.toml has a dated row for every model and voice.

## Do NOT touch
- packages/core (file a change request below), packages/api, packages/pricing, packages/ml, contracts/, deploy/, space/, tools/, README.md, web/

## Contract change requests
- cr1 (V3, done in contract 1.1.0): `CaptionEvent.translation_status: "ok" | "failed"` (default "ok"). When the translator fails after the source caption went out, the worker closes that caption as failed (final, empty tgt_text, captions_only), so the listener is told instead of waiting on a partial caption.
