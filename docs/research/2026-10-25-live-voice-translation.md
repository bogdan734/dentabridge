# Local open-source live voice translation for a 2-party dental call (EN<->RU first, other pairs via config)
Rev 2, checked 2026-10-25. Owner constraint: 100% open-source, free, local on Apple M3 Max 36 GB (native) and docker compose on CPU or CUDA Linux. [P] primary, [S] secondary, UNVERIFIED = not confirmed (memory = recalled, not re-checked). RTF = compute time / audio time. Used 12 searches, 9 fetches (mlx-audio fetch failed).

## 1) Question and constraints
Which fully open, $0 cascade (or open speech-to-speech model) lets a patient and a dentist who speak different languages hear each other in translated VOICE (not only subtitles) in a WebRTC call, run on an M3 Max (MLX/Metal, native) and in docker (CPU or CUDA; Docker Desktop on macOS has no Metal GPU: memory, UNVERIFIED), stay safe for medical terms, and be reproducible from a public repo with no paid API and no real patient data? No end-to-end latency measurement of any open stack on Apple Silicon was found; every end-to-end figure below is my sum of component numbers (estimate).

## 2) Options table
| Option | What it is | Fit | Cost and risk | Source |
|---|---|---|---|---|
| VAD: Silero VAD + Smart Turn v3.2 | Speech/silence gate plus a semantic end-of-turn model | Default endpointing: VAD 0.5 ms per 32 ms chunk on 1 CPU thread (55-62x real time); Smart Turn ~171 ms per decision (4 threads) | MIT / BSD-2; Smart Turn's default threshold over-triggered on the authors' material, needs tuning | [9] |
| STT: Parakeet-tdt-0.6b-v3 (parakeet-mlx) | FastConformer-TDT, 25 EU langs incl. ru, uk, CC-BY-4.0 (memory, UNVERIFIED) | Default EN/RU STT on Mac: 52.7x real time on M1 Max [S], so ~0.15-0.3 s for an 8 s utterance on M3 Max (my inference) | Streaming mode and MLX port licence UNVERIFIED; no independent RU WER found | [11] |
| STT: Whisper large-v3-turbo (MLX / whisper.cpp / faster-whisper) | OpenAI Whisper, MIT (memory); any language | MLX ~55x real time on M4 Max (0.86 s per ~47 s); mlx-whisper ~2x whisper.cpp on this model (2026-01); est. 0.3-0.6 s per short utterance on M3 Max (30 s window, my inference) | faster-whisper has no Metal (CPU on Mac, CUDA in docker; memory); not natively streaming (chunk + agreement policy); hallucinates on silence, gate with VAD (memory) | [11][12] |
| STT: GigaAM-v3, T-one (RU-only) | Russian ASR, MIT / Apache-2.0 | T-one streams 300 ms chunks in ~100-123 ms on CPU; GigaAM vendor WER 8.4% | No independent WER in the bench (only disagreement); T-one emits no digits or Latin letters (bad for doses, tooth numbers). Moonshine: English-first, RU not known (memory) | [9] |
| MT: TranslateGemma 4B / 12B / 27B (Jan 2026) | Gemma-3-based, 55 langs; 4B ~ 12B baseline, 12B > Gemma-3 27B on MetricX (Google-reported) | Default MT: llama.cpp 7B Q4_0 = 56.6 tok/s gen, 568 tok/s prompt on M3 Max 30-core (40-core: 66.3/760); so 4B Q4 ~55-80 tok/s, ~0.6-1.2 s per 20-word sentence; 12B ~25 tok/s, CUDA tier (my scaling by weight size) | RU/UK coverage and licence text not confirmed (memory: WMT24++ has ru, uk); only a third-party GGUF found, no Ollama listing; 36 GB M3 Max = 30-core variant (memory) | [10][14] |
| MT: NLLB-200, MADLAD-400, Opus-MT en<->ru; small Qwen/Gemma via Ollama | Classic encoder-decoder MT (CTranslate2) or general LLM | CPU docker tier: Opus-MT / NLLB-600M est. 0.1-0.3 s per sentence (UNVERIFIED); an LLM accepts a glossary in the prompt | Licences from memory: NLLB CC-BY-NC, MADLAD Apache-2.0, Opus-MT CC-BY-4.0; classic MT has weaker fluency and no glossary hook | - |
| TTS: Piper RU (denis, dmitri, irina, ruslan) | ONNX VITS voices, CPU | Default fast tier: RTF 0.20-0.65 on a 2014 Xeon without AVX2, so well under real time on M3 Max (inference); even pace, no clicks | Runtime GPL-3.0 (original archived Oct 2025 [S]); voice data CC0 (denis, dmitri), CC BY-NC-SA (ruslan), irina unknown; flat prosody; needs number/date normalisation | [9] |
| TTS: ESpeech-TTS-1 RL-V2, F5-TTS RU, XTTS-v2 | Cloning-capable, GPU-class | Quality tier: ESpeech RTF 0.17-0.36, first chunk 1.1-1.5 s, automatic stress, 6/6 lines verbatim; F5-RU WER 3.2%, first audio 1.1-3.1 s; XTTS-v2 RTF ~0.5, first audio ~0.5 s, WER 6.4% | F5-RU CC BY-NC, XTTS CPML (non-commercial); ESpeech base licence unverified; hardware per row not stated, nothing measured on Apple Silicon; XTTS has no UK (memory) | [9] |
| TTS: Kokoro, Qwen3-TTS, Chatterbox-MT, Silero | Small or newer open TTS | Kokoro 82M Apache-2.0: no official RU, stress sometimes wrong (RTF 0.2 on a 5950X CPU). Qwen3-TTS 1.7B (RU yes, UK no): README claims 97 ms first packet, bench measured RTF 2.8-3.3 on a 3090 Ti. Chatterbox-MT MIT: CPU RTF 22-24, digits garbled | Not default. Silero RU: not in the bench, licence UNVERIFIED. Mac/MLX ports UNVERIFIED (mlx-audio fetch failed) | [8][9] |
| S2S: Seamless Streaming, Hibiki(-Zero), others | Open end-to-end speech translation | Seamless: weights CC-BY-NC 4.0, README last dated 2023-12; RU/UK speech out and Apple Silicon run UNVERIFIED. Hibiki-Zero: FR/ES/PT/DE->EN only, CC BY-NC-SA 4.0. Nothing found for EN<->RU | Verdict: no usable open S2S; newer research (e.g. arXiv 2607.19810) not checked | [7][13] |
| Room: LiveKit server + Agents | Self-hosted SFU (Apache-2.0, memory); translator runs as a participant | Recipe pipeline_translator and a Cerebras cookbook show the cascaded pattern; OpenAI's guide ducks the source tile to 0.15 | Pipecat (BSD-2, memory) is an alternative; Daily needs its cloud, excluded; Jitsi not checked | [1][6] |

Paid quality ceiling, NOT used (reference only):
- OpenAI gpt-realtime-translate: $0.034/min per session, RU out yes, UK out no, no glossary or prompt [1][2].
- Google gemini-3.5-live-translate-preview: lists ru and uk, no instructions or text input, price not found [3].
- Palabra: $0.04/min, zero-shot voice cloning, RU and UK, "sub-second" claim with no number [4].
- Soniox: STT + translation $0.12/h, TTS ~$0.70/h of speech [5].

## 3) Recommendation
ADAPT: reuse LiveKit server/Agents and its translator recipe, build local plugins, the glossary/safety layer and the eval harness. Default on Mac (native): Silero VAD -> Parakeet-v3 (MLX) -> TranslateGemma-4B Q4 (llama.cpp or Ollama, glossary in prompt, clause streaming) -> Piper RU/EN. Docker cpu: faster-whisper int8 -> Opus-MT/NLLB -> Piper; cuda: Whisper-turbo -> TranslateGemma-12B -> ESpeech/XTTS (non-commercial, opt-in). EN<->RU is the pair to verify first; other pairs are model and voice names in config.
Honest latency (my sums, not measured), speaker stops to first translated audio: hold 0.3-0.5 s + STT 0.15-0.5 s + MT 0.6-1.2 s (first clause ~0.3 s if streamed) + TTS 0.1-0.3 s = about 1.3-2.5 s on M3 Max; docker CPU on a Mac about 3-6 s (guess); mid-sentence simultaneous mode adds more.
User perceives: captions of both texts within ~0.5 s, then a neutral synthetic voice 1.5-2.5 s after the pause, original ducked: a consecutive-interpreter or walkie-talkie rhythm, not live dubbing. Russian caveats: flat Piper prosody, stress and numerals, medical loanwords mispronounced, MT may drop negation or doses.

## 4) Improvements worth trying
- Routing (my design, API details UNVERIFIED): one worker per speaker direction subscribes only to that speaker's raw mic track (never the mixed room: echo loop), publishes a translated audio track plus a text stream; listener's client ducks the original to ~0.15 [1]; headphones + browser echo cancellation; cut segments at clause boundaries (max ~10 s); "repeat" button.
- Safety: always show source and translation side by side; confirm-back for allergies, drugs, pregnancy, left/right; ~200-term glossary (inject only matching terms, then check the target terms appear); digit check (numbers in source must appear in target); captions-only fallback; "not a medical interpreter" notice.
- Latency levers: clause-level MT->TTS streaming, pre-warmed models, Parakeet over Whisper for EN/RU, captions first so the wait feels shorter.
- Evaluation, all open (tools from memory, UNVERIFIED): replay 50 scripted dental utterances + ~100 FLEURS EN/RU clips; log speech-end -> first translated audio (p50/p95); jiwer WER, sacreBLEU/chrF++/COMET, ASR-BLEU of the output audio, glossary and digit accuracy, 2 bilingual raters (adequacy 1-5, safe y/n); bake off Parakeet vs Whisper-turbo vs GigaAM on RU, Piper vs ESpeech by round-trip WER and rater score.
- $0 demo: (1) replay mode: recorded session (audio, event JSON, timings) as a static page; (2) `docker compose --profile cpu|cuda up`, plus a native `make mac` for MLX; two tabs, headphones; (3) idea, UNVERIFIED: in-browser Whisper + Opus-MT + Piper (transformers.js / onnxruntime-web) for zero-install try-out.
- Licences: permissive defaults (Piper voices with CC0 data, MIT/Apache models); XTTS-v2, F5-TTS-RU, NLLB, Seamless only as opt-in downloads with a licences table; cloning a patient's voice needs consent.

## 5) Open questions for the owner
1. Repo licence: is a GPL-3.0 Piper runtime (separate process) and opt-in non-commercial models (XTTS-v2, F5-TTS-RU, NLLB) acceptable?
2. Which extra pairs get configs first (UK, TR, PL, ES)? Parakeet-v3 lacks TR (memory); Piper voices per language unchecked.
3. Is the consecutive mode with ~1.5-2.5 s lag acceptable as default, versus a rougher simultaneous mode?
4. Mac native (MLX/Metal) plus docker for CPU/CUDA only: OK, given Docker Desktop on macOS cannot reach Metal (memory)?
5. Default TTS: flat but fast Piper, or a heavier GPU-only voice (ESpeech) as opt-in?

## 6) Sources (all checked 2026-10-25)
[1] https://developers.openai.com/cookbook/examples/voice_solutions/realtime_translation_guide - P, fetched; undated (after 2026-05-07 launch)
[2] https://developers.openai.com/api/docs/pricing - P, search summary only; undated
[3] https://ai.google.dev/gemini-api/docs/live-api/live-translate - P, fetched; page updated 2026-09-16
[4] https://palabra.ai/pricing - P, fetched; undated
[5] https://soniox.com/pricing - P, fetched; undated
[6] https://docs.livekit.io/recipes/pipeline_translator - P, search summary only; undated. https://inference-docs.cerebras.ai/cookbook/agents/realtime-voice-translation - S (vendor cookbook), undated
[7] https://github.com/facebookresearch/seamless_communication - P, fetched; last dated README entry 2023-12-18 (older)
[8] https://github.com/QwenLM/Qwen3-TTS - P, fetched; released 2026-01-22
[9] https://github.com/meshoklv/ru-speech-bench - P (third-party benchmark, own measurements), fetched; measured 2026-09 on Xeon E5-2695 v2 (CPU only), Ryzen 5950X, RTX 3090 Ti; no Apple Silicon
[10] https://github.com/ggml-org/llama.cpp/discussions/4167 - P (community-measured table), fetched; instructions updated 2026-08-25, last comment 2026-06-26
[11] https://vllm-mlx.is-a.dev/benchmarks/audio/ - S (third-party docs), search summary only; undated; M4 Max and M1 Max runs
[12] https://notes.billmill.org/dev_blog/2026/01/updated_my_mlx_whisper_vs._whisper.cpp_benchmark.html - S (blog), search summary only; 2026-01; machine unnamed
[13] https://huggingface.co/kyutai/hibiki-zero-3b-pytorch-bf16 - P, search summary only; 2026 (ICML 2026 oral)
[14] https://blog.google/innovation-and-ai/technology/developers-tools/translategemma/ - P, search summary only; 2026-01
