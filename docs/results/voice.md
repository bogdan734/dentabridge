# Voice translation, mac-native

Recorded 2026-11-02 on Apple M3 Max, 36 GB (load average 13.3 at the start). 50 scripted turns of an English-speaking dentist and a Russian-speaking patient; the audio is synthetic speech from Piper voices, not human recordings.

| models | |
|---|---|
| speech to text | faster-whisper small (int8, cpu) |
| translator | qwen3:4b-instruct through Ollama |
| voices | en_US-ljspeech-medium, ru_RU-denis-medium through Piper |

| speech end to first translated audio | |
|---|---|
| median | 1.34 s |
| 95th percentile | 1.54 s |

| stage (mean) | time |
|---|---|
| stt | 953 ms |
| translate | 320 ms |
| tts_first | 68 ms |

| quality | |
|---|---|
| word error rate of the recognised speech | 10.1 % |
| translation chrF++ against the reference | 64.5 |
| turns whose glossary terms survived | 98 % |
| turns whose numbers survived | 100 % |
| turns that produced nothing | 0 of 50 |
