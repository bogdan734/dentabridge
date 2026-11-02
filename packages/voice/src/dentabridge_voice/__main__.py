"""`dentabridge-voice worker`: the call interpreter for every call room on one LiveKit server.

Needs the `worker` extra, a LiveKit server, Ollama with the translator and a Piper server with
the two voices (without Piper pass --no-voice: every call then gets captions only). The server's
key and secret come from DENTABRIDGE_LIVEKIT_KEY and DENTABRIDGE_LIVEKIT_SECRET, as for the API.
"""

import argparse
import asyncio
import logging
import os
from collections.abc import Sequence

from dentabridge_voice.adapters.faster_whisper import FasterWhisperSTT
from dentabridge_voice.adapters.ollama_translator import OllamaTranslator
from dentabridge_voice.adapters.piper_http import VOICES, PiperHttpTTS
from dentabridge_voice.rooms import LiveKitServer, watch_rooms
from dentabridge_voice.worker import Services, UtcClock


def _worker(args: argparse.Namespace) -> None:
    server = LiveKitServer(
        args.livekit_url,
        os.environ["DENTABRIDGE_LIVEKIT_KEY"],
        os.environ["DENTABRIDGE_LIVEKIT_SECRET"],
    )
    services = Services(
        stt=FasterWhisperSTT(args.whisper, args.device, args.compute_type),
        translator=OllamaTranslator(args.ollama, args.translator),
        tts=None if args.no_voice else PiperHttpTTS(args.piper, VOICES),
        wall=UtcClock(),
    )
    asyncio.run(watch_rooms(server, services))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dentabridge-voice", description=__doc__)
    steps = parser.add_subparsers(dest="step", required=True)
    worker = steps.add_parser("worker", help="interpret every call room on the LiveKit server")
    worker.add_argument(
        "--livekit-url", default=os.environ.get("DENTABRIDGE_LIVEKIT_URL", "ws://127.0.0.1:7880")
    )
    worker.add_argument("--ollama", default="http://localhost:11434")
    worker.add_argument("--translator", default="qwen3:4b-instruct")
    worker.add_argument("--piper", default="http://127.0.0.1:5100")
    worker.add_argument("--no-voice", action="store_true", help="captions only, no Piper")
    worker.add_argument("--whisper", default="small")
    worker.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    worker.add_argument("--compute-type", default="int8")
    worker.set_defaults(func=_worker)
    args = parser.parse_args(argv)
    keys = ("DENTABRIDGE_LIVEKIT_KEY", "DENTABRIDGE_LIVEKIT_SECRET")
    missing = [k for k in keys if k not in os.environ]
    if missing:
        parser.error(f"set {' and '.join(missing)} (the LiveKit server's key and secret)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
