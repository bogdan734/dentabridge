"""`make models`: what the profiles download, each shown with the licence it comes under; nothing
is fetched without a yes. The licences are read from packages/voice/LICENCES.toml, the one place
they are recorded. The vision model is not listed yet: it is published after the release gate
(WS01 ML10).
"""

import argparse
import os
import shutil
import subprocess
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LICENCES = ROOT / "packages" / "voice" / "LICENCES.toml"
VOICES = ROOT / "var" / "tools" / "piper-voices"
COMPOSE = ("docker", "compose", "-f", str(ROOT / "deploy" / "compose.yaml"), "--env-file", ".env")


@dataclass(frozen=True)
class Download:
    component: str  # the start of its row in LICENCES.toml
    command: tuple[str, ...]


def ollama_pull(model: str) -> tuple[str, ...]:
    """Into the native Ollama when there is one (the mac profile), else into the cpu profile's."""
    if shutil.which("ollama"):
        return ("ollama", "pull", model)
    return (*COMPOSE, "--profile", "cpu", "exec", "ollama", "ollama", "pull", model)


def piper_voice(voice: str) -> tuple[str, ...]:
    """Piper's own downloader, run in its GPL container, writing into the folder compose mounts."""
    user = f"{os.getuid()}:{os.getgid()}"
    mount = f"{VOICES}:/download"
    downloader = ("python", "-m", "piper.download_voices", "--data-dir", "/download", voice)
    return (*COMPOSE, "run", "--rm", "--no-deps", "--user", user, "-v", mount, "piper", *downloader)


DOWNLOADS = (
    Download("Qwen3-4B-Instruct-2507", ollama_pull("qwen3:4b-instruct")),
    Download(
        "faster-whisper small",
        (
            *("uv", "run", "--package", "dentabridge-voice", "--extra", "stt", "python", "-c"),
            "from faster_whisper import WhisperModel; WhisperModel('small')",
        ),
    ),
    Download("Piper voice en_US-ljspeech-medium", piper_voice("en_US-ljspeech-medium")),
    Download("Piper voice ru_RU-denis-medium", piper_voice("ru_RU-denis-medium")),
)


def licence_row(component: str) -> dict[str, str]:
    document = tomllib.loads(LICENCES.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = [row for kind in document.values() for row in kind]
    found = [row for row in rows if row["component"].startswith(component)]
    if len(found) != 1:
        raise LookupError(f"{LICENCES.name} needs exactly one row for {component!r}")
    return found[0]


def main(argv: Sequence[str] | None = None, ask: Callable[[str], str] = input) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    VOICES.mkdir(parents=True, exist_ok=True)
    for download in DOWNLOADS:
        row = licence_row(download.component)
        print(f"\n{row['component']}\n  {row['role']}")
        print(f"  licence: {row['licence']}\n  from: {row['origin']}")
        if ask("  download it? [y/N] ").strip().lower() != "y":
            print("  skipped")
            continue
        subprocess.run(download.command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
