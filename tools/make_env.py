"""Write .env for the compose profiles: .env.example with its empty secrets filled in.

make runs it before any profile starts. An existing .env is never touched: the running containers
and the call tokens already handed out use its keys. Delete it to get new ones.
"""

import argparse
import secrets
from collections.abc import Sequence
from pathlib import Path

SECRETS = ("DENTABRIDGE_LIVEKIT_KEY", "DENTABRIDGE_LIVEKIT_SECRET", "DENTABRIDGE_CALL_SECRET")


def fill(example: str) -> str:
    lines = []
    for line in example.splitlines():
        name, sep, value = line.partition("=")
        if sep and name in SECRETS and not value:
            # LiveKit wants a short key id and a secret of at least 32 characters
            value = (
                f"API{secrets.token_hex(8)}" if name.endswith("_KEY") else secrets.token_urlsafe(36)
            )
        lines.append(name + sep + value)
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=Path, default=Path(".env.example"))
    parser.add_argument("--out", type=Path, default=Path(".env"))
    args = parser.parse_args(argv)
    if args.out.exists():
        print(f"{args.out} exists, keeping its keys (delete it to get new ones)")
        return 0
    args.out.touch(mode=0o600, exist_ok=False)  # never readable by others, not even for a moment
    args.out.write_text(fill(args.example.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"wrote {args.out} with new LiveKit and call keys")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
