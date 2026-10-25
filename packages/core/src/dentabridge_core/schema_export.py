"""JSON Schema snapshots of every contract. The lead regenerates them on a contract change.

uv run python -m dentabridge_core.schema_export
"""

import argparse
import json
from collections.abc import Iterator, Sequence
from pathlib import Path

from pydantic import BaseModel

import dentabridge_core  # noqa: F401  (importing the package registers every contract class)
from dentabridge_core._base import Contract

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"


def _subclasses(cls: type[Contract]) -> Iterator[type[Contract]]:
    for sub in cls.__subclasses__():
        yield sub
        yield from _subclasses(sub)


# Only our own contracts: other packages may subclass Contract and must not enter these snapshots.
ALL_CONTRACTS: tuple[type[BaseModel], ...] = tuple(
    sorted(
        (m for m in _subclasses(Contract) if m.__module__.startswith("dentabridge_core.")),
        key=lambda m: m.__name__,
    )
)


def render_schema(model: type[BaseModel]) -> str:
    return json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"


def write_snapshots(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    expected = {f"{m.__name__}.json": render_schema(m) for m in ALL_CONTRACTS}
    for stale in directory.glob("*.json"):
        if stale.name not in expected:
            stale.unlink()
    for name, text in expected.items():
        (directory / name).write_text(text)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rewrite the contract schema snapshots.")
    parser.add_argument("directory", nargs="?", type=Path, default=SCHEMAS_DIR)
    write_snapshots(parser.parse_args(argv).directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
