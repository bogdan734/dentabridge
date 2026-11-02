"""Fills the generated blocks of README.md from docs/results/*.json.

A block sits between `<!-- results:NAME:start -->` and `<!-- results:NAME:end -->` and is replaced
as a whole, so no measured number in the README is typed by hand. `python tools/render_readme.py`
rewrites README.md; `--check` exits 1 when it is out of date (tests/test_docs.py runs `render`).
"""

import argparse
import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
RESULTS = ROOT / "docs" / "results"
BLOCK = re.compile(
    r"(?P<start><!-- results:(?P<name>\w+):start -->\n)"
    r"(?P<body>.*?)"
    r"(?P<end><!-- results:(?P=name):end -->)",
    re.DOTALL,
)
STAGES = {"stt": "speech to text", "translate": "translation", "tts_first": "first audio"}


def _load(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def _seconds(ms: float) -> str:
    return f"{ms / 1000:.2f} s"


def _percent(value: float | None, digits: int = 0) -> str:
    return "—" if value is None else f"{value * 100:.{digits}f} %"


def _cell(value: float | None, interval: Sequence[float] | None = None) -> str:
    if value is None:
        return "—"
    if interval is None:
        return f"{value:.2f}"
    return f"{value:.2f} ({interval[0]:.2f}\u2013{interval[1]:.2f})"


def voice(results: Path) -> str:
    doc = _load(results / "voice.json")
    r = doc["report"]
    stages = ", ".join(
        f"{STAGES.get(name, name)} {ms:.0f} ms" for name, ms in r["stage_ms"].items()
    )
    models = "; ".join(f"{role}: {name}" for role, name in doc["models"].items())
    lines = [
        f"Live translation on the `{doc['profile']}` profile: {r['n']} scripted turns of an "
        f"English-speaking dentist and a Russian-speaking patient, recorded {doc['recorded_on']} "
        f"on {doc['hardware']} (load average {doc['load_average']:.1f} when it started). "
        f"The audio is {doc['audio']}.",
        "",
        "| | |",
        "|---|---|",
        f"| Speech end to first translated audio | median **{_seconds(r['p50_ms'])}**, "
        f"95th percentile {_seconds(r['p95_ms'])} |",
        f"| Mean time per stage | {stages} |",
        f"| Word error rate of the recognised speech | {_percent(r['wer'], 1)} |",
        f"| Translation quality (chrF++ against the reference) | {r['chrf']:.1f} |",
        f"| Turns whose glossary terms survived | {_percent(r['glossary_acc'])} |",
        f"| Turns whose numbers survived | {_percent(r['digit_acc'])} |",
        "",
        f"Models: {models}.",
        "",
    ]
    return "\n".join(lines)


def vision(results: Path) -> str:
    tuned = _load(results / "ml" / "run1-tuned.json")
    base = _load(results / "ml" / "run1-zeroshot.json")
    rows = []
    for t, z in zip(tuned["codes"], base["codes"], strict=True):
        if (t["code"], t["n_pos"], t["n_neg"]) != (z["code"], z["n_pos"], z["n_neg"]):
            raise ValueError("the tuned and the untuned result must be scored on the same images")
        rows.append(
            f"| {t['code']} | {t['n_pos']} / {t['n_neg']} "
            f"| {_cell(t['sens'], t['sens_ci'])} | {_cell(t['spec'], t['spec_ci'])} "
            f"| {_cell(z['sens'], z['sens_ci'])} | {_cell(z['spec'], z['spec_ci'])} |"
        )
    lines = [
        f"First training run, scored on the `{tuned['split']}` split ({tuned['n_images']} images). "
        f"Share of answers that were valid JSON: tuned {_percent(tuned['json_valid_rate'], 1)}, "
        f"untuned base model {_percent(base['json_valid_rate'], 1)}. Intervals are 95 % "
        "cluster-bootstrap; a dash means the split has no positives or no negatives for that "
        "finding.",
        "",
        "| Finding | Positives / negatives | Tuned: sensitivity | Tuned: specificity "
        "| Untuned: sensitivity | Untuned: specificity |",
        "|---|---|---|---|---|---|",
        *rows,
        "",
    ]
    return "\n".join(lines)


BLOCKS: dict[str, Callable[[Path], str]] = {"voice": voice, "vision": vision}


def render(text: str, results: Path = RESULTS) -> str:
    def fill(match: re.Match[str]) -> str:
        name = match["name"]
        if name not in BLOCKS:
            raise KeyError(f"README.md has a results block {name!r} that no renderer fills")
        return match["start"] + BLOCKS[name](results) + match["end"]

    return BLOCK.sub(fill, text)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 when README.md is stale")
    args = parser.parse_args(argv)
    current = README.read_text(encoding="utf-8")
    rendered = render(current)
    if args.check:
        if rendered != current:
            print("README.md is out of date: run `make docs`")
            return 1
        return 0
    README.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
