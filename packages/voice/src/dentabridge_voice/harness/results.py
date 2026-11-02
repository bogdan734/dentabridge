"""docs/results/voice.json and the table the README and the model card read from it."""

from datetime import date

from pydantic import BaseModel, ConfigDict

from dentabridge_voice.harness.run import HarnessReport


class VoiceResults(BaseModel):
    model_config = ConfigDict(frozen=True)

    profile: str
    recorded_on: date
    hardware: str
    load_average: float  # what else the machine was doing: latency is only as quiet as the host
    audio: str
    models: dict[str, str]
    report: HarnessReport


def _seconds(ms: float | None) -> str:
    return "—" if ms is None else f"{ms / 1000:.2f} s"


def _percent(value: float | None, digits: int = 0) -> str:
    return "—" if value is None else f"{value * 100:.{digits}f} %"


def render_markdown(results: VoiceResults) -> str:
    r = results.report
    lines = [
        f"# Voice translation, {results.profile}",
        "",
        f"Recorded {results.recorded_on.isoformat()} on {results.hardware} "
        f"(load average {results.load_average:.1f} at the start). {r.n} scripted turns of an "
        f"English-speaking dentist and a Russian-speaking patient; the audio is {results.audio}.",
        "",
        "| models | |",
        "|---|---|",
        *[f"| {role} | {name} |" for role, name in results.models.items()],
        "",
        "| speech end to first translated audio | |",
        "|---|---|",
        f"| median | {_seconds(r.p50_ms)} |",
        f"| 95th percentile | {_seconds(r.p95_ms)} |",
        "",
        "| stage (mean) | time |",
        "|---|---|",
        *[f"| {stage} | {ms:.0f} ms |" for stage, ms in r.stage_ms.items()],
        "",
        "| quality | |",
        "|---|---|",
        f"| word error rate of the recognised speech | {_percent(r.wer, 1)} |",
        f"| translation chrF++ against the reference | {r.chrf:.1f} |",
        f"| turns whose glossary terms survived | {_percent(r.glossary_acc)} |",
        f"| turns whose numbers survived | {_percent(r.digit_acc)} |",
        f"| turns that produced nothing | {r.n_silent} of {r.n} |",
    ]
    return "\n".join(lines) + "\n"
