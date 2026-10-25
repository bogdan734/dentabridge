import pytest

from dentabridge_ml.train.config import CONFIG_DIR, load_config
from dentabridge_ml.train.record import make_record

CFG = load_config(CONFIG_DIR / "pilot-50.toml")
SUMMARY = {
    "train_runtime": 120.0,
    "train_samples_per_second": 3.33,
    "train_steps_per_second": 0.42,
    "train_loss": 2.9,
}


def _history() -> list[dict[str, float]]:
    return [{"loss": 3.1, "step": 5.0}, {"loss": 2.2, "step": 50.0}, {**SUMMARY, "step": 50.0}]


def test_record_takes_throughput_from_the_trainers_final_summary() -> None:
    record = make_record(CFG, 400, _history(), "abc1234", "AMD Instinct MI300X")
    assert (record.steps, record.train_seconds, record.samples_per_second) == (50, 120.0, 3.33)
    assert record.final_loss == 2.2, "the last logged step loss, not the run average"
    assert (record.rows, record.commit, record.hardware) == (400, "abc1234", "AMD Instinct MI300X")


def test_config_digest_is_stable_and_follows_every_field() -> None:
    first = make_record(CFG, 1, _history(), "c", "h").config_sha256
    assert first == make_record(CFG, 1, _history(), "c", "h").config_sha256
    other = CFG.model_copy(update={"lr": CFG.lr * 2})
    assert first != make_record(other, 1, _history(), "c", "h").config_sha256


def test_a_run_that_never_finished_has_no_record() -> None:
    with pytest.raises(ValueError, match="final summary"):
        make_record(CFG, 1, [{"loss": 3.1, "step": 5.0}], "c", "h")


def test_a_run_shorter_than_the_logging_interval_reports_its_mean_loss() -> None:
    record = make_record(CFG, 1, [{**SUMMARY, "step": 4.0}], "c", "h")
    assert record.final_loss == 2.9
