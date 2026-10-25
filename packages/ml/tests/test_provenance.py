import pytest

from dentabridge_ml.provenance import git_commit


def test_commit_can_be_given_for_trees_copied_without_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DENTABRIDGE_COMMIT", "abc1234")
    assert git_commit() == "abc1234"


def test_commit_comes_from_git_otherwise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DENTABRIDGE_COMMIT", raising=False)
    assert git_commit() not in ("", "unknown")
