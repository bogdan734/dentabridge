from pathlib import Path

from dentabridge_ml.pilot.__main__ import modelfile


def test_modelfile_lists_the_language_model_then_the_projector(tmp_path: Path) -> None:
    lines = modelfile(tmp_path / "text.gguf", tmp_path / "mmproj.gguf").splitlines()
    assert lines[:2] == [
        f"FROM {tmp_path.resolve()}/text.gguf",
        f"FROM {tmp_path.resolve()}/mmproj.gguf",
    ]
    assert "RENDERER qwen3-vl-instruct" in lines
    assert "PARSER qwen3-vl-instruct" in lines
    assert not any("thinking" in line for line in lines), "the tuned model does not think aloud"
