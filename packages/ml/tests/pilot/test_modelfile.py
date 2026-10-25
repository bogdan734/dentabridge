from pathlib import Path

from dentabridge_ml.pilot.__main__ import modelfile


def test_modelfile_points_at_the_directory_and_asks_for_the_instruct_renderer(
    tmp_path: Path,
) -> None:
    lines = modelfile(tmp_path).splitlines()
    assert lines[0] == f"FROM {tmp_path.resolve()}"
    assert "RENDERER qwen3-vl-instruct" in lines
    assert "PARSER qwen3-vl-instruct" in lines
    assert not any("thinking" in line for line in lines), "the tuned model does not think aloud"
