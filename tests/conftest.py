"""Helpers shared by the repo-level tests."""

import importlib.util
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


@pytest.fixture
def tool() -> Callable[[str], ModuleType]:
    """Loads a script of tools/ as a module: they are scripts, not an installed package."""

    def load(name: str) -> ModuleType:
        spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load
