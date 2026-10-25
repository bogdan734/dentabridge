import ast
from pathlib import Path

import dentabridge_core

PACKAGE_DIR = Path(dentabridge_core.__file__).parent
# Tooling and test helpers are imported by module path, not re-exported.
NOT_REEXPORTED = {"__init__", "testing", "schema_export"}


def _public_names(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return {n for n in names if not n.startswith("_")}


def test_every_public_name_exported() -> None:
    exported = set(dentabridge_core.__all__)
    missing = {
        path.name: sorted(_public_names(path) - exported)
        for path in sorted(PACKAGE_DIR.glob("*.py"))
        if path.stem not in NOT_REEXPORTED and _public_names(path) - exported
    }
    assert not missing


def test_every_exported_name_resolves() -> None:
    assert all(hasattr(dentabridge_core, name) for name in dentabridge_core.__all__)
