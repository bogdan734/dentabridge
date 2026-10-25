from dentabridge_core.schema_export import ALL_CONTRACTS, SCHEMAS_DIR, render_schema


def test_schema_snapshots_match() -> None:
    expected = {f"{m.__name__}.json": render_schema(m) for m in ALL_CONTRACTS}
    actual = {p.name: p.read_text() for p in SCHEMAS_DIR.glob("*.json")}
    assert actual.keys() == expected.keys(), "stale or missing snapshot: run schema_export"
    changed = sorted(name for name in expected if actual[name] != expected[name])
    assert not changed, f"contract changed without a refreshed snapshot: {changed}"


def test_every_contract_class_is_snapshotted() -> None:
    names = {m.__name__ for m in ALL_CONTRACTS}
    assert {"Finding", "AnalysisResult", "Quote", "Booking", "CallSession", "ChatEvent"} <= names
    assert len(names) == len(ALL_CONTRACTS), "two contracts share a class name"
