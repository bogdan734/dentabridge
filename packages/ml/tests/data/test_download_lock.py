import hashlib
import zipfile
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import httpx
import pytest

from dentabridge_core import FindingCode
from dentabridge_ml.data import download as dl
from dentabridge_ml.data.lock import LockMismatch, tree_digest, verify_lock, write_lock
from dentabridge_ml.data.sources import SOURCES, RemoteFile, SourceId, SourceSpec

PAYLOAD = bytes(range(256)) * 40


def _spec(content: bytes = PAYLOAD, *, md5: str | None = None) -> SourceSpec:
    remote = RemoteFile(
        name="data.zip",
        url="https://example.test/data.zip",
        size=len(content),
        md5=md5 or hashlib.md5(content).hexdigest(),
    )
    base = SOURCES["aku_caries"]
    return SourceSpec(
        title=base.title,
        licence=base.licence,
        page=base.page,
        files=(remote,),
        published=True,
        coverage=base.coverage,
        camera=base.camera,
    )


def _patch(monkeypatch: pytest.MonkeyPatch, spec: SourceSpec) -> None:
    monkeypatch.setattr(dl, "SOURCES", {"aku_caries": spec})


def _client(content: bytes, seen: list[str | None]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        wanted = request.headers.get("Range")
        seen.append(wanted)
        if wanted:
            start = int(wanted.removeprefix("bytes=").removesuffix("-"))
            return httpx.Response(206, content=content[start:])
        return httpx.Response(200, content=content)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_catalogue_pins_publisher_sizes_and_hashes() -> None:
    for spec in SOURCES.values():
        for remote in spec.files:
            assert remote.size > 0
            assert remote.md5 or remote.sha256
    assert not SOURCES["kaggle_oral_private"].published
    assert SOURCES["kaggle_oral_private"].files == ()
    assert all(s.licence.startswith("CC-BY") for s in SOURCES.values() if s.published)


def test_download_verifies_and_skips_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch, _spec())
    seen: list[str | None] = []
    (path,) = dl.download("aku_caries", tmp_path, _client(PAYLOAD, seen))
    assert path.read_bytes() == PAYLOAD
    dl.download("aku_caries", tmp_path, _client(PAYLOAD, seen))
    assert seen == [None], "a verified archive is not fetched again"


def test_download_resumes_a_partial_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, _spec())
    part = dl.raw_dir(tmp_path, "aku_caries")
    part.mkdir(parents=True)
    (part / "data.zip.part").write_bytes(PAYLOAD[:1000])
    seen: list[str | None] = []
    (path,) = dl.download("aku_caries", tmp_path, _client(PAYLOAD, seen))
    assert seen == ["bytes=1000-"]
    assert path.read_bytes() == PAYLOAD


class _DropsAfter(httpx.SyncByteStream):
    """Sends `cut` bytes, then the server hangs up like Zenodo did."""

    def __init__(self, content: bytes, cut: int) -> None:
        self._content, self._cut = content, cut

    def __iter__(self) -> Iterator[bytes]:
        if self._cut:
            yield self._content[: self._cut]
        raise httpx.RemoteProtocolError("peer closed connection without sending complete message")


def test_download_retries_a_dropped_connection_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch, _spec())
    monkeypatch.setattr(dl, "_sleep", lambda seconds: None)
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        wanted = request.headers.get("Range")
        seen.append(wanted)
        if wanted is None:
            return httpx.Response(200, stream=_DropsAfter(PAYLOAD, 1000))
        start = int(wanted.removeprefix("bytes=").removesuffix("-"))
        return httpx.Response(206, content=PAYLOAD[start:])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    (path,) = dl.download("aku_caries", tmp_path, client)
    assert seen == [None, "bytes=1000-"]
    assert path.read_bytes() == PAYLOAD


def test_download_gives_up_when_no_progress_is_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch, _spec())
    monkeypatch.setattr(dl, "_sleep", lambda seconds: None)
    monkeypatch.setattr(dl, "MAX_FAILURES", 3)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, stream=_DropsAfter(PAYLOAD, 0))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.RemoteProtocolError):
        dl.download("aku_caries", tmp_path, client)
    assert calls == 3


def test_a_complete_part_file_is_finished_without_another_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch, _spec())
    folder = dl.raw_dir(tmp_path, "aku_caries")
    folder.mkdir(parents=True)
    (folder / "data.zip.part").write_bytes(PAYLOAD)
    seen: list[str | None] = []
    (path,) = dl.download("aku_caries", tmp_path, _client(PAYLOAD, seen))
    assert seen == []
    assert path.read_bytes() == PAYLOAD


def test_bad_checksum_is_rejected_and_not_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch, _spec(md5="0" * 32))
    with pytest.raises(dl.ChecksumMismatch, match="md5"):
        dl.download("aku_caries", tmp_path, _client(PAYLOAD, []))
    assert not (dl.raw_dir(tmp_path, "aku_caries") / "data.zip").exists()


def test_private_source_has_no_automatic_download(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no automatic download"):
        dl.download("kaggle_oral_private", tmp_path)


def test_extract_unpacks_into_an_empty_directory_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch, _spec())
    archive = dl.raw_dir(tmp_path, "aku_caries") / "data.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("Images/a.jpg", "x")
        bundle.writestr("../escape.txt", "must stay inside")
    target = dl.extract("aku_caries", tmp_path)
    assert (target / "Images" / "a.jpg").read_text() == "x"
    assert not (tmp_path / "escape.txt").exists()
    with pytest.raises(FileExistsError):
        dl.extract("aku_caries", tmp_path)


def test_lock_detects_tamper(tmp_path: Path) -> None:
    source: SourceId = "alphadent"
    data = tmp_path / source
    (data / "images").mkdir(parents=True)
    (data / "images" / "a.jpg").write_bytes(b"one")
    (data / "images" / "b.jpg").write_bytes(b"two")
    lock = tmp_path / "data.lock.json"
    write_lock(tmp_path, lock, [source], today=date(2026, 10, 25))
    verify_lock(tmp_path, lock)

    (data / "images" / "b.jpg").write_bytes(b"TWO")
    with pytest.raises(LockMismatch, match="alphadent: tree digest differs"):
        verify_lock(tmp_path, lock)
    (data / "images" / "b.jpg").write_bytes(b"two")
    (data / "images" / "c.jpg").write_bytes(b"extra")
    with pytest.raises(LockMismatch, match="found 3 / 11"):
        verify_lock(tmp_path, lock)
    (data / "images" / "c.jpg").unlink()
    (data / "images" / "b.jpg").rename(data / "images" / "z.jpg")
    with pytest.raises(LockMismatch):
        verify_lock(tmp_path, lock)


def test_tree_digest_is_stable_and_counts_bytes(tmp_path: Path) -> None:
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "f").write_bytes(b"abc")
    first = tree_digest(tmp_path)
    assert first == tree_digest(tmp_path)
    assert (first.files, first.bytes) == (1, 3)


def test_private_source_only_under_private_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="private"):
        write_lock(tmp_path, tmp_path / "data.lock.json", ["kaggle_oral_private"])


def test_coverage_declared_for_published_sources() -> None:
    for source, spec in SOURCES.items():
        assert spec.coverage, source
    assert SOURCES["aku_caries"].coverage == {FindingCode.CARIES}
    assert SOURCES["mendeley_gingivitis"].coverage == {FindingCode.GINGIVITIS}
