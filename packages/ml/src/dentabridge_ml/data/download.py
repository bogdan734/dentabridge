"""Download and unpack a source into <root>/raw/<source>/ and <root>/<source>/.

The network is touched only here, by `make data`. Archives are untrusted input: each source gets
its own empty directory, and the checksum is verified before anything is unpacked.
"""

import hashlib
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import httpx

from dentabridge_ml.data.sources import SOURCES, RemoteFile, SourceId

CHUNK = 1 << 20
REPORT_EVERY = 256 << 20
MAX_FAILURES = 8  # consecutive reconnects without a single new byte
_sleep = time.sleep  # a module attribute so tests do not wait


class ChecksumMismatch(Exception):
    pass


def raw_dir(root: Path, source: SourceId) -> Path:
    return root / "raw" / source


def source_dir(root: Path, source: SourceId) -> Path:
    return root / source


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, remote: RemoteFile) -> None:
    if path.stat().st_size != remote.size:
        raise ChecksumMismatch(f"{path.name}: size {path.stat().st_size} != {remote.size}")
    for algorithm, expected in (("md5", remote.md5), ("sha256", remote.sha256)):
        if expected is not None and file_digest(path, algorithm) != expected:
            raise ChecksumMismatch(f"{path.name}: {algorithm} differs from the publisher's value")


def _fetch(remote: RemoteFile, target: Path, client: httpx.Client) -> None:
    """Fetch with resume. Servers hang up mid-file (Zenodo did, at ~12 MB), so reconnect."""
    part = target.with_name(target.name + ".part")
    failures = 0
    while True:
        before = part.stat().st_size if part.exists() else 0
        try:
            _fetch_once(remote, target, client)
            return
        except httpx.TransportError as error:
            grew = part.exists() and part.stat().st_size > before
            failures = 0 if grew else failures + 1
            if failures >= MAX_FAILURES:
                raise
            print(f"{remote.name}: {error}; reconnecting", file=sys.stderr, flush=True)
            _sleep(min(2.0**failures, 30.0))


def _fetch_once(remote: RemoteFile, target: Path, client: httpx.Client) -> None:
    part = target.with_name(target.name + ".part")
    start = part.stat().st_size if part.exists() else 0
    if start == remote.size:  # a finished download that was never renamed
        part.rename(target)
        return
    # identity: an archive must arrive byte for byte, never re-compressed in transit
    headers = {"Accept-Encoding": "identity"}
    if start:
        headers["Range"] = f"bytes={start}-"
    with client.stream("GET", remote.url, headers=headers, follow_redirects=True) as response:
        response.raise_for_status()
        resumed = response.status_code == 206
        done = start if resumed else 0
        reported = done
        with part.open("ab" if resumed else "wb") as handle:
            # no chunk size: bytes are written as they arrive, a dropped connection loses none
            for chunk in response.iter_bytes():
                handle.write(chunk)
                done += len(chunk)
                if done - reported >= REPORT_EVERY:
                    reported = done
                    print(f"{remote.name}: {done / remote.size:.0%}", file=sys.stderr, flush=True)
    part.rename(target)


def download(source: SourceId, root: Path, client: httpx.Client | None = None) -> list[Path]:
    """Fetch every archive of a source, resuming partial files, and verify each one."""
    spec = SOURCES[source]
    if not spec.files:
        raise FileNotFoundError(f"{source} has no automatic download: see {spec.page}")
    target_dir = raw_dir(root, source)
    target_dir.mkdir(parents=True, exist_ok=True)
    owned = client is None
    http = client or httpx.Client(timeout=httpx.Timeout(30.0, read=60.0))
    try:
        paths: list[Path] = []
        for remote in spec.files:
            target = target_dir / remote.name
            if not target.exists():
                _fetch(remote, target, http)
            try:
                verify_file(target, remote)
            except ChecksumMismatch:
                target.unlink()  # a bad archive is never kept, the next run starts clean
                raise
            paths.append(target)
        return paths
    finally:
        if owned:
            http.close()


def extract(source: SourceId, root: Path) -> Path:
    """Unpack the verified archives into an empty <root>/<source>/ directory."""
    target = source_dir(root, source)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"{target} is not empty; remove it to extract again")
    target.mkdir(parents=True, exist_ok=True)
    for remote in SOURCES[source].files:
        archive = raw_dir(root, source) / remote.name
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(target)  # strips absolute and ".." members
        elif archive.suffix == ".rar":
            if shutil.which("bsdtar") is None:
                raise RuntimeError("bsdtar is needed to unpack .rar archives")
            subprocess.run(["bsdtar", "-xf", str(archive), "-C", str(target)], check=True)
        else:
            raise ValueError(f"unknown archive type: {archive.name}")
    return target
