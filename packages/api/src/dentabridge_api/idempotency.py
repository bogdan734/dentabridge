"""Idempotency-Key handling: a retry replays the stored answer instead of acting twice.

Only successful (2xx) answers are stored; a failed attempt gives the key back, so the client can
retry it. The same key with another body is a client bug and gets 422.
"""

import hashlib
import re
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from fastapi import Request, Response
from fastapi.routing import APIRoute
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from starlette.datastructures import UploadFile

from dentabridge_api.clock import Clock
from dentabridge_api.db.models import IdempotencyRow
from dentabridge_api.errors import ApiError
from dentabridge_core import ErrorCode

KEY = re.compile(r"^[A-Za-z0-9-]{8,64}$")


@dataclass(frozen=True)
class StoredResponse:
    status: int
    body: bytes


class IdempotencyStore:
    def __init__(self, sessions: sessionmaker[Session], clock: Clock, ttl: timedelta) -> None:
        self._sessions = sessions
        self._clock = clock
        self._ttl = ttl

    def begin(self, key: str, route: str, body_sha256: str) -> StoredResponse | None:
        """None means "go ahead and act"; a stored answer means "this already happened"."""
        now = self._clock.now()
        try:
            with self._sessions.begin() as session:
                row = session.get(IdempotencyRow, (key, route))
                if row is not None and now - row.created_at > self._ttl:
                    session.delete(row)
                    session.flush()
                    row = None
                if row is None:
                    session.add(
                        IdempotencyRow(
                            key=key,
                            route=route,
                            body_sha256=body_sha256,
                            state="in_progress",
                            status=0,
                            body="",
                            created_at=now,
                        )
                    )
                    session.flush()
                    return None
                if row.body_sha256 != body_sha256:
                    raise ApiError(422, ErrorCode.IDEMPOTENCY_KEY_REUSE)
                if row.state != "done":
                    raise ApiError(409, ErrorCode.IDEMPOTENCY_IN_PROGRESS)
                return StoredResponse(row.status, row.body.encode())
        except IntegrityError as error:  # two requests raced for the same new key
            raise ApiError(409, ErrorCode.IDEMPOTENCY_IN_PROGRESS) from error

    def complete(self, key: str, route: str, status: int, body: bytes) -> None:
        with self._sessions.begin() as session:
            row = session.get(IdempotencyRow, (key, route))
            if row is not None:
                row.state, row.status, row.body = "done", status, body.decode()

    def abandon(self, key: str, route: str) -> None:
        with self._sessions.begin() as session:
            row = session.get(IdempotencyRow, (key, route))
            if row is not None:
                session.delete(row)


async def fingerprint(request: Request) -> str:
    """Hash what the client meant, not how it was encoded: a multipart retry has a new boundary."""
    if request.headers.get("content-type", "").startswith("multipart/form-data"):
        digest = hashlib.sha256()
        form = await request.form()
        for name in sorted(form.keys()):
            for value in form.getlist(name):
                data = await value.read() if isinstance(value, UploadFile) else str(value).encode()
                if isinstance(value, UploadFile):
                    await value.seek(0)
                digest.update(f"{name}\0{len(data)}\0".encode() + data)
        return digest.hexdigest()
    return hashlib.sha256(await request.body()).hexdigest()


class IdempotentRoute(APIRoute):
    """Routes on a router with this class require an Idempotency-Key and replay on a repeat."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        original = super().get_route_handler()

        async def handler(request: Request) -> Response:
            key = request.headers.get("Idempotency-Key", "")
            if not KEY.fullmatch(key):
                raise ApiError(400, ErrorCode.IDEMPOTENCY_KEY_REQUIRED)
            store: IdempotencyStore = request.app.state.idempotency
            route = f"{request.method} {self.path}"
            stored = store.begin(key, route, await fingerprint(request))
            if stored is not None:
                return Response(
                    stored.body,
                    status_code=stored.status,
                    media_type="application/json",
                    headers={"Idempotent-Replayed": "true"},
                )
            try:
                response = await original(request)
            except BaseException:
                store.abandon(key, route)
                raise
            if 200 <= response.status_code < 300:
                store.complete(key, route, response.status_code, bytes(response.body))
            else:
                store.abandon(key, route)
            return response

        return handler
