"""Refuse an oversized request while it is being read, before any of it is spooled or decoded."""

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from dentabridge_api.errors import ApiError
from dentabridge_core import ErrorCode


class BodyLimit:
    """Pure ASGI middleware: POSTs to the listed paths may not carry more than the given bytes."""

    def __init__(self, app: ASGIApp, limits: dict[str, int]) -> None:
        self.app = app
        self.limits = limits

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        limit = self.limits.get(scope.get("path", "")) if scope["type"] == "http" else None
        if limit is None or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        declared = int(dict(scope["headers"]).get(b"content-length", 0))
        received = 0

        async def limited() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                # raised while the route reads the body, where the error handlers can answer it
                if received > limit or declared > limit:
                    raise ApiError(413, ErrorCode.PAYLOAD_TOO_LARGE)
            return message

        await self.app(scope, limited, send)
