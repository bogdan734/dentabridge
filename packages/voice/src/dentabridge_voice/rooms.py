"""Which rooms need the interpreter. Every POLL_S it asks the LiveKit server for its rooms, joins
each call room that has one of the call's people in it, and serves it until serve_room returns.

A call room is named "db-<booking_id>" and carries a CallRoomMetadata, which the API's call token
put there. A room the interpreter could not serve is skipped until the call's people have left
it, so one broken call is reported once, not every poll.
"""

import asyncio
import functools
import logging
from collections.abc import Container
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Protocol

from livekit import api, rtc
from livekit.protocol.models import ParticipantInfo as ServerParticipant
from pydantic import ValidationError

from dentabridge_core import CallRoomMetadata
from dentabridge_voice.routing import ParticipantInfo, route
from dentabridge_voice.worker import Services, serve_room

POLL_S = 2.0
TOKEN_TTL = timedelta(hours=3)  # longer than a call; the server renews it while connected

log = logging.getLogger("dentabridge_voice")


@dataclass(frozen=True)
class LiveKitServer:
    url: str
    key: str
    secret: str = field(repr=False)


def call_metadata(name: str, metadata: str) -> CallRoomMetadata | None:
    if not name.startswith("db-"):
        return None
    try:
        return CallRoomMetadata.model_validate_json(metadata)
    except ValidationError:
        return None


def interpreter_token(server: LiveKitServer, room: str, meta: CallRoomMetadata) -> str:
    grants = api.VideoGrants(
        room_join=True,
        room=room,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
        agent=True,
    )
    return (
        api.AccessToken(server.key, server.secret)
        .with_identity(f"interpreter-{meta.session_id}")
        .with_name("Interpreter")
        .with_kind("agent")
        .with_ttl(TOKEN_TTL)
        .with_grants(grants)
        .to_jwt()
    )


class RoomLister(Protocol):
    """The part of the server's room service the interpreter reads."""

    async def list_rooms(self, request: api.ListRoomsRequest, /) -> api.ListRoomsResponse: ...

    async def list_participants(
        self, request: api.ListParticipantsRequest, /
    ) -> api.ListParticipantsResponse: ...


async def calls_waiting(
    service: RoomLister, skip: Container[str]
) -> list[tuple[str, CallRoomMetadata]]:
    """The call rooms that have one of the call's people in them, other than those in `skip`."""
    waiting = []
    for room in (await service.list_rooms(api.ListRoomsRequest())).rooms:
        meta = call_metadata(room.name, room.metadata)
        if meta is None or room.name in skip:
            continue
        # a room that closed since it was listed has no participants (server 1.13)
        listed = await service.list_participants(api.ListParticipantsRequest(room=room.name))
        infos = [
            ParticipantInfo(p.identity, is_agent=p.kind == ServerParticipant.Kind.AGENT)
            for p in listed.participants
        ]
        if route(infos, meta):
            waiting.append((room.name, meta))
    return waiting


async def interpret_room(
    server: LiveKitServer, room_name: str, meta: CallRoomMetadata, services: Services
) -> None:
    room = rtc.Room()
    token = interpreter_token(server, room_name, meta)
    await room.connect(server.url, token, rtc.RoomOptions(auto_subscribe=False))
    log.info("%s: interpreting %s", room_name, dict(meta.languages))
    try:
        await serve_room(room, meta, services)
    finally:
        await room.disconnect()


async def watch_rooms(server: LiveKitServer, services: Services) -> None:
    serving: dict[str, asyncio.Task[None]] = {}
    broken: set[str] = set()

    def finished(name: str, task: asyncio.Task[None]) -> None:
        serving.pop(name, None)
        if not task.cancelled() and (error := task.exception()) is not None:
            log.error("%s: the interpreter stopped", name, exc_info=error)
            broken.add(name)

    async with api.LiveKitAPI(server.url, server.key, server.secret) as lk:
        while True:
            waiting = await calls_waiting(lk.room, skip=serving.keys())
            broken &= {name for name, _ in waiting}  # retried once the call's people have left
            for name, meta in waiting:
                if name in broken:
                    continue
                task = asyncio.create_task(interpret_room(server, name, meta, services))
                task.add_done_callback(functools.partial(finished, name))
                serving[name] = task
            await asyncio.sleep(POLL_S)
