import asyncio
from uuid import UUID

import jwt
import pytest

pytest.importorskip("livekit.rtc")
from livekit import api
from livekit.protocol.models import ParticipantInfo as ServerParticipant

from dentabridge_core import CallRoomMetadata
from dentabridge_voice.rooms import LiveKitServer, call_metadata, calls_waiting, interpreter_token

SESSION = UUID("00000000-0000-4000-8000-000000000001")
META = CallRoomMetadata.model_validate(
    {
        "session_id": SESSION,
        "languages": {"patient": "ru", "dentist": "en"},
        "mode": "consecutive",
        "glossary_id": "dental-en-ru",
    }
)
SERVER = LiveKitServer("ws://127.0.0.1:7880", "devkey", "a-secret-of-at-least-thirty-two-chars")


def test_only_call_rooms_are_interpreted() -> None:
    assert call_metadata("db-booking", META.model_dump_json()) == META
    assert call_metadata("standup", META.model_dump_json()) is None
    assert call_metadata("db-booking", "") is None
    assert call_metadata("db-booking", '{"session_id": "not one"}') is None


def test_the_interpreter_joins_as_an_agent_that_publishes_and_subscribes() -> None:
    token = interpreter_token(SERVER, "db-booking", META)
    claims = jwt.decode(token, SERVER.secret, algorithms=["HS256"], options={"verify_aud": False})
    assert (claims["sub"], claims["kind"], claims["iss"]) == (
        f"interpreter-{SESSION}",
        "agent",
        "devkey",
    )
    assert claims["video"] == {
        "roomJoin": True,
        "room": "db-booking",
        "canPublish": True,
        "canSubscribe": True,
        "canPublishData": True,
        "agent": True,
    }


def test_the_secret_never_shows_in_logs() -> None:
    assert SERVER.secret not in repr(SERVER)


class Server:
    """A room service that knows these rooms and who is in them, and one room that is not a call."""

    def __init__(self, rooms: dict[str, list[ServerParticipant]]) -> None:
        self.rooms = rooms

    async def list_rooms(self, request: api.ListRoomsRequest) -> api.ListRoomsResponse:
        listed = [api.Room(name=name, metadata=META.model_dump_json()) for name in self.rooms]
        return api.ListRoomsResponse(rooms=[*listed, api.Room(name="standup", metadata="{}")])

    async def list_participants(
        self, request: api.ListParticipantsRequest
    ) -> api.ListParticipantsResponse:
        return api.ListParticipantsResponse(participants=self.rooms[request.room])


def person(
    identity: str, kind: ServerParticipant.Kind = ServerParticipant.Kind.STANDARD
) -> ServerParticipant:
    return ServerParticipant(identity=identity, kind=kind)


def test_a_call_room_is_joined_when_one_of_its_people_is_in_it() -> None:
    server = Server(
        {
            "db-waiting": [person(f"patient-{SESSION}")],
            "db-empty": [],
            "db-only-agents": [person(f"dentist-{SESSION}", ServerParticipant.Kind.AGENT)],
            "db-served": [person(f"dentist-{SESSION}")],
        }
    )
    waiting = asyncio.run(calls_waiting(server, skip={"db-served"}))
    assert waiting == [("db-waiting", META)]
