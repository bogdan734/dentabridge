"""Who is translated into what, from the people in the room and the call's metadata.

A person joins as "<role>-<session_id>" (the API mints that identity): the role says which language
they speak, the session id ties them to this call. Agents, people of other calls and the tracks the
interpreter publishes are never listened to; hearing its own voice would make it translate itself.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from dentabridge_core import VERIFIED_PAIRS, CallRoomMetadata, Role

ROLES: tuple[Role, ...] = ("patient", "dentist")
TRANSLATED = "translated-"


@dataclass(frozen=True)
class ParticipantInfo:
    identity: str
    is_agent: bool


@dataclass(frozen=True)
class Direction:
    speaker_identity: str
    role: Role
    src: str
    tgt: str
    verified: bool  # the voice harness has measured this pair


def translated_track(role: Role) -> str:
    """The audio track that carries what `role` said, in the other person's language."""
    return TRANSLATED + role


def direction_of(participant: ParticipantInfo, meta: CallRoomMetadata) -> Direction | None:
    """How this participant's speech is translated; None when they are not one of the call's two
    people, or when both people speak the same language."""
    if participant.is_agent:
        return None
    for role in ROLES:
        if participant.identity == f"{role}-{meta.session_id}":
            (other,) = (r for r in ROLES if r != role)
            src, tgt = meta.languages[role], meta.languages[other]
            if src == tgt:
                return None
            return Direction(participant.identity, role, src, tgt, (src, tgt) in VERIFIED_PAIRS)
    return None


def route(participants: Sequence[ParticipantInfo], meta: CallRoomMetadata) -> list[Direction]:
    return [d for p in participants if (d := direction_of(p, meta)) is not None]


def should_subscribe(
    participant: ParticipantInfo, track_name: str, *, is_microphone: bool, meta: CallRoomMetadata
) -> bool:
    """Only a person's own microphone, never a translated track, whoever publishes it."""
    return (
        is_microphone
        and not track_name.startswith(TRANSLATED)
        and direction_of(participant, meta) is not None
    )


def spoken(direction: Direction, meta: CallRoomMetadata) -> bool:
    """Synthetic speech only for a pair whose quality was measured, in a call that asked for it."""
    return meta.mode == "consecutive" and direction.verified
