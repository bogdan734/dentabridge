from uuid import UUID

from dentabridge_core import CallRoomMetadata
from dentabridge_voice.routing import Direction, ParticipantInfo, route, should_subscribe, spoken

SESSION = UUID("00000000-0000-4000-8000-000000000001")
OTHER_SESSION = UUID("00000000-0000-4000-8000-000000000002")
PATIENT = f"patient-{SESSION}"
DENTIST = f"dentist-{SESSION}"


def meta(patient: str = "ru", dentist: str = "en", mode: str = "consecutive") -> CallRoomMetadata:
    return CallRoomMetadata.model_validate(
        {
            "session_id": SESSION,
            "languages": {"patient": patient, "dentist": dentist},
            "mode": mode,
            "glossary_id": "dental-en-ru",
        }
    )


def human(identity: str) -> ParticipantInfo:
    return ParticipantInfo(identity=identity, is_agent=False)


def test_two_participants_two_directions() -> None:
    directions = route([human(PATIENT), human(DENTIST)], meta())
    assert directions == [
        Direction(speaker_identity=PATIENT, role="patient", src="ru", tgt="en", verified=True),
        Direction(speaker_identity=DENTIST, role="dentist", src="en", tgt="ru", verified=True),
    ]


def test_unverified_pair_passes_flag() -> None:
    directions = route([human(PATIENT), human(DENTIST)], meta(patient="de"))
    assert [(d.src, d.tgt, d.verified) for d in directions] == [
        ("de", "en", False),
        ("en", "de", False),
    ]


def test_people_who_share_a_language_need_no_interpreter() -> None:
    assert route([human(PATIENT), human(DENTIST)], meta(patient="en")) == []


def test_only_the_people_of_this_call_are_routed() -> None:
    strangers = [
        human(f"patient-{OTHER_SESSION}"),  # another call's token
        human(f"nurse-{SESSION}"),  # not a role
        human("patient"),
        human(f"patient-{SESSION}-2"),
        ParticipantInfo(identity=DENTIST, is_agent=True),  # an agent wearing a person's name
        ParticipantInfo(identity=f"interpreter-{SESSION}", is_agent=True),
    ]
    assert route([*strangers, human(PATIENT)], meta()) == [
        Direction(speaker_identity=PATIENT, role="patient", src="ru", tgt="en", verified=True)
    ]


def test_never_subscribes_agent_or_translated_tracks() -> None:
    call = meta()
    assert should_subscribe(human(PATIENT), "microphone", is_microphone=True, meta=call)
    assert not should_subscribe(
        ParticipantInfo(identity=f"interpreter-{SESSION}", is_agent=True),
        "translated-patient",
        is_microphone=True,
        meta=call,
    )
    assert not should_subscribe(
        ParticipantInfo(identity=PATIENT, is_agent=True),
        "microphone",
        is_microphone=True,
        meta=call,
    )
    assert not should_subscribe(
        human(PATIENT), "translated-dentist", is_microphone=True, meta=call
    ), "a person re-publishing the translation is not listened to"
    assert not should_subscribe(human(PATIENT), "screen", is_microphone=False, meta=call)
    assert not should_subscribe(
        human(f"patient-{OTHER_SESSION}"), "microphone", is_microphone=True, meta=call
    )


def test_only_a_verified_pair_in_a_spoken_call_gets_a_voice() -> None:
    (patient, _) = route([human(PATIENT), human(DENTIST)], meta())
    assert spoken(patient, meta())
    assert not spoken(patient, meta(mode="captions_only"))
    (unverified, _) = route([human(PATIENT), human(DENTIST)], meta(patient="de"))
    assert not spoken(unverified, meta())
