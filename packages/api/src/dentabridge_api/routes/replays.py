from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi import Path as PathParam
from fastapi.responses import FileResponse

from dentabridge_api.errors import ApiError
from dentabridge_api.ratelimit import rate_limited
from dentabridge_core import ErrorCode, ReplayManifest

router = APIRouter(tags=["replays"], dependencies=[Depends(rate_limited("read"))])

Name = Annotated[str, PathParam(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")]


def _inside(base: Path, *parts: str) -> Path:
    """A file below `base`, or 404: a name never leaves the replay folder."""
    path = (base.joinpath(*parts)).resolve()
    if not path.is_relative_to(base.resolve()) or not path.is_file():
        raise ApiError(404, ErrorCode.NOT_FOUND)
    return path


@router.get("/replays/{replay_id}", operation_id="getReplay")
def get_replay(replay_id: Name, request: Request) -> ReplayManifest:
    path = _inside(request.app.state.settings.replays_dir, replay_id, "manifest.json")
    return ReplayManifest.model_validate_json(path.read_text(encoding="utf-8"))


@router.get(
    "/replays/{replay_id}/audio/{file}", operation_id="getReplayAudio", response_class=FileResponse
)
def get_replay_audio(replay_id: Name, file: Name, request: Request) -> FileResponse:
    path = _inside(request.app.state.settings.replays_dir, replay_id, "audio", file)
    return FileResponse(path, media_type="audio/ogg")
