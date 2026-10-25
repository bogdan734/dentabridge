"""Configuration from the environment (prefix DENTABRIDGE_) or a .env file."""

import secrets
from pathlib import Path
from typing import Annotated, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DENTABRIDGE_", env_file=".env", extra="ignore")

    mode: Literal["demo", "live"] = "demo"
    uploads: Literal["on", "off"] = "on"  # off on the public Space: only the bundled samples
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:3000"]
    db_url: str = "sqlite:///var/dentabridge.db"
    vision_backend: str = "ollama:dentabridge-tuned"  # a spec for dentabridge_ml.serving
    release_config: Path = Path("release.json")  # what the shipped weights may claim
    image_max_px: int = 448  # the picture size the shipped weights were trained on
    livekit_url: str | None = None
    livekit_key: str | None = None
    livekit_secret: str | None = None
    call_secret: str = secrets.token_urlsafe(32)  # set it in .env to keep call tokens valid
    result_ttl_h: int = 24
    replays_dir: Path = Path("packages/voice/replays")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split(cls, value: object) -> object:
        return (
            [part.strip() for part in value.split(",") if part.strip()]
            if isinstance(value, str)
            else value
        )

    @field_validator("cors_origins")
    @classmethod
    def _no_wildcard(cls, value: list[str]) -> list[str]:
        if "*" in value:
            raise ValueError("a wildcard origin is refused: list the origins that may call the API")
        return value
