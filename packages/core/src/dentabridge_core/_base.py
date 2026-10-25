"""Base class and shared field types for every contract object."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints

Locale = Literal["en", "ru"]
Role = Literal["patient", "dentist"]

Iata = Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]
CurrencyCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]
CountryCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]
LanguageTag = Annotated[str, StringConstraints(pattern=r"^[a-z]{2,3}(-[A-Za-z0-9]{2,8})*$")]
Slug = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")]


class Contract(BaseModel):
    """Frozen, closed model: unknown fields are errors and instances never change."""

    # protected_namespaces=(): fields such as model_id would otherwise warn.
    # Defaulted fields are required in response schemas, so the generated TypeScript client
    # has no property that is optional only because the server fills a default.
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        protected_namespaces=(),
        json_schema_serialization_defaults_required=True,
    )
