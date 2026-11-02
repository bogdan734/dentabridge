"""The OpenAPI document as the site's client generator should see it.

FastAPI documents its own validation error; we send an ErrorResponse for every failure, so that is
what the document says. (Routes that need an Idempotency-Key declare the header themselves, in
IdempotentRoute.)
"""

from typing import Any

from fastapi import FastAPI

_ERROR_BODY = {
    "application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}},
}


def install_openapi(app: FastAPI) -> None:
    def build() -> dict[str, Any]:
        if app.openapi_schema is None:
            schema = FastAPI.openapi(app)
            for item in schema["paths"].values():
                for operation in item.values():
                    if "422" in operation["responses"]:
                        operation["responses"]["422"] = {
                            "description": "Validation error",
                            "content": _ERROR_BODY,
                        }
            for name in ("HTTPValidationError", "ValidationError"):
                schema["components"]["schemas"].pop(name, None)
            app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = build  # type: ignore[method-assign]
