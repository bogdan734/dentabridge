"""One error shape for every failure: ErrorResponse plus X-Request-Id."""

import logging
from collections.abc import Mapping

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from dentabridge_core import ErrorCode, ErrorResponse

log = logging.getLogger("dentabridge_api")


class ApiError(Exception):
    def __init__(
        self,
        status: int,
        code: ErrorCode,
        detail: str | None = None,
        retry_after_s: int | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(f"{status} {code.value}")
        self.status = status
        self.code = code
        self.detail = detail
        self.retry_after_s = retry_after_s
        self.headers = dict(headers or {})


def request_id_of(request: Request) -> str:
    return str(getattr(request.state, "request_id", "unknown"))


def error_response(request: Request, error: ApiError) -> JSONResponse:
    body = ErrorResponse(
        code=error.code,
        message_key=f"error.{error.code.value}",
        detail=error.detail,
        request_id=request_id_of(request),
        retry_after_s=error.retry_after_s,
    )
    headers = dict(error.headers)
    if error.retry_after_s is not None:
        headers["Retry-After"] = str(error.retry_after_s)
    return JSONResponse(body.model_dump(mode="json"), status_code=error.status, headers=headers)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, error: ApiError) -> JSONResponse:
        return error_response(request, error)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, error: RequestValidationError) -> JSONResponse:
        fields = ", ".join(".".join(str(part) for part in e["loc"]) for e in error.errors())
        return error_response(request, ApiError(422, ErrorCode.VALIDATION_ERROR, fields))

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, error: StarletteHTTPException) -> JSONResponse:
        code = ErrorCode.NOT_FOUND if error.status_code in (404, 405) else ErrorCode.INTERNAL_ERROR
        status = 404 if error.status_code in (404, 405) else error.status_code
        return error_response(request, ApiError(status, code))

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, error: Exception) -> JSONResponse:
        # request id, route and the exception type only: never the body, never a client address
        log.error(
            "unhandled %s on %s [%s]",
            type(error).__name__,
            request.url.path,
            request_id_of(request),
        )
        return error_response(request, ApiError(500, ErrorCode.INTERNAL_ERROR))
