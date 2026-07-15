from typing import Any, Optional

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from core.response import error_response
from logs import get_logger

logger = get_logger(__name__)


class AppError(Exception):
    status_code: int = 500
    message: str = "Internal Server Error"
    details: Optional[Any] = None

    def __init__(self, message: Optional[str] = None, details: Optional[Any] = None) -> None:
        if message is not None:
            self.message = message
        if details is not None:
            self.details = details
        super().__init__(self.message)


class NotFoundError(AppError):
    status_code = 404
    message = "Resource not found"


class ValidationError(AppError):
    status_code = 422
    message = "Validation failed"


class AIError(AppError):
    status_code = 502
    message = "AI service error"


class ConfigError(AppError):
    status_code = 500
    message = "Configuration error"


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    logger.error(
        "AppError: status=%s message=%s path=%s",
        exc.status_code,
        exc.message,
        request.url.path,
    )
    return error_response(message=exc.message, status_code=exc.status_code, details=exc.details)


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()
    logger.error("ValidationError: path=%s errors=%s", request.url.path, errors)
    return error_response(message="Validation failed", status_code=422, details=errors)


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception path=%s error=%s", request.url.path, exc)
    return error_response(message="Internal Server Error", status_code=500)


EXCEPTION_HANDLERS: dict[int | type[Exception], callable] = {
    RequestValidationError: validation_error_handler,
    AppError: app_error_handler,
    Exception: unhandled_error_handler,
}
