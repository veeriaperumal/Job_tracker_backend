from typing import Any, Optional

from fastapi.responses import JSONResponse


def _safe(val: Any) -> Any:
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, dict):
        return {k: _safe(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_safe(v) for v in val]
    return val


def success_response(data: Any = None, message: str = "OK", status_code: int = 200) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": True, "message": message, "data": data},
    )


def error_response(
    message: str = "Internal Server Error",
    status_code: int = 500,
    details: Optional[Any] = None,
) -> JSONResponse:
    body: dict[str, Any] = {"success": False, "message": message}
    if details is not None:
        body["details"] = _safe(details)
    return JSONResponse(status_code=status_code, content=body)
