"""FastAPI application entry point."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from app.api.health import router as health_router
from app.api.routes import router as api_router
from app.core.config import settings
from app.core.logging import configure_logging
from app.services.repository import initialize_database

configure_logging(settings.log_level)
logger = logging.getLogger("image_forensics")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize persistent result storage before accepting requests."""
    initialize_database()
    yield


app = FastAPI(
    title="AI Image Forensics",
    version=settings.app_version,
    docs_url="/api/docs",
    redoc_url=None,
    lifespan=lifespan,
)
app.include_router(health_router)
app.include_router(api_router)
FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Attach a request ID and basic security headers."""
    request_id = request.headers.get("X-Request-ID", "")[:100] or uuid4().hex
    request.state.request_id = request_id
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "unhandled request error",
            extra={"request_id": request_id, "status_code": 500},
        )
        response = JSONResponse(
            {"detail": "Internal server error.", "request_id": request_id},
            status_code=500,
        )
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    if request.url.path == "/api/docs":
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; img-src 'self' data:"
        )
    else:
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' blob: data:"
        )
    logger.info(
        "request served",
        extra={
            "request_id": request_id,
            "duration_ms": round((time.perf_counter() - started) * 1000),
            "status_code": response.status_code,
        },
    )
    return response


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the local dashboard."""
    return FileResponse(FRONTEND / "index.html")


@app.get("/assets/{asset_path:path}", include_in_schema=False)
def assets(asset_path: str) -> FileResponse:
    """Serve only known static frontend assets."""
    allowed = {"app.css", "app.js", "qr-code.png"}
    if asset_path not in allowed:
        return FileResponse(FRONTEND / "index.html", status_code=404)
    return FileResponse(FRONTEND / asset_path)


class RequestBodyTooLarge(Exception):
    """Raised before multipart parsing consumes an oversized request body."""


@app.exception_handler(RequestBodyTooLarge)
async def oversized_body_handler(request: Request, exc: RequestBodyTooLarge):
    """Return a safe 413 if a chunked body crosses the limit during parsing."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.info(
        "request body limit exceeded",
        extra={"request_id": request_id, "status_code": 413},
    )
    return JSONResponse(
        {
            "detail": "Request exceeds the configured upload limit.",
            "request_id": request_id,
        },
        status_code=413,
    )


class AnalyzeBodyLimit:
    """Bound request bodies before FastAPI's multipart parser is invoked."""

    def __init__(self, asgi_app, limit: int):
        self.asgi_app = asgi_app
        self.limit = limit

    def __getattr__(self, name):
        return getattr(self.asgi_app, name)

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope.get("path") != "/api/analyze"
            or scope.get("method") != "POST"
        ):
            await self.asgi_app(scope, receive, send)
            return
        body_limit = self.limit + 128 * 1024
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        declared = headers.get(b"content-length")
        if declared:
            try:
                if int(declared) > body_limit:
                    await self._reject(send)
                    return
            except ValueError:
                await self._reject(send)
                return
        received = 0
        response_started = False

        async def bounded_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > body_limit:
                    raise RequestBodyTooLarge
            return message

        async def tracked_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.asgi_app(scope, bounded_receive, tracked_send)
        except RequestBodyTooLarge:
            if not response_started:
                await self._reject(send)
            logger.warning("request body limit exceeded", extra={"status_code": 413})

    @staticmethod
    async def _reject(send):
        body = b'{"detail":"Request exceeds the configured upload limit."}'
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


# This outer ASGI wrapper limits chunked uploads before multipart parsing.
app = AnalyzeBodyLimit(app, settings.max_upload_bytes)
