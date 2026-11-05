"""Task 4: rate-limited chat completions router with primary -> backup failover.

Run: uvicorn task4_model_router.router:app --port 8200
"""

import asyncio
import logging
import math
import time
from collections.abc import Callable

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from starlette.exceptions import HTTPException

from task4_model_router.rate_limiter import RateLimiter

logger = logging.getLogger("task4")

PRIMARY_URL = "http://127.0.0.1:8201/v1/chat/completions"
BACKUP_URL = "http://127.0.0.1:8202/v1/chat/completions"
PROVIDER_TIMEOUT = 3.0  # seconds; the backup uses the same limit


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    max_tokens: int = Field(256, ge=1)


class Failover(Exception):
    """Primary returned 429 or timed out."""


class UpstreamError(Exception):
    """A provider failed in a way that ends the request."""


def gateway_error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"type": "gateway_error", "code": code, "message": message}}, status_code=status
    )


def estimate_tokens(req: ChatRequest) -> int:
    """Approximate reservation: charged up front, before the provider answers."""
    prompt_chars = sum(len(m.content) for m in req.messages)
    return math.ceil(prompt_chars / 4) + req.max_tokens


def create_app(
    primary_url: str = PRIMARY_URL,
    backup_url: str = BACKUP_URL,
    db_path: str = "rate_limits.db",
    clock: Callable[[], float] = time.time,
    http_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    app = FastAPI()
    limiter = RateLimiter(db_path, clock)
    client = http_client or httpx.AsyncClient()

    async def call(url: str, payload: dict, is_primary: bool) -> dict:
        try:
            resp = await asyncio.wait_for(client.post(url, json=payload), PROVIDER_TIMEOUT)
        except (asyncio.TimeoutError, httpx.TimeoutException):
            logger.warning("Provider timed out: %s", url)
            raise Failover() if is_primary else UpstreamError()
        except httpx.HTTPError:
            logger.warning("Provider unreachable: %s", url)
            raise UpstreamError()
        if resp.status_code == 429 and is_primary:
            raise Failover()
        if resp.status_code != 200:
            logger.warning("Provider %s returned %s", url, resp.status_code)
            raise UpstreamError()
        return resp.json()

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        scheme, _, api_key = request.headers.get("authorization", "").partition(" ")
        if scheme != "Bearer" or not api_key:
            return gateway_error(401, "unauthorized", "Missing or invalid API key.")
        try:
            req = ChatRequest.model_validate_json(await request.body())
        except ValidationError:
            return gateway_error(400, "invalid_request", "Request body is invalid.")

        if not await limiter.try_consume(api_key, estimate_tokens(req)):
            return gateway_error(429, "rate_limited", "Token rate limit exceeded. Retry later.")

        payload = req.model_dump()
        try:
            try:
                return await call(primary_url, payload, is_primary=True)
            except Failover:
                logger.info("Failing over to backup provider")
                return await call(backup_url, payload, is_primary=False)
        except UpstreamError:
            return gateway_error(502, "upstream_unavailable", "The model provider is unavailable.")

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return gateway_error(exc.status_code, "http_error", "Request could not be handled.")

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        logger.exception("Unhandled error")
        return gateway_error(500, "internal_error", "Internal gateway error.")

    return app


app = create_app()
