"""Task 4: rate-limited chat completions router with primary -> backup failover.

Run: uvicorn task4_model_router.router:app --port 8200
"""

import asyncio
import hashlib
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
TOKENS_PER_MESSAGE = 4  # role and separators around each message (OpenAI chat format)
REPLY_PRIMING_TOKENS = 3  # tokens that start the assistant's reply
DEFAULT_OUTPUT_RESERVATION = 256  # output tokens reserved when the client sends no max_tokens


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    max_tokens: int | None = Field(None, ge=1)  # forwarded only if the client sent it


class Failover(Exception):
    """Primary returned 429 or timed out."""


class UpstreamError(Exception):
    """A provider failed in a way that ends the request."""


def gateway_error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"type": "gateway_error", "code": code, "message": message}}, status_code=status
    )


def estimate_tokens(req: ChatRequest) -> int:
    """Tokens to reserve before the provider answers: an estimate of the prompt plus the output limit.

    The reservation is settled to the provider's reported usage afterwards.
    """
    prompt = sum(math.ceil(len(m.content) / 4) + TOKENS_PER_MESSAGE for m in req.messages)
    output = req.max_tokens if req.max_tokens is not None else DEFAULT_OUTPUT_RESERVATION
    return prompt + REPLY_PRIMING_TOKENS + output


def reported_tokens(result: dict) -> int | None:
    """Total tokens the provider says it used, or None if it did not say."""
    usage = result.get("usage")
    if not isinstance(usage, dict):
        return None
    total = usage.get("total_tokens")
    if type(total) is int and total >= 0:
        return total
    parts = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if all(type(n) is int and n >= 0 for n in parts):
        return sum(parts)
    return None


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
        try:
            result = resp.json()
        except ValueError:
            result = None
        if not isinstance(result, dict):
            logger.warning("Provider %s returned an invalid body", url)
            raise UpstreamError()
        return result

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        # The auth scheme is case-insensitive (RFC 9110 section 11.1).
        scheme, _, api_key = request.headers.get("authorization", "").strip().partition(" ")
        api_key = api_key.strip()
        if scheme.lower() != "bearer" or not api_key:
            return gateway_error(401, "unauthorized", "Missing or invalid API key.")
        try:
            req = ChatRequest.model_validate_json(await request.body())
        except ValidationError:
            return gateway_error(400, "invalid_request", "Request body is invalid.")

        cost = estimate_tokens(req)
        if cost > limiter.limit:  # could never fit, so it is a bad request rather than a rate limit
            return gateway_error(400, "request_too_large", "Request exceeds the per-minute token limit.")
        tenant = hashlib.sha256(api_key.encode()).hexdigest()  # raw keys never reach the database
        reservation = await limiter.reserve(tenant, cost)
        if reservation is None:
            return gateway_error(429, "rate_limited", "Token rate limit exceeded. Retry later.")

        payload = req.model_dump(exclude_none=True)
        try:
            try:
                result = await call(primary_url, payload, is_primary=True)
            except Failover:
                logger.info("Failing over to backup provider")
                result = await call(backup_url, payload, is_primary=False)
        except UpstreamError:
            await limiter.release(reservation)  # the tenant received nothing
            return gateway_error(502, "upstream_unavailable", "The model provider is unavailable.")

        # Without reported usage, the reservation stands as the charge.
        if (used := reported_tokens(result)) is not None:
            await limiter.settle(reservation, used)
        return result

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return gateway_error(exc.status_code, "http_error", "Request could not be handled.")

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        logger.exception("Unhandled error")
        return gateway_error(500, "internal_error", "Internal gateway error.")

    return app


app = create_app()
