"""Task 2: MCP security gateway - HTTP JSON-RPC reverse proxy with role checks.

Run: uvicorn task2_mcp_gateway.gateway:app --port 9000
"""

import json
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger("task2")

DOWNSTREAM_URL = "http://127.0.0.1:9001/"
DOWNSTREAM_TIMEOUT = 10.0  # seconds
TOKEN_ROLES = {"admin-token": "admin", "viewer-token": "viewer"}


def rpc_error(request_id, code: int, message: str, status: int = 200) -> JSONResponse:
    body = {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}
    return JSONResponse(body, status_code=status)


def role_for(request: Request) -> str | None:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    return TOKEN_ROLES.get(token) if scheme == "Bearer" else None


def is_valid_request(payload) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("jsonrpc") == "2.0"
        and isinstance(payload.get("method"), str)
        and type(payload.get("id")) in (str, int, type(None))
        and isinstance(payload.get("params", {}), (dict, list))
    )


def create_app(downstream_url: str = DOWNSTREAM_URL, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    client = httpx.AsyncClient(transport=transport, timeout=DOWNSTREAM_TIMEOUT)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await client.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.post("/")
    async def proxy(request: Request) -> Response:
        raw = await request.body()
        try:
            payload, parsed = json.loads(raw), True
        except ValueError:
            payload, parsed = None, False
        valid = parsed and is_valid_request(payload)
        # A notification has no "id" member at all; "id": null is still a request.
        notification = valid and "id" not in payload
        request_id = payload.get("id") if valid else None

        def reply_error(code: int, message: str, status: int = 200) -> Response:
            if notification:  # never answered with a JSON-RPC body
                return Response(status_code=202 if status == 200 else status)
            return rpc_error(request_id, code, message, status)

        role = role_for(request)
        if role is None:
            return reply_error(-32000, "Missing or invalid bearer token", status=401)
        if not parsed:
            return rpc_error(None, -32700, "Parse error")
        if not valid:
            bad_id = payload.get("id") if isinstance(payload, dict) else None
            return rpc_error(bad_id if type(bad_id) in (str, int) else None, -32600, "Invalid Request")

        if payload["method"] == "tools/call":
            name = payload.get("params", {}).get("name") if isinstance(payload.get("params"), dict) else None
            if not isinstance(name, str):
                return reply_error(-32602, "Invalid params")
            if name.startswith("admin_") and role != "admin":
                return reply_error(-32001, "Unauthorized Tool Call")

        # Forward the payload that was checked, not the raw bytes: with duplicate keys, the
        # downstream's JSON parser could otherwise read a different tool name than the one authorized.
        try:
            upstream = await client.post(
                downstream_url, content=json.dumps(payload), headers={"content-type": "application/json"}
            )
        except httpx.HTTPError:
            logger.warning("Downstream request failed", exc_info=True)
            return reply_error(-32002, "Downstream unavailable", status=502)
        if notification:
            return Response(status_code=202)
        return Response(upstream.content, status_code=upstream.status_code, media_type="application/json")

    return app


app = create_app()
