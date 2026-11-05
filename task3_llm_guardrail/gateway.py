"""Task 3: LLM gateway that redacts PII from a streamed completion in real time.

Run: uvicorn task3_llm_guardrail.gateway:app --port 8100
"""

import json
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from task3_llm_guardrail.redactor import StreamRedactor

logger = logging.getLogger("task3")

PROVIDER_URL = "http://127.0.0.1:8101/v1/chat/completions"
ERROR_BODY = {"error": {"type": "upstream_error", "message": "The model provider failed."}}


def error_response() -> JSONResponse:
    return JSONResponse(ERROR_BODY, status_code=502)


def sse(chunk: dict) -> str:
    return f"data: {json.dumps(chunk)}\n\n"


def redact_choice(choice: dict, redactor: StreamRedactor) -> dict | None:
    """Redact one choice's delta; return None if nothing is left to send."""
    delta = dict(choice.get("delta") or {})
    content = delta.get("content")
    if content is not None:
        delta["content"] = redactor.feed(content)
    if choice.get("finish_reason"):
        delta["content"] = delta.get("content", "") + redactor.flush()
    if delta.get("content") == "":
        del delta["content"]
    if not delta and not choice.get("finish_reason"):
        return None
    return {**choice, "delta": delta}


def create_app(provider_url: str = PROVIDER_URL) -> FastAPI:
    client = httpx.AsyncClient(timeout=None)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await client.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.post("/v1/chat/completions")
    async def completions(request: Request) -> Response:
        body = {**await request.json(), "stream": True}
        # Open the upstream stream first, so a provider failure becomes an HTTP error, not a 200.
        try:
            upstream = await client.send(client.build_request("POST", provider_url, json=body), stream=True)
        except httpx.HTTPError:
            logger.warning("Provider unreachable", exc_info=True)
            return error_response()
        if upstream.status_code != 200:
            logger.warning("Provider returned HTTP %s", upstream.status_code)
            await upstream.aclose()
            return error_response()

        async def stream():
            redactors: dict[int, StreamRedactor] = {}  # one per completion choice
            template, done = None, False
            try:
                async for line in upstream.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: "):]
                    if data == "[DONE]":
                        done = True
                        break
                    chunk = json.loads(data)
                    template = template or chunk
                    choices = []
                    for choice in chunk.get("choices", []):
                        redactor = redactors.setdefault(choice.get("index", 0), StreamRedactor())
                        if (safe := redact_choice(choice, redactor)) is not None:
                            choices.append(safe)
                    if choices:
                        yield sse({**chunk, "choices": choices})
            except httpx.HTTPError:
                logger.warning("Provider stream broke", exc_info=True)
            finally:
                await upstream.aclose()

            if not done:
                # Interrupted: report the failure, drop any held text, and never send [DONE].
                yield sse(ERROR_BODY)
                return
            # Choices that ended without finish_reason: flush their held text now.
            rest = [{"index": i, "delta": {"content": text}, "finish_reason": None}
                    for i, r in redactors.items() if (text := r.flush())]
            if rest and template:
                yield sse({**template, "choices": rest})
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


app = create_app()
