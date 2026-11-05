"""Task 3: LLM gateway that redacts PII from a streamed completion in real time.

Run: uvicorn task3_llm_guardrail.gateway:app --port 8100
"""

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from task3_llm_guardrail.redactor import StreamRedactor

logger = logging.getLogger("task3")

PROVIDER_URL = "http://127.0.0.1:8101/v1/chat/completions"
ERROR_BODY = {"error": {"type": "upstream_error", "message": "The model provider failed."}}
TEXT_FIELDS = ("content", "refusal")  # model-generated text in a delta


class MalformedEvent(ValueError):
    """The provider sent an event that is not a valid completion chunk."""


def error_response() -> JSONResponse:
    return JSONResponse(ERROR_BODY, status_code=502)


def sse(chunk: dict) -> str:
    return f"data: {json.dumps(chunk)}\n\n"


async def sse_data(lines: AsyncIterator[str]) -> AsyncIterator[str]:
    """Yield the data of each SSE event.

    Follows SSE framing: one optional space after the colon, several data lines
    joined with newlines, a blank line ends the event, and lines starting with
    ":" are comments.
    """
    data: list[str] = []
    async for line in lines:
        if not line:
            if data:
                yield "\n".join(data)
                data = []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if field == "data":
            data.append(value[1:] if value.startswith(" ") else value)
    if data:  # last event without a trailing blank line
        yield "\n".join(data)


def parse_chunk(data: str) -> dict:
    try:
        chunk = json.loads(data)
    except ValueError as exc:
        raise MalformedEvent("invalid JSON") from exc
    if not isinstance(chunk, dict) or not isinstance(chunk.get("choices", []), list):
        raise MalformedEvent("not a completion chunk")
    for choice in chunk.get("choices", []):
        if not (
            isinstance(choice, dict)
            and isinstance(choice.get("index", 0), int)
            and isinstance(choice.get("delta") or {}, dict)
        ):
            raise MalformedEvent("invalid choice")
    return chunk


class ChoiceRedactor:
    """Redaction state for one completion choice: its text fields and each tool call's arguments."""

    def __init__(self) -> None:
        self.text = {field: StreamRedactor() for field in TEXT_FIELDS}
        self.tool_args: dict[int, StreamRedactor] = {}

    def redact(self, delta: dict, final: bool) -> dict:
        delta = dict(delta)
        for field, redactor in self.text.items():
            value = delta.get(field)
            out = redactor.feed(value) if isinstance(value, str) else ""
            if final:
                out += redactor.flush()
            if out:
                delta[field] = out
            elif isinstance(value, str):
                del delta[field]  # everything is held back for now

        calls = delta.get("tool_calls")
        if isinstance(calls, list):
            delta["tool_calls"] = [self._redact_call(call) for call in calls]
        if final:
            self._flush_calls(delta)
        return delta

    def _redact_call(self, call):
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("arguments"), str):
            return call
        if not isinstance(index := call.get("index", 0), int):
            raise MalformedEvent("invalid tool call index")
        redactor = self.tool_args.setdefault(index, StreamRedactor())
        return {**call, "function": {**function, "arguments": redactor.feed(function["arguments"])}}

    def _flush_calls(self, delta: dict) -> None:
        for index, redactor in self.tool_args.items():
            if not (rest := redactor.flush()):
                continue
            calls = delta.setdefault("tool_calls", [])
            call = next((c for c in calls if isinstance(c, dict) and c.get("index", 0) == index), None)
            if call is None:
                calls.append({"index": index, "function": {"arguments": rest}})
            else:
                function = call.setdefault("function", {})
                function["arguments"] = function.get("arguments", "") + rest


def redact_choice(choice: dict, redactor: ChoiceRedactor) -> dict | None:
    """Redact one choice's delta; return None if nothing is left to send."""
    final = bool(choice.get("finish_reason"))
    delta = redactor.redact(choice.get("delta") or {}, final)
    if not delta and not final:
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
            redactors: dict[int, ChoiceRedactor] = {}  # one per completion choice
            template, done = None, False
            try:
                async for data in sse_data(upstream.aiter_lines()):
                    if data == "[DONE]":
                        done = True
                        break
                    chunk = parse_chunk(data)
                    template = template or chunk
                    choices = []
                    for choice in chunk.get("choices", []):
                        redactor = redactors.setdefault(choice.get("index", 0), ChoiceRedactor())
                        if (safe := redact_choice(choice, redactor)) is not None:
                            choices.append(safe)
                    # Chunks without choices (such as a final usage chunk) pass through.
                    if choices or not chunk.get("choices"):
                        yield sse({**chunk, "choices": choices})
            except httpx.HTTPError:
                logger.warning("Provider stream broke", exc_info=True)
            except MalformedEvent as exc:
                logger.warning("Provider sent a malformed event: %s", exc)
            finally:
                await upstream.aclose()

            if not done:
                # Interrupted: report the failure, drop any held text, and never send [DONE].
                yield sse(ERROR_BODY)
                return
            # Choices that ended without finish_reason: flush their held text now.
            rest = [{"index": i, "delta": delta, "finish_reason": None}
                    for i, r in redactors.items() if (delta := r.redact({}, final=True))]
            if rest and template:
                yield sse({**template, "choices": rest})
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


app = create_app()
