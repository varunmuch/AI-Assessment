"""Mock LLM provider that streams OpenAI-style SSE chunks.

Run: uvicorn task3_llm_guardrail.mock_provider:app --port 8101
"""

import asyncio
import json
import time

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, StreamingResponse

DEFAULT_TEXT = "Contact jane.doe@example.com, SSN 123-45-6789, card 4111 1111 1111 1111. Thanks!"


def chunk(content: str | None = None, finish_reason: str | None = None, index: int = 0) -> str:
    delta = {"content": content} if content is not None else {}
    body = {
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "model": "mock",
        "choices": [{"index": index, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(body)}\n\n"


def create_app(
    text: str | list[str] = DEFAULT_TEXT,
    chunk_size: int = 5,
    delay: float = 0.0,
    status: int = 200,
    fail_after: int | None = None,
    send_done: bool = True,
) -> FastAPI:
    """`text` may be a list: one completion choice per entry, streamed interleaved.

    `status` != 200 returns an error instead of a stream. `fail_after=N` breaks the
    connection after N content chunks; `send_done=False` ends the stream without [DONE].
    """
    texts = [text] if isinstance(text, str) else text
    app = FastAPI()
    app.state.finished_at = None

    @app.post("/v1/chat/completions")
    async def completions():
        if status != 200:
            return PlainTextResponse("provider exploded", status_code=status)

        async def stream():
            sent = 0
            for i in range(0, max(map(len, texts)), chunk_size):
                for index, t in enumerate(texts):
                    if t[i : i + chunk_size]:
                        if sent == fail_after:
                            raise RuntimeError("simulated connection drop")
                        yield chunk(t[i : i + chunk_size], index=index)
                        sent += 1
                await asyncio.sleep(delay)
            if not send_done:
                return
            for index in range(len(texts)):
                yield chunk(finish_reason="stop", index=index)
            yield "data: [DONE]\n\n"
            app.state.finished_at = time.monotonic()

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


app = create_app()
