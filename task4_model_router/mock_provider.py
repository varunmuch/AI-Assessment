"""Mock LLM providers with controllable delay and status code.

Run: uvicorn task4_model_router.mock_provider:primary --port 8201
     uvicorn task4_model_router.mock_provider:backup --port 8202
"""

import asyncio

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse


def create_app(name: str) -> FastAPI:
    app = FastAPI()
    app.state.delay = 0.0  # seconds
    app.state.status = 200
    app.state.body = "provider error"  # returned when status != 200
    app.state.calls = 0
    app.state.requests = []  # JSON body of every request received
    app.state.usage = None  # e.g. {"total_tokens": 42}; included in successful responses when set

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        app.state.calls += 1
        app.state.requests.append(await request.json())
        await asyncio.sleep(app.state.delay)
        if app.state.status != 200:
            return PlainTextResponse(app.state.body, status_code=app.state.status)
        body = {"model": name, "choices": [{"message": {"role": "assistant", "content": f"hi from {name}"}}]}
        if app.state.usage is not None:
            body["usage"] = app.state.usage
        return body

    return app


primary = create_app("primary")
backup = create_app("backup")
