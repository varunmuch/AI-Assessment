"""Mock LLM providers with controllable delay and status code.

Run: uvicorn task4_model_router.mock_provider:primary --port 8201
     uvicorn task4_model_router.mock_provider:backup --port 8202
"""

import asyncio

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse


def create_app(name: str) -> FastAPI:
    app = FastAPI()
    app.state.delay = 0.0  # seconds
    app.state.status = 200
    app.state.body = "provider error"  # returned when status != 200
    app.state.calls = 0

    @app.post("/v1/chat/completions")
    async def completions():
        app.state.calls += 1
        await asyncio.sleep(app.state.delay)
        if app.state.status != 200:
            return PlainTextResponse(app.state.body, status_code=app.state.status)
        return {"model": name, "choices": [{"message": {"role": "assistant", "content": f"hi from {name}"}}]}

    return app


primary = create_app("primary")
backup = create_app("backup")
