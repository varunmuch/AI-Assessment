"""Mock downstream MCP server (HTTP JSON-RPC). Records every request it receives.

Run: uvicorn task2_mcp_gateway.mock_downstream:app --port 9001
"""

from fastapi import FastAPI, Request

TOOLS = [{"name": "get_status"}, {"name": "admin_reset_key"}]


def create_app() -> FastAPI:
    app = FastAPI()
    app.state.received = []  # list of (method, params) tuples

    @app.post("/")
    async def rpc(request: Request) -> dict:
        body = await request.json()
        method, params = body.get("method"), body.get("params") or {}
        app.state.received.append((method, params))
        if method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            result = {"content": [{"type": "text", "text": f"ran {params.get('name')}"}]}
        else:
            return {"jsonrpc": "2.0", "id": body.get("id"),
                    "error": {"code": -32601, "message": "Method not found"}}
        return {"jsonrpc": "2.0", "id": body.get("id"), "result": result}

    return app


app = create_app()
