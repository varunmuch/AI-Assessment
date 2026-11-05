"""Task 2: wire-format parsing, proxy forwarding, authorization."""

import json

import httpx
import pytest

from task2_mcp_gateway import gateway, mock_downstream

ADMIN = {"Authorization": "Bearer admin-token"}
VIEWER = {"Authorization": "Bearer viewer-token"}


@pytest.fixture
def downstream():
    return mock_downstream.create_app()


@pytest.fixture
async def client(downstream):
    app = gateway.create_app("http://downstream/", transport=httpx.ASGITransport(app=downstream))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        yield c


def call(name: str, request_id=7) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "method": "tools/call", "params": {"name": name}}


# --- Wire format ---------------------------------------------------------

async def test_malformed_json_is_parse_error(client):
    r = await client.post("/", content=b"{not json", headers=VIEWER)
    assert r.json() == {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}


@pytest.mark.parametrize(
    "payload",
    [
        [],
        "just a string",
        {"id": 1, "method": "tools/list"},  # missing jsonrpc
        {"jsonrpc": "1.0", "id": 1, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 1},  # missing method
        {"jsonrpc": "2.0", "id": 1, "method": 5},
        {"jsonrpc": "2.0", "id": {"x": 1}, "method": "tools/list"},
    ],
)
async def test_invalid_structure_is_invalid_request(client, downstream, payload):
    r = await client.post("/", json=payload, headers=VIEWER)
    assert r.json()["error"]["code"] == -32600
    assert downstream.state.received == []


async def test_invalid_request_preserves_valid_id(client):
    r = await client.post("/", json={"jsonrpc": "2.0", "id": 99, "method": 5}, headers=VIEWER)
    assert r.json()["id"] == 99


@pytest.mark.parametrize("request_id", [1, "abc-123"])
async def test_ids_preserved_on_forward_and_on_denial(client, request_id):
    forwarded = await client.post("/", json=call("get_status", request_id), headers=VIEWER)
    denied = await client.post("/", json=call("admin_reset_key", request_id), headers=VIEWER)
    assert forwarded.json()["id"] == request_id
    assert denied.json()["id"] == request_id


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer nope"}, {"Authorization": "admin-token"}])
async def test_missing_or_unknown_token_is_401_jsonrpc(client, downstream, headers):
    r = await client.post("/", json=call("get_status"), headers=headers)
    assert r.status_code == 401
    assert r.json()["jsonrpc"] == "2.0" and "error" in r.json()
    assert downstream.state.received == []


@pytest.mark.parametrize("header", ["bearer admin-token", "BEARER admin-token", "Bearer   admin-token"])
async def test_bearer_scheme_is_case_insensitive(client, downstream, header):
    r = await client.post("/", json=call("admin_reset_key"), headers={"Authorization": header})
    assert r.json()["result"]["content"][0]["text"] == "ran admin_reset_key"
    assert downstream.state.received == [("tools/call", {"name": "admin_reset_key"})]


# --- Forwarding ----------------------------------------------------------

async def test_tools_list_forwarded_unchanged(client, downstream):
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    r = await client.post("/", json=payload, headers=VIEWER)
    assert r.json() == {"jsonrpc": "2.0", "id": 1, "result": {"tools": mock_downstream.TOOLS}}
    assert downstream.state.received == [("tools/list", {})]


async def test_allowed_tool_call_forwarded(client, downstream):
    r = await client.post("/", json=call("get_status"), headers=VIEWER)
    assert r.json()["result"]["content"][0]["text"] == "ran get_status"
    assert downstream.state.received == [("tools/call", {"name": "get_status"})]


async def test_other_methods_forwarded(client, downstream):
    r = await client.post("/", json={"jsonrpc": "2.0", "id": 3, "method": "ping"}, headers=VIEWER)
    assert r.json()["error"]["code"] == -32601  # downstream's own answer
    assert downstream.state.received == [("ping", {})]


# --- Authorization -------------------------------------------------------

async def test_viewer_cannot_call_admin_tool(client, downstream):
    r = await client.post("/", json=call("admin_reset_key"), headers=VIEWER)
    assert r.json() == {
        "jsonrpc": "2.0",
        "id": 7,
        "error": {"code": -32001, "message": "Unauthorized Tool Call"},
    }
    assert downstream.state.received == []


async def test_admin_can_call_admin_tool(client, downstream):
    r = await client.post("/", json=call("admin_reset_key"), headers=ADMIN)
    assert r.json()["result"]["content"][0]["text"] == "ran admin_reset_key"
    assert downstream.state.received == [("tools/call", {"name": "admin_reset_key"})]


async def test_duplicate_name_key_cannot_smuggle_admin_tool():
    """The gateway checks the last "name"; a first-wins downstream must not see the other one."""
    received = []

    def downstream(request: httpx.Request) -> httpx.Response:
        received.append(request.content)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 7, "result": {"content": []}})

    app = gateway.create_app("http://downstream/", transport=httpx.MockTransport(downstream))
    raw = b'{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"admin_reset_key","name":"get_status"}}'
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        r = await c.post("/", content=raw, headers=VIEWER)
    assert r.status_code == 200
    assert len(received) == 1 and b"admin_reset_key" not in received[0]
    assert json.loads(received[0])["params"] == {"name": "get_status"}


# --- Downstream failures -------------------------------------------------

@pytest.mark.parametrize("error", [httpx.ConnectError("refused"), httpx.ReadTimeout("slow")])
async def test_downstream_failure_returns_jsonrpc_error_with_id(error):
    def fail(request):
        raise error

    app = gateway.create_app("http://downstream/", transport=httpx.MockTransport(fail))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        r = await c.post("/", json=call("get_status", "req-5"), headers=VIEWER)
    assert r.status_code == 502
    assert r.json() == {
        "jsonrpc": "2.0",
        "id": "req-5",
        "error": {"code": -32003, "message": "Downstream unavailable"},
    }


# --- Notifications (no "id" member) --------------------------------------

def notify(name: str) -> dict:
    return {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": name}}


async def test_denied_notification_is_dropped_without_body(client, downstream):
    r = await client.post("/", json=notify("admin_reset_key"), headers=VIEWER)
    assert r.status_code == 202 and r.content == b""
    assert downstream.state.received == []


async def test_allowed_notification_is_forwarded_without_body(client, downstream):
    r = await client.post("/", json=notify("get_status"), headers=VIEWER)
    assert r.status_code == 202 and r.content == b""
    assert downstream.state.received == [("tools/call", {"name": "get_status"})]


async def test_notification_with_bad_token_is_401_without_body(client, downstream):
    r = await client.post("/", json=notify("get_status"), headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401 and r.content == b""
    assert downstream.state.received == []


async def test_explicit_null_id_is_a_request_not_a_notification(client):
    r = await client.post("/", json=call("admin_reset_key", request_id=None), headers=VIEWER)
    assert r.json() == {"jsonrpc": "2.0", "id": None, "error": {"code": -32001, "message": "Unauthorized Tool Call"}}
