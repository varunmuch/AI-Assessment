"""Task 1: stdio isolation, protocol compliance, validation edge cases."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
VALID_REFUND = {"customer_id": "CUST-00042", "amount": 25.5, "reason": "Damaged on arrival"}


class StdioClient:
    def __init__(self):
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "task1_mcp_server.server"],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.stdout_lines: list[str] = []
        self.next_id = 1

    def send(self, message: dict) -> None:
        self.send_raw(json.dumps(message))

    def send_raw(self, line: str) -> None:
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def request(self, method: str, params: dict) -> dict:
        request_id = self.next_id
        self.next_id += 1
        self.send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        response = self.read()
        assert response["id"] == request_id
        return response

    def read(self) -> dict:
        line = self.proc.stdout.readline()
        self.stdout_lines.append(line)
        return json.loads(line)

    def call(self, name: str, arguments: dict) -> dict:
        return self.request("tools/call", {"name": name, "arguments": arguments})

    def close(self) -> None:
        self.proc.stdin.close()
        rest = self.proc.stdout.read()
        self.stdout_lines.extend(line for line in rest.splitlines() if line)
        self.proc.wait(timeout=10)
        self.proc.stderr.close()
        self.proc.stdout.close()


@pytest.fixture(scope="module")
def client():
    c = StdioClient()
    c.request(
        "initialize",
        {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1"},
        },
    )
    c.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    yield c
    c.close()


def test_stdio_isolation_every_stdout_line_is_jsonrpc():
    c = StdioClient()
    c.request(
        "initialize",
        {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}},
    )
    c.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    c.request("tools/list", {})
    c.call("get_customer_record", {"customer_id": "CUST-00001"})
    c.call("trigger_refund", VALID_REFUND)
    c.call("get_customer_record", {"customer_id": "bad"})
    c.call("trigger_refund", {**VALID_REFUND, "amount": -1})
    c.close()

    assert len(c.stdout_lines) == 6
    for line in c.stdout_lines:
        message = json.loads(line)
        assert message["jsonrpc"] == "2.0"
        assert "id" in message and ("result" in message) != ("error" in message)


def test_tools_list_exposes_exactly_two_tools(client):
    names = {t["name"] for t in client.request("tools/list", {})["result"]["tools"]}
    assert names == {"get_customer_record", "trigger_refund"}


def test_valid_calls_return_results(client):
    record = client.call("get_customer_record", {"customer_id": "CUST-00042"})
    assert record["result"]["isError"] is False
    assert json.loads(record["result"]["content"][0]["text"])["name"] == "Alan Turing"

    refund = client.call("trigger_refund", VALID_REFUND)
    assert refund["result"]["isError"] is False
    assert "CUST-00042" in refund["result"]["content"][0]["text"]


def test_invalid_params_is_protocol_error_with_matching_id(client):
    response = client.call("get_customer_record", {"customer_id": "CUST-42"})
    # request() already asserted the id matches.
    assert "result" not in response
    assert response["error"]["code"] == -32602


@pytest.mark.parametrize("customer_id", ["CUST-42", "cust-00042", "CUST-000421", 42])
def test_rejects_bad_customer_ids(client, customer_id):
    for tool, args in [
        ("get_customer_record", {"customer_id": customer_id}),
        ("trigger_refund", {**VALID_REFUND, "customer_id": customer_id}),
    ]:
        assert client.call(tool, args)["error"]["code"] == -32602


@pytest.mark.parametrize(
    "arguments",
    [
        {**VALID_REFUND, "amount": 0},
        {**VALID_REFUND, "amount": -5.0},
        {**VALID_REFUND, "amount": "25.5"},
        {**VALID_REFUND, "amount": float("nan")},  # json.dumps emits the NaN literal
        {**VALID_REFUND, "reason": "too short"},
        {"customer_id": "CUST-00042", "amount": 25.5},
        {**VALID_REFUND, "extra": "field"},
    ],
    ids=["zero", "negative", "string", "nan", "short_reason", "missing_field", "extra_field"],
)
def test_rejects_bad_refund_args(client, arguments):
    assert client.call("trigger_refund", arguments)["error"]["code"] == -32602


def test_rejects_extra_field_on_get_customer_record(client):
    response = client.call("get_customer_record", {"customer_id": "CUST-00042", "x": 1})
    assert response["error"]["code"] == -32602


@pytest.mark.parametrize(
    "line, code",
    [("{not json", -32700), ('{"jsonrpc": "2.0", "id": 9}', -32600), ("[1, 2]", -32600)],
    ids=["malformed_json", "no_method", "not_an_object"],
)
def test_undecodable_message_gets_error_with_null_id(client, line, code):
    client.send_raw(line)
    response = client.read()
    assert response == {"jsonrpc": "2.0", "id": None, "error": response["error"]}
    assert response["error"]["code"] == code
    # The server keeps serving afterwards.
    assert "result" in client.request("tools/list", {})
