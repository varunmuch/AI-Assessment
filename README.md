# Reference Solution: 4-Task Assessment

Python 3.12. Install and run all tests:

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pytest
```

## Task 1: MCP server with strict validation

**Design.** [server.py](task1_mcp_server/server.py) uses the official SDK's low-level `Server`
and exposes exactly two tools. Each tool's arguments are checked by a Pydantic model with
`strict=True, extra="forbid"`. If validation fails, the server raises `MCPError(-32602)`, which
the SDK sends as a JSON-RPC error rather than a tool result. All logs go to stderr, and there is
no `print()`.

```bash
python -m task1_mcp_server.server
```

**Evaluation criteria**
- **STDIO isolation**: a test starts the server as a subprocess, sends initialize plus valid and
  invalid calls, and checks that every stdout line is a JSON-RPC 2.0 message.
- **Protocol compliance**: invalid params return -32602 with the matching id; valid calls return results.
- **Validation edge cases**: tests reject `CUST-42`, `cust-00042`, `CUST-000421`, a numeric ID,
  an amount of 0, a negative or string amount, NaN, a reason under 10 characters, and missing or
  extra fields.

**Beyond the brief**
- The SDK silently drops input lines it can't decode. A small relay answers them with `id: null`:
  -32700 for malformed JSON, -32600 for JSON that isn't a JSON-RPC message.

## Task 2: MCP security gateway proxy

**Design.** [gateway.py](task2_mcp_gateway/gateway.py) is a FastAPI app with a single `POST /`
endpoint. It looks up the bearer token's role in a static map. A `tools/call` for a tool whose name
starts with `admin_` is refused unless the role is admin. Every other request is forwarded unchanged
with httpx, and the downstream response is returned unchanged.
[mock_downstream.py](task2_mcp_gateway/mock_downstream.py) answers `tools/list` and `tools/call`
and records every request it receives.

```bash
uvicorn task2_mcp_gateway.mock_downstream:app --port 9001
uvicorn task2_mcp_gateway.gateway:app --port 9000      # tokens: admin-token, viewer-token
```

**Evaluation criteria**
- **Wire format**: malformed JSON returns -32700, and seven kinds of invalid structure return -32600
  without reaching downstream. Integer and string ids are preserved in every response.
- **Forwarding**: `tools/list`, an allowed `tools/call`, and another method (`ping`) all reach the
  mock, and the gateway returns the mock's responses unchanged.
- **Authorization**: a viewer calling `admin_reset_key` gets `-32001 "Unauthorized Tool Call"`, and
  the mock confirms it never received the call. An admin making the same call succeeds. A missing or
  unknown token gets HTTP 401 with a JSON-RPC error body.

**Beyond the brief**
- **Downstream failure**: if downstream is unreachable or times out (10 s), the gateway returns
  HTTP 502 with error `-32002 "Downstream unavailable"` and the original id.
- **Notifications** (no `id` member): they pass the same checks but never get a JSON-RPC body. The
  gateway returns 202 (or 401 / 502) with an empty body. Denied notifications are dropped, not
  forwarded. A request with `"id": null` is a request, not a notification, and gets a response.

## Task 3: Streaming PII guardrail

**Design.** [gateway.py](task3_llm_guardrail/gateway.py) streams the provider's SSE response with
httpx and passes each chunk's text through a [StreamRedactor](task3_llm_guardrail/redactor.py).
The redactor sends text as soon as it can no longer be part of an email, SSN or card number. It
holds back only a possible partial match at the end of the buffer, and never splits a complete
match. Every pattern has a maximum length, so held text never exceeds `MAX_HOLDBACK` (690 chars).
Held text is redacted and flushed when the stream finishes. Each completion choice gets its own
redactor. [mock_provider.py](task3_llm_guardrail/mock_provider.py) streams OpenAI-style chunks,
with a configurable chunk size and delay.

```bash
uvicorn task3_llm_guardrail.mock_provider:app --port 8101
uvicorn task3_llm_guardrail.gateway:app --port 8100
```

**Evaluation criteria**
- **Chunking and buffer state**: end-to-end output is identical at chunk sizes 1, 3, 7 and the full
  message. Held PII is redacted when it is flushed at stream end.
- **Partial streams**: every PII sample is split at every possible position and is always
  redacted. Look-alikes (order numbers, dates, phone numbers, `user@localhost`) pass through unchanged.
- **Low latency and bounded buffer**: with a 100 ms delay between chunks, the first safe token reaches
  the client well before the provider finishes (tested on real local servers, because httpx's
  in-process transport buffers whole responses). When 60,000+ characters are fed one at a time, held
  text stays within `MAX_HOLDBACK` after every feed.

**Beyond the brief**
- **Multiple choices**: when a response has `n > 1` choices, each choice is redacted separately.
- **Provider failures**: an upstream error status returns HTTP 502 before streaming starts. A stream
  that breaks or ends without `[DONE]` ends with an error event instead of `[DONE]`.

## Task 4: Rate-limiting and model fallback router

**Design.** [router.py](task4_model_router/router.py) charges each request using the brief's
formula, quoted exactly: "Estimate a request's tokens as ceil(prompt_chars / 4) + max_tokens".
The charge is reserved up front and not corrected afterwards: the brief forbids tiktoken and
doesn't ask for reconciliation. [rate_limiter.py](task4_model_router/rate_limiter.py) opens one
SQLite connection per check and runs `BEGIN IMMEDIATE`. Inside that transaction it deletes rows
older than 60 s, sums the tenant's usage, and records the request only if the total stays at or
under 50,000. The primary is called inside `asyncio.wait_for(..., 3.0)`. A 429 or a timeout
switches to the backup. Every error uses one standardized payload, and provider error bodies are
never passed to the client.

```bash
uvicorn task4_model_router.mock_provider:primary --port 8201
uvicorn task4_model_router.mock_provider:backup --port 8202
uvicorn task4_model_router.router:app --port 8200
```

**Evaluation criteria**
- **Concurrency and timeout races**: a primary delayed 3.5 s fails over; at 2.5 s it answers and the
  backup is never called. 25 concurrent 5,000-token requests produce exactly 10 successes and
  15 × 429 (the test fails if the transaction is removed).
- **Eviction and token tracking**: with an injected clock, a full window still returns 429 at
  +59.9 s and is open again at +60 s. Exactly 50,000 tokens is allowed; one more is rejected.
  Tenants are tracked separately.
- **Fallback and sanitization**: a primary 429 fails over. When the primary returns 429 and the backup
  returns 500 with a stack trace, the client gets only the standard 502 payload, with no traceback,
  path, hostname, port or provider name.

**Beyond the brief**
- **Non-positive `max_tokens`** is rejected with 400, and the limiter refuses negative amounts.

## Assumptions

**Task 1**
- A call to an unknown tool returns -32602.
- `amount` accepts JSON integers (JSON doesn't distinguish ints from floats).
- A well-formed but unknown customer ID returns a normal "No customer" result.

**Task 2**
- The 401 body uses code -32000 and `id: null`. A token error takes priority over parse errors.
- A `tools/call` without a string `params.name` returns -32602. Batch requests return -32600.
- The downstream URL, the timeout and the token map are constants in code.

**Task 3**
- An email needs a TLD of 2–24 letters, so `user@localhost` is not redacted.
- A card number is 13–16 digits with optional single spaces or dashes; there is no Luhn check.
- Only `delta.content` is redacted. Other fields, such as tool calls, pass through unchanged.
- If a stream is interrupted, held text is dropped rather than flushed.

**Task 4**
- The tenant key is the full bearer token. `max_tokens` defaults to 256 and must be at least 1.
  The database file defaults to `rate_limits.db`.
- Only the primary fails over. Other provider errors, including a backup timeout (also 3 s),
  return `upstream_unavailable` (502). Rows with `ts <= now - 60` are evicted.
