# Reference Solution: 4-Task Assessment

Python 3.12. Setup and tests:

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pytest
```

## Task 1: MCP server with strict validation (stdio)

`task1_mcp_server/server.py` uses the official SDK's low-level `Server` with two handlers,
`tools/list` and `tools/call`, and exposes exactly `get_customer_record` and `trigger_refund`. Each
tool has a Pydantic model with `strict=True, extra="forbid"`: IDs must match `^CUST-[0-9]{5}$`,
`amount` must be a finite float `> 0` (strings and NaN are rejected), and `reason` needs at least 10
characters. A `ValidationError` is re-raised as `MCPError(-32602)`, so the SDK sends a JSON-RPC
error response with the request id, not a tool result with `isError`. The SDK silently drops
stdin lines it cannot decode, so a small relay between `stdio_server` and `server.run` answers
them with `id: null`: -32700 for malformed JSON, -32600 for JSON that is not a JSON-RPC message. Logging writes to stderr,
and the code has no `print()`.

Run: `python -m task1_mcp_server.server`

- **STDIO isolation**: `test_stdio_isolation_every_stdout_line_is_jsonrpc` spawns the server, sends
  initialize, valid calls and invalid calls, then checks that every stdout line is JSON-RPC 2.0.
- **Protocol compliance**: `test_invalid_params_is_protocol_error_with_matching_id` and
  `test_valid_calls_return_results`. Every request in the test client checks that the response id matches.
- **Validation edge cases**: parametrized tests for `CUST-42`, `cust-00042`, `CUST-000421`, a
  non-string ID, amount `0` / negative / string / NaN, a short reason, a missing field and an extra field.
  `test_undecodable_message_gets_error_with_null_id` covers malformed JSON and bad envelopes.

## Task 2: MCP security gateway proxy

`task2_mcp_gateway/gateway.py` is a FastAPI app with one `POST /` endpoint. It maps the bearer
token to a role through a static dict and returns 401 with a JSON-RPC error body if the token is
missing or unknown. It then parses the body (bad JSON gives -32700) and checks the JSON-RPC
structure (bad structure gives -32600). A `tools/call` whose `params.name` starts with `admin_` is
refused with -32001 "Unauthorized Tool Call" unless the role is admin. Everything else is forwarded
byte-for-byte with httpx, and the downstream response is returned unchanged. If downstream
cannot be reached or times out (10 s), the gateway returns HTTP 502 with JSON-RPC error -32002
"Downstream unavailable" and the original request id. A message with no `id` member is a notification (`"id": null` is
still a request): it goes through the same auth and admin checks but never gets a JSON-RPC body.
It returns 202 with an empty body, or the error's HTTP status (401, 502) with an empty body; a
denied notification is dropped without being forwarded.
`mock_downstream.py` answers `tools/list` and `tools/call` and keeps a list of every request it received.

Run: `uvicorn task2_mcp_gateway.mock_downstream:app --port 9001` and
`uvicorn task2_mcp_gateway.gateway:app --port 9000`. Tokens: `admin-token`, `viewer-token`.

- **Wire format**: tests for parse errors, seven invalid structures (-32600, nothing reaches
  downstream), and id preservation for int and string ids on forwarded, denied and invalid requests.
- **Forwarding**: `tools/list`, an allowed `tools/call` and an unrelated method (`ping`) all reach
  the mock, and the gateway returns the mock's response unchanged.
- **Downstream failures**: a connection error and a timeout each return the -32002 error with the
  original id.
- **Notifications**: a denied notification gets 202 with no body and never reaches downstream; an
  allowed one is forwarded and gets 202; a bad token gets 401 with no body; `"id": null` gets a response.
- **Authorization**: when a viewer calls `admin_reset_key`, the gateway returns the exact -32001
  error body, and the mock's received list is still empty. The same call as admin succeeds.

## Task 3: LLM gateway streaming PII guardrail

`task3_llm_guardrail/gateway.py` streams the upstream response with `httpx` `client.stream`. For each
SSE chunk it passes each choice's `delta.content` to that choice's own `StreamRedactor`
(keyed by choice index, so `n > 1` responses are covered) and sends on only the text that is
already safe. The redactor (`redactor.py`) finds the earliest position where a PII match could
still be forming: a regex anchored at the end of the buffer matches every possible prefix of an
email, SSN or card number. Text before that position is redacted and sent immediately; only the
possible partial match is held back. A complete match is never split. One character of released
context is kept so lookbehinds stay correct. Every pattern has a maximum length, so the held text
can never exceed `MAX_HOLDBACK` (2 × the longest match = 690 chars). The buffer is flushed and
redacted when the stream sends `finish_reason` or ends. The gateway opens the upstream stream
before answering, so an upstream error status (or an unreachable provider) becomes HTTP 502 with
`{"error": {"type": "upstream_error", ...}}` instead of a 200 stream. If the stream breaks or ends
without `[DONE]`, the gateway sends that error as a final SSE event, drops held text and never sends `[DONE]`. `mock_provider.py` streams OpenAI-style
chunks, with configurable `chunk_size` and `delay` between chunks.

Run: `uvicorn task3_llm_guardrail.mock_provider:app --port 8101` and
`uvicorn task3_llm_guardrail.gateway:app --port 8100`.

- **Chunking / buffer state**: an end-to-end test over real local servers gives byte-identical
  redacted output at chunk sizes 1, 3, 7 and the full message. Another test checks that held
  PII is flushed, redacted, at stream end. A three-choice stream is redacted per choice at chunk
  sizes 1 and 7.
- **Provider failures** (extra): an upstream 500 gives a 502 JSON error before any stream starts;
  a dropped connection or a stream without `[DONE]` ends with the error event and no `[DONE]`.
- **Partial streams**: each PII sample is split at every boundary position and always comes out as
  `[REDACTED]`. Non-PII lookalikes (order numbers, dates, phone numbers, `user@localhost`, 12-digit
  runs) come through unchanged.
- **Latency / bound**: the mock waits 100 ms between chunks, and the first safe token reaches the
  client at least 0.5 s before the upstream finishes. A test feeding 60k+ chars one at a time
  (long digit runs, long words, an unterminated email) checks that the held buffer stays within
  `MAX_HOLDBACK` after every feed.

## Task 4: Rate-limiting and model fallback router

`task4_model_router/router.py` follows the brief's requirement verbatim: "Estimate a request's tokens
as ceil(prompt_chars / 4) + max_tokens" (`prompt_chars` = total length of message contents). This is
approximate reservation accounting: the estimate is charged up front, before the provider answers,
and is not corrected afterwards (the brief forbids tiktoken, and provider-usage reconciliation is
not requested). `max_tokens` must
be at least 1, and the limiter refuses negative amounts, so usage can never be lowered to get
around the limit.
`rate_limiter.py` then opens its own aiosqlite connection to an on-disk database for each call and
runs `BEGIN IMMEDIATE`. Inside that transaction it deletes rows older than 60 s, sums the tenant's
usage, and inserts the new usage only if the total stays at or under 50,000. Because the lock is
taken at the start, concurrent requests check-and-record one at a time. The clock is injectable.
The router calls the primary inside `asyncio.wait_for(..., 3.0)`; a 429 or a timeout switches to
the backup. Every error goes through one `gateway_error()` helper (including the handlers for 404s
and unexpected exceptions), and provider bodies are never returned. `mock_provider.py` builds
providers whose delay, status code and error body can be changed at runtime.

Run: `uvicorn task4_model_router.mock_provider:primary --port 8201`,
`uvicorn task4_model_router.mock_provider:backup --port 8202`,
`uvicorn task4_model_router.router:app --port 8200`.

- **Concurrency and timeouts**: a primary delayed 3.5 s fails over to the backup. A primary at
  2.5 s answers itself, and the backup is never called. 25 concurrent 5,000-token requests produce
  exactly 10 × 200 and 15 × 429. This test fails if the transaction is removed.
- **Eviction and tracking**: with a fake clock, a full window still returns 429 at +59.9 s and is
  allowed again at +60 s. Hitting exactly 50,000 is allowed, and one more token is rejected. Tenants are isolated.
  A negative or zero `max_tokens` gets 400 and records no usage; the limiter raises on negative input.
- **Fallback and sanitization**: a primary 429 fails over. When the primary returns 429 and the
  backup returns 500 with a traceback, the client gets exactly the standardized 502 payload,
  and the test checks that no traceback, path, hostname, port or provider name leaked.

## Assumptions

- Task 1: undecodable lines are answered with `id: null`, since no id can be read from them.
- Task 1: a call to an unknown tool also returns -32602 (Invalid params).
- Task 1: strict mode still accepts a JSON integer for `amount`, because JSON makes no int/float distinction.
- Task 1: `get_customer_record` for a well-formed but unknown ID returns a normal "No customer" result.
- Task 2: the 401 body uses JSON-RPC code -32000 with `id: null` (auth is checked before parsing).
- Task 2: a `tools/call` without a string `params.name` is refused with -32602. Batch requests are not supported (-32600).
- Task 2: downstream URL, the 10 s downstream timeout, and the token map are constants in code.
- Task 3: an email requires a dot-separated TLD of 2–24 letters (so `user@localhost` is not redacted).
- Task 3: a card is 13–16 digits separated by at most one space or dash, with no digit directly
  before or after. There is no Luhn check.
- Task 3: only `delta.content` is redacted; other delta fields (such as tool calls) pass through.
- Task 3: on an interrupted stream, held (unreleased) text is discarded rather than flushed.
- Task 3: chunks without content (role, finish_reason) pass through. Held text is flushed before the finish chunk.
- Task 4: the API key is the full bearer token. `max_tokens` defaults to 256 when not given and must be >= 1.
- Task 4: only the primary fails over (on 429 or timeout). Other primary errors, and any backup
  error or timeout (also 3 s), return `upstream_unavailable` (502).
- Task 4: a request that would land exactly on 50,000 tokens is allowed. Rows with `ts <= now - 60` are evicted.
- Task 4: the database path defaults to `rate_limits.db` in the working directory.
