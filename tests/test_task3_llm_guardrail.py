"""Task 3: chunking/buffer state, partial-stream matching, latency and buffer bound."""

import json
import time

import httpx
import pytest

from task3_llm_guardrail import gateway, mock_provider
from task3_llm_guardrail.redactor import MAX_HOLDBACK, REDACTED, StreamRedactor
from tests.conftest import serve

TEXT = (
    "Hi! Email jane.doe@example.com or bob+news@mail.co.uk. SSN 123-45-6789. "
    "Cards: 4111 1111 1111 1111, 4111-1111-1111-1111 and 4111111111111. "
    "Order 12345 ships 2024-06-01; call 555-1234. Bye."
)
EXPECTED = (
    f"Hi! Email {REDACTED} or {REDACTED}. SSN {REDACTED}. "
    f"Cards: {REDACTED}, {REDACTED} and {REDACTED}. "
    "Order 12345 ships 2024-06-01; call 555-1234. Bye."
)
PII_SAMPLES = ["jane.doe@example.com", "123-45-6789", "4111 1111 1111 1111", "4111-1111-1111-1111", "4111111111111"]


def stream_redact(text: str, chunk_size: int) -> str:
    r = StreamRedactor()
    out = "".join(r.feed(text[i : i + chunk_size]) for i in range(0, len(text), chunk_size))
    return out + r.flush()


async def collect(url: str) -> list[tuple[float, int, str]]:
    """Return (arrival_time, choice_index, content) for each content delta from the gateway."""
    events = []
    async with httpx.AsyncClient(timeout=10) as client:
        async with client.stream("POST", f"{url}/v1/chat/completions", json={"messages": []}) as r:
            async for line in r.aiter_lines():
                if line.startswith("data: {"):
                    for choice in json.loads(line[6:])["choices"]:
                        if content := choice["delta"].get("content"):
                            events.append((time.monotonic(), choice["index"], content))
    return events


# --- Stream chunking and buffer state ------------------------------------

@pytest.mark.parametrize("chunk_size", [1, 3, 7, len(TEXT)])
async def test_identical_output_at_every_chunk_size_end_to_end(chunk_size):
    with serve(mock_provider.create_app(TEXT, chunk_size)) as provider_url:
        with serve(gateway.create_app(f"{provider_url}/v1/chat/completions")) as url:
            events = await collect(url)
    assert "".join(content for _, _, content in events) == EXPECTED


@pytest.mark.parametrize("chunk_size", [1, 7])
async def test_every_completion_choice_is_redacted(chunk_size):
    texts = [TEXT, "Second choice: mail ops@corp.example.org, SSN 987-65-4321.", "clean text"]
    with serve(mock_provider.create_app(texts, chunk_size)) as provider_url:
        with serve(gateway.create_app(f"{provider_url}/v1/chat/completions")) as url:
            events = await collect(url)
    per_choice = ["".join(c for _, i, c in events if i == index) for index in range(3)]
    assert per_choice == [EXPECTED, f"Second choice: mail {REDACTED}, SSN {REDACTED}.", "clean text"]


def test_buffer_flushed_redacted_at_stream_end():
    r = StreamRedactor()
    assert r.feed("ends with 123-45-6789") == "ends with "
    assert r.flush() == REDACTED


# --- Pattern matching over partial streams --------------------------------

@pytest.mark.parametrize("pii", PII_SAMPLES)
def test_pii_split_at_every_boundary_is_redacted(pii):
    text = f"before {pii} after"
    for split in range(1, len(text)):
        r = StreamRedactor()
        out = r.feed(text[:split]) + r.feed(text[split:]) + r.flush()
        assert out == f"before {REDACTED} after", split


def test_non_pii_text_untouched():
    text = "Order 12345, ref 2024-06-01, phone 555-1234, user@localhost, 123456789012 digits."
    for size in (1, 3, 7, len(text)):
        assert stream_redact(text, size) == text


# --- Low latency and bounded buffer ---------------------------------------

async def test_first_safe_token_arrives_before_upstream_finishes():
    provider = mock_provider.create_app("Hello there, " + "word " * 10, chunk_size=6, delay=0.1)
    with serve(provider) as provider_url:
        with serve(gateway.create_app(f"{provider_url}/v1/chat/completions")) as url:
            events = await collect(url)
    first_token_at, _, first = events[0]
    assert first == "Hello "
    assert first_token_at < provider.state.finished_at - 0.5


def test_buffer_never_exceeds_fixed_bound():
    long_text = ("9" * 5000 + " " + "x" * 5000 + " " + "1 2 3 " * 2000 + "a@" + "b" * 5000) * 3
    r = StreamRedactor()
    for ch in long_text:
        r.feed(ch)
        assert len(r._held) <= MAX_HOLDBACK


# --- Provider failures -----------------------------------------------------

async def test_upstream_error_status_is_reported_before_streaming():
    with serve(mock_provider.create_app(status=500)) as provider_url:
        with serve(gateway.create_app(f"{provider_url}/v1/chat/completions")) as url:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.post(f"{url}/v1/chat/completions", json={"messages": []})
    assert r.status_code == 502
    assert r.json()["error"]["type"] == "upstream_error"
    assert "exploded" not in r.text


@pytest.mark.parametrize("failure", [{"fail_after": 3}, {"send_done": False}], ids=["dropped", "no_done"])
async def test_interrupted_stream_is_not_reported_as_complete(failure):
    with serve(mock_provider.create_app(TEXT, chunk_size=5, **failure)) as provider_url:
        with serve(gateway.create_app(f"{provider_url}/v1/chat/completions")) as url:
            async with httpx.AsyncClient(timeout=10) as client:
                async with client.stream("POST", f"{url}/v1/chat/completions", json={"messages": []}) as r:
                    lines = [line async for line in r.aiter_lines() if line.startswith("data: ")]
    assert "data: [DONE]" not in lines
    assert json.loads(lines[-1][6:])["error"]["type"] == "upstream_error"
