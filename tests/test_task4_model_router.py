"""Task 4: timeout races, concurrency at the limit, eviction, isolation, fallback, sanitization."""

import asyncio

import httpx
import pytest

from task4_model_router import mock_provider, router
from task4_model_router.rate_limiter import RateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def env(tmp_path):
    primary, backup = mock_provider.create_app("primary"), mock_provider.create_app("backup")
    clock = FakeClock()
    upstream = httpx.AsyncClient(mounts={
        "http://primary": httpx.ASGITransport(app=primary),
        "http://backup": httpx.ASGITransport(app=backup),
    })
    app = router.create_app(
        "http://primary/v1/chat/completions",
        "http://backup/v1/chat/completions",
        db_path=str(tmp_path / "usage.db"),
        clock=clock,
        http_client=upstream,
    )
    gateway = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://router", timeout=30)
    return gateway, primary, backup, clock


def chat(gateway, key="tenant-a", max_tokens=100, prompt="hello"):
    return gateway.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens},
    )


# --- Async concurrency and timeout races ----------------------------------

async def test_primary_delayed_past_timeout_fails_over(env):
    gateway, primary, backup, _ = env
    primary.state.delay = 3.5
    r = await chat(gateway)
    assert r.status_code == 200 and r.json()["model"] == "backup"


async def test_primary_within_timeout_does_not_fail_over(env):
    gateway, primary, backup, _ = env
    primary.state.delay = 2.5
    r = await chat(gateway)
    assert r.status_code == 200 and r.json()["model"] == "primary"
    assert backup.state.calls == 0


async def test_concurrent_requests_at_limit_admit_exactly_allowed(env):
    gateway, *_ = env
    # Each request costs ceil(0/4) + 5000 = 5000 tokens, so exactly 10 of 25 fit in 50,000.
    responses = await asyncio.gather(*(chat(gateway, max_tokens=5000, prompt="") for _ in range(25)))
    codes = [r.status_code for r in responses]
    assert codes.count(200) == 10 and codes.count(429) == 15


# --- Rate-limiter eviction and token tracking -----------------------------

async def test_token_estimate_counts_prompt_chars_and_max_tokens(env):
    gateway, *_ = env
    # ceil(9 / 4) + 49_997 = 50_000: exactly at the limit, so allowed; one more token is not.
    assert (await chat(gateway, prompt="x" * 9, max_tokens=49_997)).status_code == 200
    assert (await chat(gateway, prompt="", max_tokens=1)).status_code == 429


@pytest.mark.parametrize("max_tokens", [-100_000, 0])
async def test_non_positive_max_tokens_rejected_and_not_counted(env, max_tokens):
    gateway, *_ = env
    r = await chat(gateway, max_tokens=max_tokens)
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request"
    # No negative usage was recorded, so the limit still applies normally.
    assert (await chat(gateway, prompt="", max_tokens=50_000)).status_code == 200
    assert (await chat(gateway, prompt="", max_tokens=1)).status_code == 429


async def test_limiter_refuses_negative_consumption(tmp_path):
    with pytest.raises(ValueError):
        await RateLimiter(str(tmp_path / "usage.db")).try_consume("tenant-a", -1)


async def test_usage_expires_after_window(env):
    gateway, _, _, clock = env
    assert (await chat(gateway, prompt="", max_tokens=50_000)).status_code == 200
    clock.now += 59.9
    assert (await chat(gateway, prompt="", max_tokens=1)).status_code == 429
    clock.now += 0.1
    assert (await chat(gateway, prompt="", max_tokens=50_000)).status_code == 200


async def test_tenants_are_isolated(env):
    gateway, *_ = env
    assert (await chat(gateway, key="tenant-a", prompt="", max_tokens=50_000)).status_code == 200
    assert (await chat(gateway, key="tenant-a", prompt="", max_tokens=1)).status_code == 429
    assert (await chat(gateway, key="tenant-b", prompt="", max_tokens=50_000)).status_code == 200


async def test_rate_limited_response_is_standardized(env):
    gateway, *_ = env
    r = await chat(gateway, max_tokens=50_001)
    assert r.status_code == 429
    assert r.json()["error"]["type"] == "gateway_error" and r.json()["error"]["code"] == "rate_limited"


# --- Fallback and sanitization --------------------------------------------

async def test_primary_429_fails_over(env):
    gateway, primary, backup, _ = env
    primary.state.status = 429
    r = await chat(gateway)
    assert r.status_code == 200 and r.json()["model"] == "backup"


async def test_both_failing_yields_sanitized_standard_error(env):
    gateway, primary, backup, _ = env
    primary.state.status, primary.state.body = 429, "quota exceeded on primary-llm.internal.corp"
    backup.state.status = 500
    backup.state.body = (
        'Traceback (most recent call last):\n  File "/srv/backup/app.py", line 42\n'
        "ConnectionError: db-01.internal:5432 refused"
    )
    r = await chat(gateway)
    assert r.status_code == 502
    assert r.json() == {"error": {
        "type": "gateway_error",
        "code": "upstream_unavailable",
        "message": "The model provider is unavailable.",
    }}
    for leaked in ("Traceback", "app.py", "internal", "5432", "primary", "backup", "quota"):
        assert leaked not in r.text


async def test_missing_api_key_is_standardized(env):
    gateway, *_ = env
    r = await gateway.post("/v1/chat/completions", json={"messages": []})
    assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized"
