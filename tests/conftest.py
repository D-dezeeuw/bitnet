"""Test fixtures.

The whole suite runs against a stub of llama-server, so it needs neither the
1.2 GB GGUF nor a compiled backend and can run on any CI runner.
"""

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio

import app as app_module
from app import app as fastapi_app


class StubBackend:
    """Minimal stand-in for bitnet.cpp's llama-server.

    Records the payloads it receives so tests can assert on the exact prompt
    that was built, and can be told to stall so admission control is testable.
    """

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.content = "Hello there"
        self.stopped_limit = False
        self.tokens_predicted = 2
        self.tokens_evaluated = 5
        self.delay = 0.0
        self.status_code = 200
        self.unavailable = False

    @property
    def last_prompt(self) -> str | None:
        return self.requests[-1]["prompt"] if self.requests else None

    async def _sse(self) -> AsyncIterator[bytes]:
        for token in self.content.split(" "):
            chunk = {"content": token + " ", "stop": False}
            yield f"data: {json.dumps(chunk)}\n\n".encode()
        final = {"content": "", "stop": True, "stopped_limit": self.stopped_limit}
        yield f"data: {json.dumps(final)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    async def handler(self, request: httpx.Request) -> httpx.Response:
        if self.unavailable:
            raise httpx.ConnectError("stub backend is down", request=request)

        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})

        payload = json.loads(request.content)
        self.requests.append(payload)

        if self.delay:
            await asyncio.sleep(self.delay)

        if self.status_code != 200:
            return httpx.Response(self.status_code, text="backend failure")

        if payload.get("stream"):
            return httpx.Response(200, content=self._sse())

        return httpx.Response(
            200,
            json={
                "content": self.content,
                "tokens_predicted": self.tokens_predicted,
                "tokens_evaluated": self.tokens_evaluated,
                "stopped_limit": self.stopped_limit,
            },
        )


@pytest.fixture
def settings(monkeypatch):
    """Reset mutable settings between tests."""
    monkeypatch.setattr(app_module.settings, "api_key", None)
    monkeypatch.setattr(app_module.settings, "queue_timeout", 0.2)
    monkeypatch.setattr(app_module.settings, "role_stop_fallback", False)
    # Reset the sampler defaults too. settings is a module-level singleton, so a
    # test that assigns to it directly rather than through monkeypatch leaks the
    # value into every test that follows -- which is exactly what happened.
    monkeypatch.setattr(app_module.settings, "repeat_penalty", 1.1)
    monkeypatch.setattr(app_module.settings, "repeat_last_n", 64)
    return app_module.settings


@pytest.fixture
def backend() -> StubBackend:
    return StubBackend()


@pytest_asyncio.fixture
async def client(backend, settings) -> AsyncIterator[httpx.AsyncClient]:
    async with fastapi_app.router.lifespan_context(fastapi_app):
        # Swap the real backend client for one wired to the stub.
        await fastapi_app.state.client.aclose()
        fastapi_app.state.client = httpx.AsyncClient(
            transport=httpx.MockTransport(backend.handler),
            base_url="http://stub-backend",
        )
        transport = httpx.ASGITransport(app=fastapi_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as c:
            yield c


# The inference lock is created per-lifespan (inside the running loop), so each
# test gets a fresh one and no cross-test cleanup is needed.
