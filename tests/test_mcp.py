"""Tests for the /mcp endpoint.

These drive the real MCP JSON-RPC protocol against the mounted server rather
than calling the tool functions directly, so they cover the transport, the
auth wrapper and the tool schemas together. The backend is still the stub, so
no model is needed.
"""

import json

import pytest

MCP_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}

PROTOCOL_VERSION = "2025-06-18"


def rpc(method: str, params: dict | None = None, req_id: int = 1) -> dict:
    body: dict = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        body["params"] = params
    return body


def initialize() -> dict:
    return rpc(
        "initialize",
        {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0"},
        },
    )


def parse(response) -> dict:
    """Read a JSON-RPC result whether it arrived as JSON or as an SSE frame."""
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        for line in response.text.splitlines():
            if line.startswith("data: "):
                return json.loads(line[6:])
        raise AssertionError(f"no data frame in SSE response: {response.text!r}")
    return response.json()


@pytest.fixture
def keyed(settings):
    """The MCP endpoint refuses to serve without a configured key."""
    settings.api_key = "test-key-123"
    return settings.api_key


# --------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_key_in_query_string_is_accepted(client, keyed):
    """The whole point of the endpoint: a key supplied in the URL.

    Claude Desktop's connector dialog takes a URL and nothing else, so this is
    the only way a key can reach the server from that client.
    """
    r = await client.post(f"/mcp?key={keyed}", json=initialize(), headers=MCP_HEADERS)
    assert r.status_code == 200
    assert parse(r)["result"]["serverInfo"]["name"] == "bitnet"


@pytest.mark.asyncio
async def test_key_in_bearer_header_is_accepted(client, keyed):
    r = await client.post(
        "/mcp",
        json=initialize(),
        headers={**MCP_HEADERS, "Authorization": f"Bearer {keyed}"},
    )
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_missing_key_is_rejected(client, keyed):
    r = await client.post("/mcp", json=initialize(), headers=MCP_HEADERS)
    assert r.status_code == 401
    assert r.json()["error"]["code"] == -32001


@pytest.mark.asyncio
async def test_wrong_key_is_rejected(client, keyed):
    r = await client.post("/mcp?key=wrong", json=initialize(), headers=MCP_HEADERS)
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_key_prefix_is_not_enough(client, keyed):
    """A prefix of the real key must not authenticate."""
    r = await client.post(
        f"/mcp?key={keyed[:-1]}", json=initialize(), headers=MCP_HEADERS
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_endpoint_disabled_without_server_key(client, settings):
    """With no key configured the endpoint refuses to serve at all.

    An open MCP endpoint on a public host would let anyone spend the single
    inference slot, so this fails closed rather than open.
    """
    settings.api_key = None
    r = await client.post("/mcp", json=initialize(), headers=MCP_HEADERS)
    assert r.status_code == 503
    assert "BITNET_API_KEY" in r.json()["error"]["message"]


@pytest.mark.asyncio
async def test_url_encoded_key_is_decoded(client, settings):
    """Keys with URL-significant characters survive the query string."""
    settings.api_key = "a+b/c=d e"
    r = await client.post(
        "/mcp?key=a%2Bb%2Fc%3Dd+e", json=initialize(), headers=MCP_HEADERS
    )
    assert r.status_code == 200


# --------------------------------------------------------------------------
# Protocol
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tools_are_listed(client, keyed):
    await client.post(f"/mcp?key={keyed}", json=initialize(), headers=MCP_HEADERS)
    r = await client.post(
        f"/mcp?key={keyed}", json=rpc("tools/list", {}, 2), headers=MCP_HEADERS
    )
    assert r.status_code == 200
    names = {t["name"] for t in parse(r)["result"]["tools"]}
    assert names == {"bitnet_chat", "bitnet_status"}


@pytest.mark.asyncio
async def test_chat_tool_schema_declares_its_arguments(client, keyed):
    await client.post(f"/mcp?key={keyed}", json=initialize(), headers=MCP_HEADERS)
    r = await client.post(
        f"/mcp?key={keyed}", json=rpc("tools/list", {}, 2), headers=MCP_HEADERS
    )
    chat = next(t for t in parse(r)["result"]["tools"] if t["name"] == "bitnet_chat")
    props = chat["inputSchema"]["properties"]
    assert set(props) >= {"prompt", "system", "max_tokens", "temperature"}
    assert chat["inputSchema"]["required"] == ["prompt"]


# --------------------------------------------------------------------------
# Tool behaviour
# --------------------------------------------------------------------------


async def call_tool(client, key: str, name: str, arguments: dict) -> dict:
    await client.post(f"/mcp?key={key}", json=initialize(), headers=MCP_HEADERS)
    r = await client.post(
        f"/mcp?key={key}",
        json=rpc("tools/call", {"name": name, "arguments": arguments}, 3),
        headers=MCP_HEADERS,
    )
    assert r.status_code == 200, r.text
    return parse(r)


@pytest.mark.asyncio
async def test_chat_tool_returns_backend_content(client, keyed, backend):
    backend.content = "Generated reply"
    result = await call_tool(client, keyed, "bitnet_chat", {"prompt": "Hello"})
    assert "Generated reply" in json.dumps(result["result"])


@pytest.mark.asyncio
async def test_chat_tool_builds_the_same_prompt_as_the_rest_api(
    client, keyed, backend
):
    """The tool must not become a second prompt-construction path.

    Prompt format has drifted here before; this pins the tool to the same
    template, including the trailing generation prompt.
    """
    await call_tool(client, keyed, "bitnet_chat", {"prompt": "Hi"})
    prompt = backend.last_prompt
    assert "User: Hi<|eot_id|>" in prompt
    assert prompt.endswith("Assistant: ")


@pytest.mark.asyncio
async def test_chat_tool_passes_a_system_message_through(client, keyed, backend):
    await call_tool(
        client, keyed, "bitnet_chat", {"prompt": "Hi", "system": "Be terse."}
    )
    assert "Be terse." in backend.last_prompt


@pytest.mark.asyncio
async def test_chat_tool_honours_stop_tokens(client, keyed, backend):
    await call_tool(client, keyed, "bitnet_chat", {"prompt": "Hi"})
    assert "<|eot_id|>" in backend.requests[-1]["stop"]


@pytest.mark.asyncio
async def test_truncated_reply_is_flagged_to_the_caller(client, keyed, backend):
    """A reply cut off by max_tokens must say so.

    Returning truncated text silently would have the calling model treat a
    fragment as a complete answer.
    """
    backend.stopped_limit = True
    result = await call_tool(
        client, keyed, "bitnet_chat", {"prompt": "Hi", "max_tokens": 5}
    )
    assert "truncated" in json.dumps(result["result"])


@pytest.mark.asyncio
async def test_oversized_max_tokens_is_an_error_not_a_crash(client, keyed):
    result = await call_tool(
        client, keyed, "bitnet_chat", {"prompt": "Hi", "max_tokens": 999_999}
    )
    assert result["result"]["isError"] is True


@pytest.mark.asyncio
async def test_backend_failure_surfaces_as_a_tool_error(client, keyed, backend):
    backend.unavailable = True
    result = await call_tool(client, keyed, "bitnet_chat", {"prompt": "Hi"})
    assert result["result"]["isError"] is True
    assert "unavailable" in json.dumps(result["result"]).lower()


@pytest.mark.asyncio
async def test_status_tool_reports_the_model_and_context(client, keyed):
    result = await call_tool(client, keyed, "bitnet_status", {})
    payload = json.dumps(result["result"])
    assert "bitnet-b1.58-2B-4T" in payload
    assert "4096" in payload


@pytest.mark.asyncio
async def test_chat_tool_releases_the_slot(client, keyed, backend):
    """A tool call must not leave the single inference slot held.

    The slot is shared with /v1/chat/completions, so a leak here would wedge
    the whole deployment, not just MCP.
    """
    await call_tool(client, keyed, "bitnet_chat", {"prompt": "Hi"})
    r = await client.get("/v1/status")
    assert r.json()["busy"] is False


@pytest.mark.asyncio
async def test_slot_is_released_after_a_backend_failure(client, keyed, backend):
    backend.unavailable = True
    await call_tool(client, keyed, "bitnet_chat", {"prompt": "Hi"})
    backend.unavailable = False
    r = await client.get("/v1/status")
    assert r.json()["busy"] is False


# --------------------------------------------------------------------------
# Failure isolation
#
# The MCP endpoint shares a process with the API: entrypoint.sh exits when
# uvicorn does, so anything fatal here restart-loops the container and takes
# /v1 and the UI down with it. These pin that /mcp can never do that.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kill_switch_disables_mcp_but_not_the_api(
    backend, settings, monkeypatch
):
    import httpx as _httpx

    import app as app_module

    monkeypatch.setattr(app_module.settings, "mcp_enabled", False)
    monkeypatch.setattr(app_module.settings, "api_key", "k")
    async with app_module.app.router.lifespan_context(app_module.app):
        await app_module.app.state.client.aclose()
        app_module.app.state.client = _httpx.AsyncClient(
            transport=_httpx.MockTransport(backend.handler),
            base_url="http://stub-backend",
        )
        transport = _httpx.ASGITransport(app=app_module.app)
        async with _httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as c:
            # The API keeps working...
            assert (await c.get("/health")).status_code == 200
            # ...and /mcp reports unavailable rather than hanging or crashing.
            r = await c.post(
                "/mcp?key=k", json=initialize(), headers=MCP_HEADERS
            )
            assert r.status_code == 503


@pytest.mark.asyncio
async def test_api_starts_even_if_the_session_manager_never_does(
    backend, settings, monkeypatch
):
    """A session manager that hangs must not block startup.

    The previous code did `await started.wait()` unbounded, so a manager that
    raised or stalled left uvicorn stuck in startup forever: no healthcheck, no
    logs, just a restart loop.
    """
    import httpx as _httpx

    import app as app_module

    monkeypatch.setattr(app_module.settings, "api_key", "k")
    monkeypatch.setattr(app_module, "MCP_STARTUP_TIMEOUT", 0.25)

    import contextlib

    class _StalledManager:
        @contextlib.asynccontextmanager
        async def run(self):
            # A real async context manager that fails on entry, so this
            # exercises the same path a genuinely broken session manager takes
            # rather than tripping over a malformed test double.
            raise RuntimeError("session manager exploded")
            yield  # pragma: no cover - unreachable, marks this a generator

    class _StalledServer:
        session_manager = _StalledManager()

        def streamable_http_app(self, **kw):
            return None

    monkeypatch.setattr(app_module, "build_mcp_server", lambda *a: _StalledServer())

    async with app_module.app.router.lifespan_context(app_module.app):
        await app_module.app.state.client.aclose()
        app_module.app.state.client = _httpx.AsyncClient(
            transport=_httpx.MockTransport(backend.handler),
            base_url="http://stub-backend",
        )
        transport = _httpx.ASGITransport(app=app_module.app)
        async with _httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as c:
            # Startup completed at all, which is the point.
            assert (await c.get("/health")).status_code == 200
            r = await c.post("/mcp?key=k", json=initialize(), headers=MCP_HEADERS)
            assert r.status_code == 503
