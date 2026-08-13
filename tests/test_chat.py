"""Completion behaviour: finish_reason, streaming, stops, error mapping."""

import json


def body(**kw):
    payload = {"messages": [{"role": "user", "content": "hi"}]}
    payload.update(kw)
    return payload


class TestFinishReason:
    """Non-streaming used to hardcode "stop" even when the reply was truncated."""

    async def test_stop_when_backend_stopped_naturally(self, client, backend):
        backend.stopped_limit = False
        resp = await client.post("/v1/chat/completions", json=body())
        assert resp.json()["choices"][0]["finish_reason"] == "stop"

    async def test_length_when_backend_hit_the_token_limit(self, client, backend):
        backend.stopped_limit = True
        resp = await client.post("/v1/chat/completions", json=body())
        assert resp.json()["choices"][0]["finish_reason"] == "length"


async def test_usage_is_reported_from_the_backend(client, backend):
    backend.tokens_evaluated = 11
    backend.tokens_predicted = 7
    usage = (await client.post("/v1/chat/completions", json=body())).json()["usage"]
    assert usage == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
    }


async def test_response_shape_is_openai_compatible(client):
    data = (await client.post("/v1/chat/completions", json=body())).json()
    assert data["object"] == "chat.completion"
    assert data["id"].startswith("chatcmpl-")
    assert data["choices"][0]["message"]["role"] == "assistant"


class TestStops:
    async def test_eot_is_always_a_stop(self, client, backend):
        """The GGUF declares eos as <|end_of_text|>, so llama-server never stops
        on <|eot_id|> by itself - it has to be listed."""
        await client.post("/v1/chat/completions", json=body())
        assert "<|eot_id|>" in backend.requests[-1]["stop"]

    async def test_role_string_stops_are_off_by_default(self, client, backend):
        await client.post("/v1/chat/completions", json=body())
        assert "User:" not in backend.requests[-1]["stop"]

    async def test_role_string_stops_can_be_re_enabled(self, client, backend, settings):
        settings.role_stop_fallback = True
        await client.post("/v1/chat/completions", json=body())
        assert "User:" in backend.requests[-1]["stop"]

    async def test_dead_llama2_token_is_gone(self, client, backend):
        await client.post("/v1/chat/completions", json=body())
        assert "</s>" not in backend.requests[-1]["stop"]

    async def test_client_stops_are_forwarded(self, client, backend):
        await client.post("/v1/chat/completions", json=body(stop=["END"]))
        assert "END" in backend.requests[-1]["stop"]


class TestForwardedParameters:
    """These used to be accepted and silently discarded."""

    async def test_top_p_forwarded(self, client, backend):
        await client.post("/v1/chat/completions", json=body(top_p=0.5))
        assert backend.requests[-1]["top_p"] == 0.5

    async def test_penalties_forwarded(self, client, backend):
        await client.post(
            "/v1/chat/completions",
            json=body(presence_penalty=0.4, frequency_penalty=-0.2),
        )
        assert backend.requests[-1]["presence_penalty"] == 0.4
        assert backend.requests[-1]["frequency_penalty"] == -0.2

    async def test_continuation_changes_the_prompt(self, client, backend):
        await client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "partial"},
                ],
                "continuation": True,
            },
        )
        assert backend.last_prompt == "User: hi<|eot_id|>Assistant: partial"


class TestStreaming:
    async def _frames(self, client, **kw):
        frames = []
        async with client.stream(
            "POST", "/v1/chat/completions", json=body(stream=True, **kw)
        ) as resp:
            assert resp.status_code == 200
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    frames.append(line[6:])
        return frames

    async def test_terminates_with_done(self, client):
        frames = await self._frames(client)
        assert frames[-1] == "[DONE]"

    async def test_chunks_are_openai_shaped(self, client):
        frames = await self._frames(client)
        first = json.loads(frames[0])
        assert first["object"] == "chat.completion.chunk"
        assert "content" in first["choices"][0]["delta"]

    async def test_content_streams_through(self, client, backend):
        backend.content = "alpha beta gamma"
        frames = await self._frames(client)
        text = "".join(
            json.loads(f)["choices"][0]["delta"].get("content", "")
            for f in frames
            if f != "[DONE]"
        )
        assert text.strip() == "alpha beta gamma"

    async def test_truncation_reported_as_length(self, client, backend):
        backend.stopped_limit = True
        frames = await self._frames(client)
        reasons = [
            json.loads(f)["choices"][0]["finish_reason"]
            for f in frames
            if f != "[DONE]"
        ]
        assert "length" in reasons

    async def test_slot_is_released_after_streaming(self, client):
        await self._frames(client)
        assert (await client.get("/v1/status")).json()["busy"] is False


class TestBackendErrors:
    async def test_backend_failure_maps_to_502(self, client, backend):
        backend.status_code = 500
        resp = await client.post("/v1/chat/completions", json=body())
        assert resp.status_code == 502

    async def test_backend_down_maps_to_503(self, client, backend):
        backend.unavailable = True
        resp = await client.post("/v1/chat/completions", json=body())
        assert resp.status_code == 503

    async def test_backend_text_is_not_leaked_to_the_caller(self, client, backend):
        backend.status_code = 500
        resp = await client.post("/v1/chat/completions", json=body())
        assert "backend failure" not in resp.text

    async def test_slot_released_after_backend_error(self, client, backend):
        backend.status_code = 500
        await client.post("/v1/chat/completions", json=body())
        assert (await client.get("/v1/status")).json()["busy"] is False


async def test_summarize_returns_a_summary(client, backend):
    backend.content = "  A tidy summary.  "
    resp = await client.post(
        "/v1/summarize", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert resp.status_code == 200
    assert resp.json()["summary"] == "A tidy summary."


class TestRepetitionPenalty:
    """llama-server defaults repeat_penalty to 1.0 (disabled), which made the
    model restate the same sentence until it hit n_predict. These pin the
    penalty as always-sent rather than optional."""

    async def test_penalty_is_always_sent(self, client, backend):
        await client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        payload = backend.requests[-1]
        assert payload["repeat_penalty"] == 1.1
        assert payload["repeat_last_n"] == 64

    async def test_penalty_is_never_the_disabling_default(self, client, backend):
        """1.0 is the value that caused the looping; the default must not be it."""
        await client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert backend.requests[-1]["repeat_penalty"] > 1.0

    async def test_request_can_override_the_penalty(self, client, backend):
        await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "repeat_penalty": 1.3,
                "repeat_last_n": 128,
            },
        )
        assert backend.requests[-1]["repeat_penalty"] == 1.3
        assert backend.requests[-1]["repeat_last_n"] == 128

    async def test_caller_can_opt_out_with_one(self, client, backend):
        """1.0 is a legitimate explicit choice even though it is a bad default."""
        await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "repeat_penalty": 1.0},
        )
        assert backend.requests[-1]["repeat_penalty"] == 1.0

    async def test_settings_drive_the_default(
        self, client, backend, settings, monkeypatch
    ):
        # monkeypatch, not direct assignment: settings is a singleton and a raw
        # assignment here leaked 1.25 into every later test.
        monkeypatch.setattr(settings, "repeat_penalty", 1.25)
        await client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert backend.requests[-1]["repeat_penalty"] == 1.25

    async def test_below_one_is_rejected(self, client):
        """Under 1.0 rewards repetition, which is never what a caller wants."""
        r = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "repeat_penalty": 0.5},
        )
        assert r.status_code == 422

    async def test_mcp_tool_gets_the_penalty_too(
        self, client, backend, settings, monkeypatch
    ):
        """The MCP path builds its payload through the same helper, so it must
        inherit this rather than looping where the REST path does not."""
        monkeypatch.setattr(settings, "api_key", "k")
        init = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                           "clientInfo": {"name": "t", "version": "1"}}}
        h = {"Content-Type": "application/json",
             "Accept": "application/json, text/event-stream"}
        await client.post("/mcp?key=k", json=init, headers=h)
        await client.post("/mcp?key=k", headers=h, json={
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "bitnet_chat", "arguments": {"prompt": "hi"}}})
        assert backend.requests[-1]["repeat_penalty"] == 1.1
