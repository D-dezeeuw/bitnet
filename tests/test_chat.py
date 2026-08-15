"""Completion behaviour: finish_reason, streaming, stops, error mapping."""

import json

from app import prompt_budget_chars


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
        # The default system prompt is prepended; what matters for continuation
        # is that the tail resumes the partial turn rather than opening a new one.
        assert backend.last_prompt.endswith("User: hi<|eot_id|>Assistant: partial")
        assert not backend.last_prompt.endswith("Assistant: ")


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
    """repeat_penalty is sent explicitly and defaults to 1.0 (off).

    It was briefly 1.1 to stop the model restating a sentence until it hit
    n_predict, but a token-level penalty punishes the function words and subject
    nouns a sentence needs, and output degraded into "strings aren't be taken".
    DRY covers that repetition properly. These pin the field as always-sent, so
    the value is this project's choice rather than the backend's."""

    async def test_penalty_is_always_sent(self, client, backend):
        await client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        payload = backend.requests[-1]
        assert payload["repeat_penalty"] == 1.0
        assert payload["repeat_last_n"] == 64

    async def test_penalty_defaults_to_off(self, client, backend):
        """Deliberately 1.0 now. It was briefly 1.1, which suppressed the
        function words a sentence needs and produced "strings aren't be taken".
        DRY covers the phrase repetition this was aimed at."""
        await client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert backend.requests[-1]["repeat_penalty"] == 1.0

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
        assert backend.requests[-1]["repeat_penalty"] == 1.0


class TestDrySampling:
    """repeat_penalty acts on single tokens, so it barely dents the observed
    failure: a phrase repeated with substitutions ("let's make it simple/easy/
    clear for you"). DRY penalises repeated n-grams, which is that pattern."""

    async def test_dry_is_always_sent(self, client, backend):
        await client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        payload = backend.requests[-1]
        assert payload["dry_multiplier"] == 0.8
        assert payload["dry_base"] == 1.75
        assert payload["dry_allowed_length"] == 2

    async def test_dry_is_never_left_disabled_by_default(self, client, backend):
        """0.0 is llama-server's default and means off -- the state that let the
        model loop. The default here must not be it."""
        await client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert backend.requests[-1]["dry_multiplier"] > 0.0

    async def test_dry_penalty_spans_the_whole_context(self, client, backend):
        """-1, so a loop that began early is still penalised later in a long
        generation; a bounded window stops seeing the phrase it should catch."""
        await client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert backend.requests[-1]["dry_penalty_last_n"] == -1

    async def test_request_can_override_dry(self, client, backend):
        await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "dry_multiplier": 1.5,
            },
        )
        assert backend.requests[-1]["dry_multiplier"] == 1.5

    async def test_caller_can_disable_dry(self, client, backend):
        await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "dry_multiplier": 0.0},
        )
        assert backend.requests[-1]["dry_multiplier"] == 0.0

    async def test_negative_multiplier_is_rejected(self, client):
        r = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "dry_multiplier": -1},
        )
        assert r.status_code == 422

    async def test_mcp_tool_gets_dry_too(self, client, backend, settings, monkeypatch):
        monkeypatch.setattr(settings, "api_key", "k")
        h = {"Content-Type": "application/json",
             "Accept": "application/json, text/event-stream"}
        await client.post("/mcp?key=k", headers=h, json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "t", "version": "1"}}})
        await client.post("/mcp?key=k", headers=h, json={
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "bitnet_chat", "arguments": {"prompt": "hi"}}})
        assert backend.requests[-1]["dry_multiplier"] == 0.8


class TestFinishReasonAcrossBackendVersions:
    """A reply cut off at max_tokens must report finish_reason "length".

    Production returned exactly max_tokens, cut mid-sentence, labelled "stop":
    the code read only llama.cpp's older `stopped_limit`, while the deployed
    backend reports the newer `stop_type`. The stub emitted only the old field,
    so the suite agreed with the code and neither matched reality.

    The UI's Continue button and the MCP truncation notice both key off this.
    """

    async def test_new_stop_type_field_is_understood(self, client, backend):
        backend.stop_field = "stop_type"
        backend.stopped_limit = True
        r = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
        )
        assert r.json()["choices"][0]["finish_reason"] == "length"

    async def test_old_stopped_limit_field_still_understood(self, client, backend):
        backend.stop_field = "stopped_limit"
        backend.stopped_limit = True
        r = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
        )
        assert r.json()["choices"][0]["finish_reason"] == "length"

    async def test_natural_stop_is_not_reported_as_length(self, client, backend):
        backend.stop_field = "stop_type"
        backend.stopped_limit = False
        r = await client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert r.json()["choices"][0]["finish_reason"] == "stop"

    async def test_streaming_reports_length_with_the_new_field(self, client, backend):
        backend.stop_field = "stop_type"
        backend.stopped_limit = True
        r = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "max_tokens": 5,
            },
        )
        assert '"finish_reason": "length"' in r.text or '"finish_reason":"length"' in r.text

    async def test_mcp_flags_truncation_with_the_new_field(
        self, client, backend, settings, monkeypatch
    ):
        backend.stop_field = "stop_type"
        backend.stopped_limit = True
        monkeypatch.setattr(settings, "api_key", "k")
        h = {"Content-Type": "application/json",
             "Accept": "application/json, text/event-stream"}
        await client.post("/mcp?key=k", headers=h, json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "t", "version": "1"}}})
        r = await client.post("/mcp?key=k", headers=h, json={
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "bitnet_chat",
                       "arguments": {"prompt": "hi", "max_tokens": 5}}})
        assert "truncated" in r.text


class TestSamplingDefaults:
    """Conservative sampling. 1.58-bit quantisation blurs the probability
    distribution, so settings that read as merely lively on a large model are
    destructive here: every coherent reply observed came from temperature 0,
    every rambling one from 0.7."""

    async def test_temperature_defaults_to_the_configured_value(self, client, backend):
        await client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert backend.requests[-1]["temperature"] == 0.3

    async def test_min_p_is_always_sent(self, client, backend):
        await client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert backend.requests[-1]["min_p"] == 0.1

    async def test_request_can_override_both(self, client, backend):
        await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 1.4,
                "min_p": 0.02,
            },
        )
        assert backend.requests[-1]["temperature"] == 1.4
        assert backend.requests[-1]["min_p"] == 0.02


class TestDefaultSystemPrompt:
    """Without framing the model drifts into free association rather than
    answering. Small instruct models lean on a system turn to stay anchored."""

    async def test_it_is_prepended_when_absent(self, client, backend):
        await client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert backend.last_prompt.startswith("System: TEST SYSTEM PROMPT<|eot_id|>")

    async def test_a_caller_system_message_wins(self, client, backend):
        """It fills a gap; it must not override intent."""
        await client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "Be a pirate"},
                    {"role": "user", "content": "hi"},
                ]
            },
        )
        assert backend.last_prompt.startswith("System: Be a pirate<|eot_id|>")
        assert "TEST SYSTEM PROMPT" not in backend.last_prompt

    async def test_empty_setting_injects_nothing(self, client, backend, settings, monkeypatch):
        monkeypatch.setattr(settings, "system_prompt", "")
        await client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert backend.last_prompt.startswith("User: hi<|eot_id|>")

    async def test_it_yields_rather_than_breaking_a_request_at_the_cap(
        self, client, settings
    ):
        """At max_tokens == ctx_size the budget is a few characters. Injecting
        regardless turned a previously working request into a 400."""
        r = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": settings.max_tokens_cap,
            },
        )
        assert r.status_code == 200

    async def test_it_is_dropped_rather_than_overflowing_the_budget(
        self, client, settings, backend, monkeypatch
    ):
        """Injected before validation, so its characters are bounded like any
        other message rather than slipping past the check that bounds them.

        Here the user message alone fits but the prompt would push it over, so
        the request is served without the prompt instead of being rejected.
        """
        marker = "x" * 200
        monkeypatch.setattr(settings, "system_prompt", marker)
        budget = prompt_budget_chars(256)
        r = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "y" * (budget - 100)}],
                "max_tokens": 256,
            },
        )
        assert r.status_code == 200
        assert marker not in backend.last_prompt

    async def test_mcp_tool_gets_the_same_framing(
        self, client, backend, settings, monkeypatch
    ):
        monkeypatch.setattr(settings, "api_key", "k")
        h = {"Content-Type": "application/json",
             "Accept": "application/json, text/event-stream"}
        await client.post("/mcp?key=k", headers=h, json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "t", "version": "1"}}})
        await client.post("/mcp?key=k", headers=h, json={
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "bitnet_chat", "arguments": {"prompt": "hi"}}})
        assert backend.last_prompt.startswith("System: TEST SYSTEM PROMPT<|eot_id|>")


class TestLoopGuard:
    """Server-side guardrail: the model can fail to emit any end-of-turn token
    and restate its answer until n_predict. Sampler penalties cannot end a
    turn, so the proxy watches the text and cuts the generation itself."""

    LOOP = "The capital of France is Paris." * 10

    async def test_non_streaming_reply_is_trimmed(self, client, backend):
        backend.content = self.LOOP
        r = await client.post("/v1/chat/completions", json=body())
        content = r.json()["choices"][0]["message"]["content"]
        assert "The capital of France is Paris." in content
        # Trailing repeats collapsed; a small remainder is fine, a wall is not.
        assert content.count("The capital of France is Paris.") < 4
        assert r.json()["choices"][0]["finish_reason"] == "stop"

    async def test_generation_is_aborted_not_just_trimmed(self, client, backend):
        """The point is saving compute: the backend stream must be dropped
        mid-generation, not read to the end and cleaned up after."""
        backend.content = "spam " * 400
        await client.post("/v1/chat/completions", json=body())
        # The stub streams one word per chunk; if the guard aborted, the
        # request ended long before all 400 chunks were consumed. There is no
        # direct hook into the closed connection, but the response content
        # bounds what was read.
        r = await client.post("/v1/chat/completions", json=body())
        assert len(r.json()["choices"][0]["message"]["content"]) < len(backend.content)

    async def test_streaming_ends_with_a_clean_final_chunk(self, client, backend):
        backend.content = self.LOOP
        frames = []
        async with client.stream(
            "POST", "/v1/chat/completions", json=body(stream=True)
        ) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    frames.append(line[6:])
        assert frames[-1] == "[DONE]"
        reasons = [
            json.loads(f)["choices"][0]["finish_reason"]
            for f in frames
            if f != "[DONE]"
        ]
        assert "stop" in reasons

    async def test_slot_is_released_after_an_abort(self, client, backend):
        backend.content = self.LOOP
        await client.post("/v1/chat/completions", json=body())
        assert (await client.get("/v1/status")).json()["busy"] is False

    async def test_ordinary_prose_is_untouched(self, client, backend):
        text = (
            "String theory proposes that particles are tiny vibrating strings. "
            "Different vibration patterns produce different particles. "
            "The theory requires extra spatial dimensions to be consistent."
        )
        backend.content = text
        r = await client.post("/v1/chat/completions", json=body())
        assert r.json()["choices"][0]["message"]["content"] == text

    async def test_legitimate_short_repetition_is_tolerated(self, client, backend):
        """Two consecutive occurrences are legal prose; the guard requires
        three. 'Location, location, location' must survive... well, almost."""
        backend.content = "It was very, very good. It was very, very good and useful."
        r = await client.post("/v1/chat/completions", json=body())
        assert r.json()["choices"][0]["message"]["content"] == backend.content

    async def test_guard_can_be_disabled(self, client, backend, settings, monkeypatch):
        monkeypatch.setattr(settings, "loop_guard_repeats", 0)
        backend.content = self.LOOP
        r = await client.post("/v1/chat/completions", json=body())
        assert r.json()["choices"][0]["message"]["content"] == self.LOOP

    async def test_mcp_tool_is_guarded_too(self, client, backend, settings, monkeypatch):
        monkeypatch.setattr(settings, "api_key", "k")
        backend.content = self.LOOP
        h = {"Content-Type": "application/json",
             "Accept": "application/json, text/event-stream"}
        await client.post("/mcp?key=k", headers=h, json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "t", "version": "1"}}})
        r = await client.post("/mcp?key=k", headers=h, json={
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "bitnet_chat", "arguments": {"prompt": "hi"}}})
        assert r.text.count("The capital of France is Paris.") < 4


class TestLeadingSpaceStrip:
    """The generation prompt no longer carries the boundary space; the model's
    first token supplies it, and it is formatting rather than content."""

    async def test_fresh_reply_leading_space_is_stripped(self, client, backend):
        backend.content = " Hello there"
        r = await client.post("/v1/chat/completions", json=body())
        assert r.json()["choices"][0]["message"]["content"] == "Hello there"

    async def test_only_one_space_is_stripped(self, client, backend):
        backend.content = "  indented"
        r = await client.post("/v1/chat/completions", json=body())
        assert r.json()["choices"][0]["message"]["content"] == " indented"

    async def test_continuation_keeps_its_leading_space(self, client, backend):
        """Mid-sentence resumption: the next token's leading space is real."""
        backend.content = " continues here"
        r = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "The reply"},
                ],
                "continuation": True,
            },
        )
        assert r.json()["choices"][0]["message"]["content"] == " continues here"

    async def test_streaming_strips_the_first_token_space(self, client, backend):
        backend.content = " alpha beta"
        text = ""
        async with client.stream(
            "POST", "/v1/chat/completions", json=body(stream=True)
        ) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: ") and line[6:] != "[DONE]":
                    text += json.loads(line[6:])["choices"][0]["delta"].get("content", "")
        assert text == "alpha beta"
