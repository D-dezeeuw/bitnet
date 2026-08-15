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

    async def test_role_label_stops_are_on_by_default(self, client, backend):
        """The model routinely ends a turn by writing the next speaker's label
        instead of an end token, so this is often the only stop that fires."""
        await client.post("/v1/chat/completions", json=body())
        assert "User:" in backend.requests[-1]["stop"]

    async def test_role_label_stops_follow_the_prompt_format(
        self, client, backend, settings, monkeypatch
    ):
        """The bitnet template's next-speaker label is "Human:", not "User:";
        stopping on the wrong one stops on nothing."""
        monkeypatch.setattr(settings, "prompt_format", "bitnet")
        await client.post("/v1/chat/completions", json=body())
        stops = backend.requests[-1]["stop"]
        assert "Human:" in stops
        assert "User:" not in stops

    async def test_role_label_stops_can_be_disabled(self, client, backend, settings, monkeypatch):
        monkeypatch.setattr(settings, "role_stops", False)
        await client.post("/v1/chat/completions", json=body())
        assert "User:" not in backend.requests[-1]["stop"]

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
        # Both guards: the echo guard would otherwise cut this text at the
        # first restatement, which is its job but not what this test measures.
        monkeypatch.setattr(settings, "loop_guard_repeats", 0)
        monkeypatch.setattr(settings, "echo_similarity", 0.0)
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


class TestLoopGuardReviewFindings:
    """Each of these encodes a defect the adversarial review confirmed in the
    first version of the guard."""

    async def test_whitespace_flood_is_caught(self, client, backend):
        """A model emitting endless newlines is a real degenerate mode; the
        original whitespace skip made the guard permanently blind to it."""
        backend.content = "An answer." + "\n" * 200
        r = await client.post("/v1/chat/completions", json=body())
        content = r.json()["choices"][0]["message"]["content"]
        assert len(content) < 100

    async def test_dash_banner_survives(self, client, backend):
        """A 40-dash divider is legitimate formatting, not a loop."""
        text = "Here is the section:\n" + "-" * 40 + "\nContent below the rule."
        backend.content = text
        r = await client.post("/v1/chat/completions", json=body())
        assert r.json()["choices"][0]["message"]["content"] == text

    async def test_markdown_table_separator_survives(self, client, backend):
        text = "| a | b | c |\n|---|---|---|---|---|---|---|---|---|\n| 1 | 2 | 3 |"
        backend.content = text
        r = await client.post("/v1/chat/completions", json=body())
        assert r.json()["choices"][0]["message"]["content"] == text

    async def test_long_answer_restated_is_caught(self, client, backend):
        """The headline failure: the model restating its ENTIRE answer. At
        MAX_PHRASE=200 a 220-char answer slipped through."""
        answer = (
            "String theory proposes that all particles are tiny vibrating "
            "strings whose vibration patterns determine their properties, and "
            "it requires several extra spatial dimensions to be mathematically "
            "consistent with quantum mechanics. "
        )
        assert len(answer) > 200
        backend.content = answer * 5
        r = await client.post("/v1/chat/completions", json=body())
        content = r.json()["choices"][0]["message"]["content"]
        assert content.count("String theory proposes") < 4

    async def test_no_dangling_fragment_after_trim(self, client, backend):
        """When the tripping chunk straddles a repeat boundary the detected
        block is a rotation of the true phrase; the trim must not leave a
        mid-word fragment of it dangling."""
        backend.content = ("I cannot answer that question. " * 6).strip()
        r = await client.post("/v1/chat/completions", json=body())
        content = r.json()["choices"][0]["message"]["content"]
        assert not content.endswith(("I canno", "I cannot an", "I c"))
        assert content.count("I cannot answer") >= 1

    async def test_repeats_of_one_is_clamped_not_vacuous(
        self, client, backend, settings, monkeypatch
    ):
        """repeats=1 made the detector trivially true and aborted every
        generation at its 12th character."""
        monkeypatch.setattr(settings, "loop_guard_repeats", 1)
        backend.content = "Hello, world! This is a perfectly ordinary reply."
        r = await client.post("/v1/chat/completions", json=body())
        assert (
            r.json()["choices"][0]["message"]["content"] == backend.content
        )

    async def test_guard_trip_reports_nonzero_prompt_tokens(self, client, backend):
        """On an abort the backend's counts chunk never arrives; usage must be
        estimated, not reported as a 0-token prompt."""
        backend.content = "The capital of France is Paris. " * 10
        r = await client.post("/v1/chat/completions", json=body())
        usage = r.json()["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0

    async def test_unterminated_stream_is_reported_as_length(self, client, backend):
        """A stream that ends without the final chunk did not finish; calling
        it 'stop' would present the fragment as a complete answer."""
        backend.truncate_stream = True
        r = await client.post("/v1/chat/completions", json=body())
        assert r.json()["choices"][0]["finish_reason"] == "length"
        assert r.json()["usage"]["prompt_tokens"] > 0


class TestTopPDefault:
    async def test_top_p_disabled_not_omitted(self, client, backend):
        """Omitting top_p does not turn it off -- the backend then applies its
        own 0.95 default over the min_p-first design. 1.0 disables it."""
        await client.post("/v1/chat/completions", json=body())
        assert backend.requests[-1]["top_p"] == 1.0


class TestContinuationSpaceEdgeCase:
    async def test_flag_without_assistant_tail_still_strips(self, client, backend):
        """continuation=true with a user-ending history builds a FRESH prompt
        (the flag is ignored), so the boundary-space strip must apply. The UI
        hits this by stopping a reply before its first token."""
        backend.content = " Sure, here it is."
        r = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "continuation": True,
            },
        )
        assert r.json()["choices"][0]["message"]["content"] == "Sure, here it is."


class TestEchoGuard:
    """This model's dominant failure is not an exact loop: it answers correctly
    in one sentence, then restates that sentence with a word changed. Nothing
    matches exactly, so DRY and the exact loop guard both miss it -- but the
    reply was already complete after sentence one.

    Every REAL_* case below is output captured from the deployment.
    """

    REAL_GRAVITY = (
        "Gravity is a force that pulls objects towards the center of the Earth."
        "Gravity is a force that pulls objects toward the center of the Earth.\n"
        "Gravity is a force that pulls object toward the center of the earth."
    )

    async def test_restatement_is_cut_leaving_the_answer(self, client, backend):
        backend.content = self.REAL_GRAVITY
        r = await client.post("/v1/chat/completions", json=body())
        content = r.json()["choices"][0]["message"]["content"]
        assert content == (
            "Gravity is a force that pulls objects towards the center of the Earth."
        )

    async def test_fused_seam_is_a_sentence_boundary(self, client, backend):
        """The restatement fuses onto the previous sentence with no space
        ("Earth.Gravity"). Requiring whitespace made the guard blind to exactly
        the boundary it exists to find."""
        backend.content = "The capital of France is Paris.The capital of France is paris."
        r = await client.post("/v1/chat/completions", json=body())
        assert (
            r.json()["choices"][0]["message"]["content"]
            == "The capital of France is Paris."
        )

    async def test_repeated_sentence_openings_are_cut(self, client, backend):
        """Second signal: looser paraphrase scores far too low on similarity to
        act on safely, but the model keeps restarting the same sentence."""
        backend.content = (
            "In simple terms strings connect to each other in space. "
            "In simple terms strings connect and form patterns together. "
            "In simple terms strings connect and vibrate in many dimensions."
        )
        r = await client.post("/v1/chat/completions", json=body())
        content = r.json()["choices"][0]["message"]["content"]
        assert content.count("In simple terms") == 1

    async def test_legitimately_similar_sentences_survive(self, client, backend):
        """The closest legitimate pair measured scores 0.98 -- one word inverts
        the meaning. Cutting it would silently delete correct content, which is
        worse than leaving repetition in. The threshold sits above it."""
        text = (
            "This function returns the total number of active users. "
            "This function returns the total number of inactive users."
        )
        backend.content = text
        r = await client.post("/v1/chat/completions", json=body())
        assert r.json()["choices"][0]["message"]["content"] == text

    async def test_parallel_prose_survives(self, client, backend):
        text = (
            "The first option is to rebuild the container from scratch. "
            "The second option is to patch the running instance in place. "
            "The third option is to roll back entirely."
        )
        backend.content = text
        r = await client.post("/v1/chat/completions", json=body())
        assert r.json()["choices"][0]["message"]["content"] == text

    async def test_ordinary_multi_sentence_answer_survives(self, client, backend):
        text = (
            "Gravity is the attraction between masses. It is described by "
            "general relativity as the curvature of spacetime. On Earth it "
            "accelerates objects at about 9.8 metres per second squared."
        )
        backend.content = text
        r = await client.post("/v1/chat/completions", json=body())
        assert r.json()["choices"][0]["message"]["content"] == text

    async def test_guard_can_be_disabled(self, client, backend, settings, monkeypatch):
        monkeypatch.setattr(settings, "echo_similarity", 0.0)
        backend.content = self.REAL_GRAVITY
        r = await client.post("/v1/chat/completions", json=body())
        assert r.json()["choices"][0]["message"]["content"] == self.REAL_GRAVITY

    async def test_streaming_is_guarded_too(self, client, backend):
        backend.content = self.REAL_GRAVITY
        frames = []
        async with client.stream(
            "POST", "/v1/chat/completions", json=body(stream=True)
        ) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    frames.append(line[6:])
        text = "".join(
            json.loads(f)["choices"][0]["delta"].get("content", "")
            for f in frames if f != "[DONE]"
        )
        assert text.count("Gravity is a force") <= 2
        assert frames[-1] == "[DONE]"


class TestEchoGuardCountRule:
    """A single similarity threshold provably cannot separate degenerate from
    legitimate output here, because the degenerate text scores LOWER.

    Measured: an observed wall of "The string theory is a {term,way,method}
    that describes the strings of string." scores 0.86-0.95 between sentences,
    while a legitimate pair differing only by "active"/"inactive" scores 0.98.
    What separates them is the COUNT of near-duplicates -- one versus
    seventeen -- so they are counted at a low threshold and one is forgiven.
    """

    SCREENSHOT_WALL = (
        "The string theory is a mathematical concept that describes the "
        "strings of strings. It is a theory that describes the strings of "
        "string theory.\n"
        "The string theory is a concept that describes the string of string "
        "theory.\n"
        "The string theory is a term that describes the strings of string.\n"
        "The string theory is a way that describes the strings of string.\n"
        "The string theory is a method that describes the strings of string.\n"
        "The string theory is a technique that describes the strings of string."
    )

    async def test_varied_word_wall_is_cut(self, client, backend):
        """Captured from the deployment: 256 tokens of the same sentence with
        one noun swapped each time. Scores too low for the strong threshold and
        shares too short a prefix for the prefix rule; only the count catches
        it."""
        backend.content = self.SCREENSHOT_WALL
        r = await client.post("/v1/chat/completions", json=body())
        content = r.json()["choices"][0]["message"]["content"]
        assert content.count("The string theory is a") <= 2
        assert len(content) < len(self.SCREENSHOT_WALL) / 2

    async def test_one_near_duplicate_is_forgiven(self, client, backend):
        """0.98 similar and the difference IS the meaning. Real prose has at
        most one of these; degenerate output has many."""
        text = (
            "This function returns the total number of active users. "
            "This function returns the total number of inactive users."
        )
        backend.content = text
        r = await client.post("/v1/chat/completions", json=body())
        assert r.json()["choices"][0]["message"]["content"] == text

    async def test_technical_prose_with_a_shared_subject_survives(self, client, backend):
        text = (
            "The container joins the proxy network on a static address. "
            "Nginx forwards requests to that address on port 8010. "
            "TLS terminates at the proxy, so the backend speaks plain HTTP."
        )
        backend.content = text
        r = await client.post("/v1/chat/completions", json=body())
        assert r.json()["choices"][0]["message"]["content"] == text

    async def test_cut_keeps_the_sentence_that_answered(self, client, backend):
        """Cutting at the FIRST near-duplicate keeps the original statement and
        drops every restatement after it."""
        backend.content = (
            "The string theory is a term that describes the strings of string. "
            "The string theory is a way that describes the strings of string. "
            "The string theory is a method that describes the strings of string."
        )
        r = await client.post("/v1/chat/completions", json=body())
        content = r.json()["choices"][0]["message"]["content"]
        assert content == (
            "The string theory is a term that describes the strings of string."
        )

    async def test_a_genuinely_diverging_third_sentence_is_kept(self, client, backend):
        """Only ONE of these is a near-duplicate; the third restates loosely
        enough (0.81) to be ordinary variation. One is forgiven, so the reply
        stands."""
        text = (
            "Gravity pulls objects with mass toward one another constantly. "
            "Gravity pulls objects with mass toward each other constantly. "
            "Gravity pulls objects that have mass toward each other always."
        )
        backend.content = text
        r = await client.post("/v1/chat/completions", json=body())
        assert r.json()["choices"][0]["message"]["content"] == text
