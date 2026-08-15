"""Golden tests for prompt construction (MDL-1, MDL-4).

Two renderable formats exist because Microsoft shipped two disagreeing
templates for this model:

"hf" -- tokenizer_config.json in microsoft/bitnet-b1.58-2B-4T:

    {% for message in messages %}{{ message['role'] | capitalize + ': '
       + message['content'] | trim + '<|eot_id|>' }}{% endfor %}
    {% if add_generation_prompt %}{{ 'Assistant: ' }}{% endif %}

"bitnet" -- the template embedded in ggml-model-i2_s.gguf by Microsoft's own
conversion script (Human:/BITNETAssistant: labels, <|end_of_text|>
terminators), with the conversion bug repaired: no eos_token appended after
the generation prompt.

Neither rendered prompt ends with a trailing space. The template says
"Assistant: ", but a prompt ending in a bare space encodes that space as a
standalone token, whereas training text merges it into the reply's first
token (" Sure") -- an out-of-distribution boundary a 1.58-bit model handles
badly. The API strips the model's own leading space from fresh replies
instead.

These are exact-string assertions on purpose: earlier format drift differed
only in role case, separator, and generation prompt, and each difference was
silent.
"""

import pytest

from app import Message, build_prompt


def msgs(*pairs):
    return [Message(role=r, content=c) for r, c in pairs]


def test_single_user_turn():
    assert build_prompt(msgs(("user", "hi"))) == "User: hi<|eot_id|>Assistant:"


def test_multi_turn_uses_eot_separator_not_newlines():
    prompt = build_prompt(msgs(("user", "hi"), ("assistant", "hello")))
    assert prompt == "User: hi<|eot_id|>Assistant: hello<|eot_id|>Assistant:"
    assert "\n" not in prompt


def test_system_role_is_capitalized_and_kept():
    prompt = build_prompt(msgs(("system", "Be terse"), ("user", "hi")))
    assert prompt == "System: Be terse<|eot_id|>User: hi<|eot_id|>Assistant:"


def test_roles_are_capitalized_not_lowercased():
    # The old builder emitted "user:"/"assistant:", which is out of distribution.
    prompt = build_prompt(msgs(("user", "hi")))
    assert prompt.startswith("User: ")
    assert "user:" not in prompt


def test_generation_prompt_has_no_trailing_space():
    """A bare trailing space is its own token; training merged that space into
    the reply's first token. The model supplies the space itself now."""
    prompt = build_prompt(msgs(("user", "hi")))
    assert prompt.endswith("Assistant:")
    assert not prompt.endswith(" ")


def test_content_is_trimmed():
    assert build_prompt(msgs(("user", "  hi  "))) == "User: hi<|eot_id|>Assistant:"


def test_continuation_resumes_the_assistant_turn():
    # Must NOT append a fresh "Assistant:" after the partial, which is what
    # made the Continue button restart instead of continue. The boundary space
    # returns HERE: "Assistant: partial" tokenized as one string BPE-merges
    # " partial" exactly as training did.
    prompt = build_prompt(
        msgs(("user", "hi"), ("assistant", "partial reply")), continuation=True
    )
    assert prompt == "User: hi<|eot_id|>Assistant: partial reply"
    assert prompt.count("Assistant:") == 1


def test_continuation_ignored_when_last_turn_is_not_assistant():
    prompt = build_prompt(msgs(("user", "hi")), continuation=True)
    assert prompt == "User: hi<|eot_id|>Assistant:"


class TestBitnetFormat:
    """The GGUF-embedded template, selected with BITNET_PROMPT_FORMAT=bitnet.

    This is the format Microsoft's own conversion script wrote into the file
    and their demo effectively runs. Which template this quantised checkpoint
    answers better under is an open question; the switch exists so it can be
    settled by deployment rather than argument.
    """

    @pytest.fixture(autouse=True)
    def _use_bitnet(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "prompt_format", "bitnet")

    def test_single_user_turn(self):
        prompt = build_prompt(msgs(("user", "hi")))
        assert prompt == "Human: hi\n\nBITNETAssistant:"

    def test_no_eos_after_generation_prompt(self):
        """The embedded template appends eos_token right after the generation
        prompt -- generating after an eos is nonsensical, a conversion bug.
        The repair drops it."""
        prompt = build_prompt(msgs(("user", "hi")))
        assert "<|end_of_text|>" not in prompt

    def test_assistant_turns_end_with_end_of_text(self):
        prompt = build_prompt(msgs(("user", "hi"), ("assistant", "hello"),
                                   ("user", "more")))
        assert prompt == (
            "Human: hi\n\nBITNETAssistant: hello<|end_of_text|>"
            "Human: more\n\nBITNETAssistant:"
        )

    def test_system_is_folded_into_the_first_user_turn(self):
        """The embedded template has no system branch at all; dropping the
        system message silently would discard caller intent."""
        prompt = build_prompt(msgs(("system", "Be terse"), ("user", "hi")))
        assert prompt == "Human: Be terse\n\nhi\n\nBITNETAssistant:"

    def test_no_trailing_space_here_either(self):
        assert not build_prompt(msgs(("user", "hi"))).endswith(" ")

    def test_continuation(self):
        prompt = build_prompt(
            msgs(("user", "hi"), ("assistant", "partial")), continuation=True
        )
        assert prompt == "Human: hi\n\nBITNETAssistant: partial"


class TestBitnetFormatReviewFindings:
    """Defects the adversarial review confirmed in the first renderer."""

    @pytest.fixture(autouse=True)
    def _use_bitnet(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "prompt_format", "bitnet")

    def test_mid_conversation_system_message_is_not_dropped(self):
        """The UI's compaction injects a system summary mid-list; dropping it
        silently would lose the summarized history exactly when compaction
        was needed."""
        prompt = build_prompt(msgs(
            ("system", "Context summary: earlier facts"),
            ("user", "next question"),
        ))
        assert "Context summary: earlier facts" in prompt

    def test_multiple_system_messages_all_survive(self):
        prompt = build_prompt(msgs(
            ("system", "one"), ("system", "two"), ("user", "q"),
        ))
        assert "one" in prompt and "two" in prompt

    def test_system_with_no_user_turn_still_reaches_the_model(self):
        prompt = build_prompt(msgs(("system", "only framing")))
        assert "only framing" in prompt
        assert prompt.endswith("BITNETAssistant:")

    def test_trailing_assistant_turn_still_gets_a_generation_prompt(self):
        """Few-shot lists ending in an assistant example must not leave
        generation dangling after an end-of-text token with no role."""
        prompt = build_prompt(msgs(("user", "q"), ("assistant", "example")))
        assert prompt.endswith("BITNETAssistant:")
