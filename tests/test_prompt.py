"""Golden tests for prompt construction (MDL-1, MDL-4).

The target format comes from tokenizer_config.json in
microsoft/bitnet-b1.58-2B-4T:

    {% for message in messages %}{{ message['role'] | capitalize + ': '
       + message['content'] | trim + '<|eot_id|>' }}{% endfor %}
    {% if add_generation_prompt %}{{ 'Assistant: ' }}{% endif %}

These are exact-string assertions on purpose: the previous format differed only
in role case, separator, and generation prompt, and each difference was silent.
"""

from app import Message, build_prompt


def msgs(*pairs):
    return [Message(role=r, content=c) for r, c in pairs]


def test_single_user_turn():
    assert build_prompt(msgs(("user", "hi"))) == "User: hi<|eot_id|>Assistant: "


def test_multi_turn_uses_eot_separator_not_newlines():
    prompt = build_prompt(msgs(("user", "hi"), ("assistant", "hello")))
    assert prompt == "User: hi<|eot_id|>Assistant: hello<|eot_id|>Assistant: "
    assert "\n" not in prompt


def test_system_role_is_capitalized_and_kept():
    prompt = build_prompt(msgs(("system", "Be terse"), ("user", "hi")))
    assert prompt == "System: Be terse<|eot_id|>User: hi<|eot_id|>Assistant: "


def test_roles_are_capitalized_not_lowercased():
    # The old builder emitted "user:"/"assistant:", which is out of distribution.
    prompt = build_prompt(msgs(("user", "hi")))
    assert prompt.startswith("User: ")
    assert "user:" not in prompt


def test_generation_prompt_keeps_its_trailing_space():
    assert build_prompt(msgs(("user", "hi"))).endswith("Assistant: ")


def test_content_is_trimmed():
    assert build_prompt(msgs(("user", "  hi  "))) == "User: hi<|eot_id|>Assistant: "


def test_continuation_resumes_the_assistant_turn():
    # Must NOT append a fresh "Assistant: " after the partial, which is what
    # made the Continue button restart instead of continue.
    prompt = build_prompt(
        msgs(("user", "hi"), ("assistant", "partial reply")), continuation=True
    )
    assert prompt == "User: hi<|eot_id|>Assistant: partial reply"
    assert not prompt.endswith("Assistant: ")
    assert prompt.count("Assistant: ") == 1


def test_continuation_preserves_a_trailing_space_in_the_partial():
    # A trailing space is a real token boundary when resuming mid-sentence.
    prompt = build_prompt(
        msgs(("user", "hi"), ("assistant", "half a ")), continuation=True
    )
    assert prompt == "User: hi<|eot_id|>Assistant: half a "


def test_continuation_ignored_when_last_turn_is_not_assistant():
    prompt = build_prompt(msgs(("user", "hi")), continuation=True)
    assert prompt == "User: hi<|eot_id|>Assistant: "
