"""The unified provider interface and its adapters."""

import json
from types import SimpleNamespace

import pytest

from weaveragent.errors import ConfigurationError, ProviderError
from weaveragent.llm import LLMConfig, MockProvider, available_providers, get_provider
from weaveragent.llm.anthropic_provider import (
    AnthropicProvider,
    _from_anthropic_message,
    _to_anthropic_messages,
)
from weaveragent.llm.openai_provider import OpenAIProvider, _from_openai_completion
from weaveragent.llm.registry import register_provider
from weaveragent.types import Message, StopReason, ToolCall, ToolResult, Usage

# -- registry ---------------------------------------------------------------


def test_registry_lists_and_builds_providers():
    assert {"anthropic", "claude", "openai", "mock"} <= set(available_providers())
    assert get_provider("mock").name == "mock"


def test_unknown_provider_is_a_configuration_error():
    with pytest.raises(ConfigurationError, match="unknown provider"):
        get_provider("nope")


def test_custom_provider_can_be_registered():
    register_provider("custom", lambda **kw: MockProvider(["hi"], **kw))
    assert get_provider("custom").complete([Message.user("x")]).content == "hi"


# -- mock provider ----------------------------------------------------------


def test_mock_replays_script_and_records_calls():
    provider = MockProvider(["first", "second"])
    assert provider.complete([Message.user("a")]).content == "first"
    assert provider.complete([Message.user("b")]).content == "second"
    assert len(provider.calls) == 2


def test_mock_raises_when_script_is_exhausted():
    provider = MockProvider(["only"])
    provider.complete([Message.user("a")])
    with pytest.raises(ProviderError, match="script exhausted"):
        provider.complete([Message.user("b")])


def test_mock_repeat_last_keeps_answering():
    provider = MockProvider(["same"], repeat_last=True)
    assert [provider.complete([]).content for _ in range(3)] == ["same"] * 3


def test_mock_callable_entry_sees_history():
    provider = MockProvider([lambda history: MockProvider.answer(f"saw {len(history)}")])
    assert provider.complete([Message.user("a"), Message.user("b")]).content == "saw 2"


def test_mock_stream_yields_words_that_rejoin():
    provider = MockProvider(["alpha beta gamma"])
    assert "".join(provider.stream([Message.user("x")])) == "alpha beta gamma"


# -- config merging ---------------------------------------------------------


def test_per_call_config_overlays_provider_config():
    provider = MockProvider(config=LLMConfig(max_tokens=100, extra={"a": 1}))
    merged = provider._merged(LLMConfig(max_tokens=500, extra={"b": 2}))
    assert merged.max_tokens == 500
    assert merged.extra == {"a": 1, "b": 2}


def test_base_stream_falls_back_to_complete():
    """A provider with no native streaming still supports stream()."""
    from weaveragent.llm.base import LLMProvider

    provider = MockProvider(["text"])
    # Invoke the base implementation directly, bypassing MockProvider's override.
    assert list(LLMProvider.stream(provider, [Message.user("x")])) == ["text"]


# -- Anthropic translation --------------------------------------------------


def _anthropic_client(message):
    return SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: message))


def test_anthropic_request_shape():
    provider = AnthropicProvider(client=_anthropic_client(None), model="claude-opus-5")
    request = provider._build_request(
        [Message.user("hello")],
        tools=[{"name": "t", "description": "d", "parameters": {"type": "object"}}],
        system="be brief",
        config=None,
        streaming=False,
    )
    assert request["model"] == "claude-opus-5"
    assert request["system"] == "be brief"
    # Neutral `parameters` becomes Anthropic's `input_schema`.
    assert request["tools"][0]["input_schema"] == {"type": "object"}
    assert "parameters" not in request["tools"][0]
    assert request["thinking"] == {"type": "adaptive"}


def test_anthropic_streaming_sets_eager_input_streaming():
    provider = AnthropicProvider(client=_anthropic_client(None))
    request = provider._build_request(
        [Message.user("x")],
        tools=[{"name": "t", "description": "", "parameters": {}}],
        system=None,
        config=None,
        streaming=True,
    )
    assert request["tools"][0]["eager_input_streaming"] is True


def test_anthropic_drops_temperature_on_models_that_reject_it(caplog):
    provider = AnthropicProvider(client=_anthropic_client(None), model="claude-opus-5")
    request = provider._build_request(
        [Message.user("x")], None, None, LLMConfig(temperature=0.5), streaming=False
    )
    assert "temperature" not in request
    assert "rejects temperature" in caplog.text


def test_anthropic_keeps_temperature_where_supported():
    provider = AnthropicProvider(client=_anthropic_client(None), model="claude-opus-4-6")
    request = provider._build_request(
        [Message.user("x")], None, None, LLMConfig(temperature=0.5), streaming=False
    )
    assert request["temperature"] == 0.5


def test_anthropic_omits_thinking_for_older_models():
    provider = AnthropicProvider(client=_anthropic_client(None), model="claude-haiku-4-5")
    request = provider._build_request([Message.user("x")], None, None, None, streaming=False)
    assert "thinking" not in request


def test_anthropic_extra_overrides_defaults():
    provider = AnthropicProvider(client=_anthropic_client(None), model="claude-opus-5")
    request = provider._build_request(
        [Message.user("x")],
        None,
        None,
        LLMConfig(extra={"thinking": None, "output_config": {"effort": "low"}}),
        streaming=False,
    )
    assert "thinking" not in request
    assert request["output_config"] == {"effort": "low"}


def test_anthropic_message_translation_round_trip():
    history = [
        Message.system("ignored here"),
        Message.user("what is 2+2"),
        Message.assistant("let me compute", [ToolCall("calc", {"e": "2+2"}, id="tu_1")]),
        Message.tool([ToolResult("tu_1", "calc", "4")]),
    ]
    wire = _to_anthropic_messages(history)

    assert [m["role"] for m in wire] == ["user", "assistant", "user"]
    assert wire[1]["content"][0] == {"type": "text", "text": "let me compute"}
    assert wire[1]["content"][1]["type"] == "tool_use"
    assert wire[1]["content"][1]["id"] == "tu_1"
    result_block = wire[2]["content"][0]
    assert result_block["type"] == "tool_result"
    assert result_block["tool_use_id"] == "tu_1"
    assert "is_error" not in result_block


def test_anthropic_failed_tool_result_is_flagged():
    wire = _to_anthropic_messages([Message.tool([ToolResult("tu_1", "calc", "bad", ok=False)])])
    assert wire[0]["content"][0]["is_error"] is True


def test_anthropic_response_normalization():
    message = SimpleNamespace(
        stop_reason="tool_use",
        model="claude-opus-5",
        content=[
            SimpleNamespace(type="thinking", thinking="ignored"),
            SimpleNamespace(type="text", text="I will calculate."),
            SimpleNamespace(type="tool_use", id="tu_9", name="calc", input={"e": "1+1"}),
        ],
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=4,
            cache_read_input_tokens=2,
            cache_creation_input_tokens=1,
        ),
    )
    response = _from_anthropic_message(message)

    assert response.content == "I will calculate."
    assert response.stop_reason is StopReason.TOOL_USE
    assert response.wants_tools
    assert response.tool_calls[0].id == "tu_9"
    assert response.usage == Usage(10, 4, 2, 1)


def test_anthropic_refusal_is_surfaced_not_swallowed():
    """A refusal returns HTTP 200 with no usable content; it must not read as an empty answer."""
    message = SimpleNamespace(
        stop_reason="refusal",
        model="claude-opus-5",
        content=[],
        stop_details=SimpleNamespace(category="cyber", explanation="declined"),
        usage=None,
    )
    response = _from_anthropic_message(message)
    assert response.stop_reason is StopReason.REFUSAL
    assert "refused: cyber" in response.content


def test_anthropic_wraps_sdk_failures():
    def boom(**_):
        raise RuntimeError("network down")

    provider = AnthropicProvider(client=SimpleNamespace(messages=SimpleNamespace(create=boom)))
    with pytest.raises(ProviderError, match="Anthropic request failed"):
        provider.complete([Message.user("x")])


# -- OpenAI translation -----------------------------------------------------


def test_openai_request_shape_and_tool_envelope():
    provider = OpenAIProvider(client=SimpleNamespace(), model="gpt-4o")
    request = provider._build_request(
        [Message.user("hi")],
        tools=[{"name": "t", "description": "d", "parameters": {"type": "object"}}],
        system="sys",
        config=None,
    )
    assert request["messages"][0] == {"role": "system", "content": "sys"}
    assert request["tools"][0]["type"] == "function"
    assert request["tools"][0]["function"]["name"] == "t"


def test_openai_tool_results_become_individual_tool_messages():
    from weaveragent.llm.openai_provider import _to_openai_messages

    wire = _to_openai_messages(
        [
            Message.assistant("calling", [ToolCall("calc", {"e": "1"}, id="c1")]),
            Message.tool([ToolResult("c1", "calc", "1"), ToolResult("c2", "calc", "2")]),
        ]
    )
    assert wire[0]["tool_calls"][0]["function"]["arguments"] == json.dumps({"e": "1"})
    assert [m["role"] for m in wire[1:]] == ["tool", "tool"]
    assert wire[1]["tool_call_id"] == "c1"


def test_openai_response_normalization():
    completion = SimpleNamespace(
        model="gpt-4o",
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content="working",
                    tool_calls=[
                        SimpleNamespace(
                            id="c1",
                            function=SimpleNamespace(name="calc", arguments='{"e": "1+1"}'),
                        )
                    ],
                ),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
    )
    response = _from_openai_completion(completion)
    assert response.stop_reason is StopReason.TOOL_USE
    assert response.tool_calls[0].arguments == {"e": "1+1"}
    assert response.usage.total_tokens == 10


def test_openai_malformed_tool_arguments_do_not_crash():
    completion = SimpleNamespace(
        model="gpt-4o",
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="c1", function=SimpleNamespace(name="calc", arguments="{not json")
                        )
                    ],
                ),
            )
        ],
        usage=None,
    )
    response = _from_openai_completion(completion)
    assert "__malformed_arguments__" in response.tool_calls[0].arguments


def test_unscripted_mock_echoes_the_last_user_message():
    """An empty script makes the provider usable for smoke tests without a key."""
    provider = MockProvider()
    assert provider.complete([Message.user("hello there")]).content == "[mock] hello there"
    assert provider.complete([]).content == "[mock] no input"


def test_exhausted_script_still_raises():
    """An empty script echoes, but running out of a real script is a test bug."""
    provider = MockProvider(["one"])
    provider.complete([Message.user("a")])
    with pytest.raises(ProviderError, match="script exhausted"):
        provider.complete([Message.user("b")])
