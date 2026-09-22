"""Contract tests against the real provider SDKs.

These skip when the optional SDK is not installed, which is what keeps the
core dependency-free: the rest of the suite must pass without them. CI runs
the matrix bare to enforce that, and runs this file in a job that installs
the extras.

What they guard: the adapters build request payloads by hand, so a field the
SDK renames or drops would otherwise surface as a 400 at runtime instead of a
failure here.
"""

from __future__ import annotations

import pytest

from weaveragent.llm.base import LLMConfig
from weaveragent.types import Message

anthropic = pytest.importorskip("anthropic", reason="install .[anthropic] to run")


def _request(**kwargs):
    from types import SimpleNamespace

    from weaveragent.llm.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider(
        client=SimpleNamespace(), model=kwargs.pop("model", "claude-opus-5")
    )
    return provider._build_request(
        [Message.user("hi")],
        tools=kwargs.pop("tools", None),
        system=kwargs.pop("system", None),
        config=kwargs.pop("config", None),
        streaming=kwargs.pop("streaming", False),
    )


def test_every_request_key_is_a_real_sdk_parameter():
    from anthropic.types import message_create_params

    request = _request(
        system="be brief",
        tools=[{"name": "t", "description": "d", "parameters": {"type": "object"}}],
        config=LLMConfig(max_tokens=2048),
    )
    known = set(message_create_params.MessageCreateParamsNonStreaming.__annotations__)
    assert set(request) <= known, f"not SDK parameters: {set(request) - known}"


def test_tool_definition_keys_are_real_sdk_tool_params():
    from anthropic.types import ToolParam

    from weaveragent.llm.anthropic_provider import _to_anthropic_tool

    schema = {"name": "t", "description": "d", "parameters": {"type": "object"}}
    known = set(ToolParam.__annotations__)

    for streaming in (False, True):
        definition = _to_anthropic_tool(schema, streaming=streaming)
        assert set(definition) <= known, f"not ToolParam fields: {set(definition) - known}"


def test_eager_input_streaming_is_still_a_tool_field():
    """Set on streaming tool calls; a rename would silently stop taking effect."""
    from anthropic.types import ToolParam

    assert "eager_input_streaming" in ToolParam.__annotations__


def test_adaptive_thinking_variant_still_exists():
    """The adapter sends {"type": "adaptive"} on models that accept it."""
    import anthropic.types as types

    assert hasattr(types, "ThinkingConfigAdaptiveParam")
    assert _request()["thinking"] == {"type": "adaptive"}


def test_sdk_is_the_major_version_we_pin():
    """pyproject requires anthropic>=1.0.0; 0.x had a different surface."""
    assert int(anthropic.__version__.split(".")[0]) >= 1


def test_openai_adapter_constructs_against_the_real_sdk():
    openai = pytest.importorskip("openai", reason="install .[openai] to run")

    from weaveragent.llm.openai_provider import OpenAIProvider

    provider = OpenAIProvider(api_key="test-key-not-used", model="gpt-4o")
    assert hasattr(provider._client.chat.completions, "create")
    assert openai  # imported successfully

    request = provider._build_request(
        [Message.user("hi")],
        tools=[{"name": "t", "description": "d", "parameters": {"type": "object"}}],
        system="sys",
        config=None,
    )
    assert request["tools"][0]["type"] == "function"
    assert "max_completion_tokens" in request
