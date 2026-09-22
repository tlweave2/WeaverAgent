"""Anthropic (Claude) provider."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from typing import Any

from ..errors import ProviderError, ProviderNotInstalled
from ..types import LLMResponse, Message, Role, StopReason, ToolCall, Usage
from .base import LLMConfig, LLMProvider

logger = logging.getLogger(__name__)

_STOP_REASONS = {
    "end_turn": StopReason.END_TURN,
    "tool_use": StopReason.TOOL_USE,
    "max_tokens": StopReason.MAX_TOKENS,
    "stop_sequence": StopReason.STOP_SEQUENCE,
    "refusal": StopReason.REFUSAL,
    "pause_turn": StopReason.OTHER,
}

# Models that take `thinking: {"type": "adaptive"}`. Earlier models use the
# deprecated `budget_tokens` form and reject adaptive, so thinking is left
# unset for them rather than guessed at.
_ADAPTIVE_THINKING_PREFIXES = (
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
)

# Models where temperature/top_p/top_k were removed and are rejected with a 400.
_NO_SAMPLING_PREFIXES = (
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
)


class AnthropicProvider(LLMProvider):
    """Claude models via the official ``anthropic`` SDK.

    Install with ``pip install 'agentforge[anthropic]'``. Credentials are
    resolved by the SDK from ``ANTHROPIC_API_KEY``, ``ANTHROPIC_AUTH_TOKEN``,
    or an ``ant auth login`` profile, so no key needs to be passed here.

    Adaptive thinking is enabled by default on models that support it. Override
    it, or set request options with no portable equivalent, through
    ``LLMConfig.extra`` -- for example
    ``extra={"thinking": {"type": "adaptive", "display": "summarized"},
    "output_config": {"effort": "high"}}``.
    """

    name = "anthropic"
    default_model = "claude-opus-5"

    def __init__(
        self,
        *,
        model: str | None = None,
        config: LLMConfig | None = None,
        api_key: str | None = None,
        client: Any = None,
        **client_kwargs: Any,
    ) -> None:
        super().__init__(model=model, config=config)
        if client is not None:
            self._client = client
            return
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ProviderNotInstalled(
                "the anthropic SDK is required for AnthropicProvider; "
                "install it with: pip install 'agentforge[anthropic]'"
            ) from exc
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        self._client = anthropic.Anthropic(**client_kwargs)

    # -- request construction -------------------------------------------------

    def _build_request(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None,
        system: str | None,
        config: LLMConfig | None,
        *,
        streaming: bool,
    ) -> dict[str, Any]:
        merged = self._merged(config)
        model = merged.model or self.model

        system_parts = [m.content for m in messages if m.role is Role.SYSTEM and m.content]
        if system:
            system_parts.insert(0, system)

        request: dict[str, Any] = {
            "model": model,
            "max_tokens": merged.max_tokens,
            "messages": _to_anthropic_messages(messages),
        }
        if system_parts:
            request["system"] = "\n\n".join(system_parts)
        if merged.stop_sequences:
            request["stop_sequences"] = merged.stop_sequences
        if tools:
            request["tools"] = [_to_anthropic_tool(t, streaming=streaming) for t in tools]

        if merged.temperature is not None:
            if model.startswith(_NO_SAMPLING_PREFIXES):
                logger.warning(
                    "%s rejects temperature; dropping it. Use output_config.effort "
                    "via LLMConfig.extra to tune this model instead.",
                    model,
                )
            else:
                request["temperature"] = merged.temperature

        if model.startswith(_ADAPTIVE_THINKING_PREFIXES):
            request["thinking"] = {"type": "adaptive"}

        # `extra` wins over every default above, including thinking.
        request.update(merged.extra)
        if request.get("thinking") is None:
            request.pop("thinking", None)
        if merged.timeout is not None:
            request["timeout"] = merged.timeout
        return request

    # -- generation -----------------------------------------------------------

    def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        system: str | None = None,
        config: LLMConfig | None = None,
    ) -> LLMResponse:
        request = self._build_request(messages, tools, system, config, streaming=False)
        try:
            message = self._client.messages.create(**request)
        except Exception as exc:  # noqa: BLE001 - normalized into ProviderError
            raise ProviderError(f"Anthropic request failed: {exc}") from exc
        return _from_anthropic_message(message)

    def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        system: str | None = None,
        config: LLMConfig | None = None,
    ) -> Iterator[str]:
        request = self._build_request(messages, tools, system, config, streaming=True)
        try:
            with self._client.messages.stream(**request) as stream:
                yield from stream.text_stream
        except Exception as exc:  # noqa: BLE001 - normalized into ProviderError
            raise ProviderError(f"Anthropic stream failed: {exc}") from exc

    def complete_streaming(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        system: str | None = None,
        config: LLMConfig | None = None,
    ) -> LLMResponse:
        """Stream the request but return the assembled response.

        Streaming avoids HTTP timeouts on long or high-``max_tokens`` requests,
        which is why this, not :meth:`complete`, is the safe path for large
        generations that still need the full structured result.
        """
        request = self._build_request(messages, tools, system, config, streaming=True)
        try:
            with self._client.messages.stream(**request) as stream:
                message = stream.get_final_message()
        except Exception as exc:  # noqa: BLE001 - normalized into ProviderError
            raise ProviderError(f"Anthropic stream failed: {exc}") from exc
        return _from_anthropic_message(message)


def _to_anthropic_tool(schema: dict[str, Any], *, streaming: bool) -> dict[str, Any]:
    """Reshape a neutral tool schema into Anthropic's ``input_schema`` form."""
    definition: dict[str, Any] = {
        "name": schema["name"],
        "description": schema.get("description", ""),
        "input_schema": schema.get("parameters", {"type": "object", "properties": {}}),
    }
    if streaming:
        # Large tool inputs then stream as they are generated instead of
        # arriving in one burst; inputs are validated in Tool._validate.
        definition["eager_input_streaming"] = True
    return definition


def _to_anthropic_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Translate neutral messages into Anthropic content blocks.

    System messages are dropped here because they are hoisted into the
    top-level ``system`` parameter by the caller.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        if message.role is Role.SYSTEM:
            continue

        if message.role is Role.TOOL:
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": result.tool_call_id,
                            "content": result.content,
                            **({"is_error": True} if result.is_error else {}),
                        }
                        for result in message.tool_results
                    ],
                }
            )
            continue

        if message.role is Role.ASSISTANT:
            blocks: list[dict[str, Any]] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            for call in message.tool_calls:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    }
                )
            if blocks:
                out.append({"role": "assistant", "content": blocks})
            continue

        out.append({"role": "user", "content": message.content})
    return out


def _from_anthropic_message(message: Any) -> LLMResponse:
    """Normalize an Anthropic ``Message`` into an :class:`LLMResponse`.

    ``stop_reason`` is read before the content blocks: a refusal returns HTTP
    200 with no usable content, and reading blindly would yield an empty
    answer with no explanation.
    """
    stop_reason = _STOP_REASONS.get(getattr(message, "stop_reason", None), StopReason.OTHER)

    if stop_reason is StopReason.REFUSAL:
        details = getattr(message, "stop_details", None)
        category = getattr(details, "category", None) or "unspecified"
        explanation = getattr(details, "explanation", None) or ""
        return LLMResponse(
            content=f"[refused: {category}] {explanation}".strip(),
            stop_reason=StopReason.REFUSAL,
            model=getattr(message, "model", ""),
            usage=_usage_from(message),
            raw=message,
        )

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in getattr(message, "content", []) or []:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(block.text)
        elif block_type == "tool_use":
            arguments = block.input if isinstance(block.input, dict) else {}
            tool_calls.append(ToolCall(name=block.name, arguments=arguments, id=block.id))

    return LLMResponse(
        content="".join(text_parts),
        tool_calls=tool_calls,
        stop_reason=stop_reason,
        model=getattr(message, "model", ""),
        usage=_usage_from(message),
        raw=message,
    )


def _usage_from(message: Any) -> Usage:
    usage = getattr(message, "usage", None)
    if usage is None:
        return Usage()
    return Usage(
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )
