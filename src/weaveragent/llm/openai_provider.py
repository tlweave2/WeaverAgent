"""OpenAI-compatible provider.

Also works against any endpoint that speaks the Chat Completions API (vLLM,
Ollama, Together, OpenRouter, ...) by passing ``base_url``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import Any

from ..errors import ProviderError, ProviderNotInstalled
from ..types import LLMResponse, Message, Role, StopReason, ToolCall, Usage
from .base import LLMConfig, LLMProvider

_FINISH_REASONS = {
    "stop": StopReason.END_TURN,
    "tool_calls": StopReason.TOOL_USE,
    "function_call": StopReason.TOOL_USE,
    "length": StopReason.MAX_TOKENS,
    "content_filter": StopReason.REFUSAL,
}


class OpenAIProvider(LLMProvider):
    """Chat Completions models via the official ``openai`` SDK.

    Install with ``pip install 'weaveragent[openai]'``. The SDK reads
    ``OPENAI_API_KEY`` from the environment.
    """

    name = "openai"
    default_model = "gpt-4o"

    def __init__(
        self,
        *,
        model: str | None = None,
        config: LLMConfig | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any = None,
        **client_kwargs: Any,
    ) -> None:
        super().__init__(model=model, config=config)
        if client is not None:
            self._client = client
            return
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ProviderNotInstalled(
                "the openai SDK is required for OpenAIProvider; "
                "install it with: pip install 'weaveragent[openai]'"
            ) from exc
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**client_kwargs)

    def _build_request(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None,
        system: str | None,
        config: LLMConfig | None,
    ) -> dict[str, Any]:
        merged = self._merged(config)
        wire_messages = _to_openai_messages(messages)
        if system:
            wire_messages.insert(0, {"role": "system", "content": system})

        request: dict[str, Any] = {
            "model": merged.model or self.model,
            "max_completion_tokens": merged.max_tokens,
            "messages": wire_messages,
        }
        if merged.temperature is not None:
            request["temperature"] = merged.temperature
        if merged.stop_sequences:
            request["stop"] = merged.stop_sequences
        if merged.timeout is not None:
            request["timeout"] = merged.timeout
        if tools:
            request["tools"] = [_to_openai_tool(t) for t in tools]
        request.update(merged.extra)
        return request

    def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        system: str | None = None,
        config: LLMConfig | None = None,
    ) -> LLMResponse:
        request = self._build_request(messages, tools, system, config)
        try:
            completion = self._client.chat.completions.create(**request)
        except Exception as exc:  # noqa: BLE001 - normalized into ProviderError
            raise ProviderError(f"OpenAI request failed: {exc}") from exc
        return _from_openai_completion(completion)

    def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        system: str | None = None,
        config: LLMConfig | None = None,
    ) -> Iterator[str]:
        request = self._build_request(messages, tools, system, config)
        request["stream"] = True
        try:
            for chunk in self._client.chat.completions.create(**request):
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                text = getattr(delta, "content", None)
                if text:
                    yield text
        except Exception as exc:  # noqa: BLE001 - normalized into ProviderError
            raise ProviderError(f"OpenAI stream failed: {exc}") from exc


def _to_openai_tool(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema.get("description", ""),
            "parameters": schema.get("parameters", {"type": "object", "properties": {}}),
        },
    }


def _to_openai_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Translate neutral messages into Chat Completions form.

    Tool results become one ``role: "tool"`` message each, which is how this
    API pairs them with their call -- unlike Anthropic, which batches them
    into a single user turn.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        if message.role is Role.SYSTEM:
            out.append({"role": "system", "content": message.content})
        elif message.role is Role.USER:
            out.append({"role": "user", "content": message.content})
        elif message.role is Role.ASSISTANT:
            entry: dict[str, Any] = {"role": "assistant", "content": message.content or None}
            if message.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, default=str),
                        },
                    }
                    for call in message.tool_calls
                ]
            out.append(entry)
        elif message.role is Role.TOOL:
            for result in message.tool_results:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.tool_call_id,
                        "content": result.content,
                    }
                )
    return out


def _from_openai_completion(completion: Any) -> LLMResponse:
    choices = getattr(completion, "choices", None) or []
    if not choices:
        raise ProviderError("OpenAI response contained no choices")

    choice = choices[0]
    message = choice.message
    tool_calls: list[ToolCall] = []

    for raw_call in getattr(message, "tool_calls", None) or []:
        function = getattr(raw_call, "function", None)
        if function is None:
            continue
        try:
            arguments = json.loads(function.arguments or "{}")
        except json.JSONDecodeError:
            # A malformed argument payload is reported to the model as a failed
            # call rather than crashing the run.
            arguments = {"__malformed_arguments__": function.arguments}
        tool_calls.append(
            ToolCall(
                name=function.name,
                arguments=arguments if isinstance(arguments, dict) else {},
                id=raw_call.id,
            )
        )

    usage = getattr(completion, "usage", None)
    return LLMResponse(
        content=getattr(message, "content", None) or "",
        tool_calls=tool_calls,
        stop_reason=_FINISH_REASONS.get(getattr(choice, "finish_reason", None), StopReason.OTHER),
        model=getattr(completion, "model", ""),
        usage=Usage(
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
        if usage
        else Usage(),
        raw=completion,
    )
