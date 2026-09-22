"""An in-process provider for tests and offline demos."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any

from ..errors import ProviderError
from ..types import LLMResponse, Message, Role, StopReason, ToolCall, Usage
from .base import LLMConfig, LLMProvider

Script = LLMResponse | str | Callable[[list[Message]], LLMResponse]


class MockProvider(LLMProvider):
    """Replays a scripted sequence of responses.

    This is what makes the reasoning engines testable without a network call
    or an API key. Each entry of ``script`` may be:

    * an :class:`~weaveragent.types.LLMResponse` -- returned as-is;
    * a ``str`` -- returned as a final text answer;
    * a callable taking the message history and returning a response, for
      behavior that depends on what the agent has done so far.

    Every call is recorded on :attr:`calls` for assertions.
    """

    name = "mock"
    default_model = "mock-model"

    def __init__(
        self,
        script: Sequence[Script] = (),
        *,
        model: str | None = None,
        config: LLMConfig | None = None,
        repeat_last: bool = False,
    ) -> None:
        super().__init__(model=model, config=config)
        self.script: list[Script] = list(script)
        self.repeat_last = repeat_last
        self.calls: list[dict[str, Any]] = []
        self._index = 0

    @staticmethod
    def tool_call(name: str, arguments: dict[str, Any], *, text: str = "") -> LLMResponse:
        """Build a response that requests one tool call."""
        return LLMResponse(
            content=text,
            tool_calls=[ToolCall(name=name, arguments=arguments)],
            stop_reason=StopReason.TOOL_USE,
            model="mock-model",
        )

    @staticmethod
    def answer(text: str) -> LLMResponse:
        """Build a final text response."""
        return LLMResponse(content=text, stop_reason=StopReason.END_TURN, model="mock-model")

    def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        system: str | None = None,
        config: LLMConfig | None = None,
    ) -> LLMResponse:
        history = list(messages)
        self.calls.append(
            {
                "messages": history,
                "tools": [t["name"] for t in (tools or [])],
                "system": system,
            }
        )

        if not self.script:
            # No script at all: echo, so `weaveragent run -p mock` and other
            # smoke tests work without an API key. An *exhausted* script is a
            # different matter -- see below.
            return self._echo(history)

        if self._index >= len(self.script):
            if self.repeat_last:
                entry = self.script[-1]
            else:
                raise ProviderError(
                    f"MockProvider script exhausted after {len(self.script)} response(s); "
                    "add more entries or pass repeat_last=True"
                )
        else:
            entry = self.script[self._index]
            self._index += 1

        if callable(entry) and not isinstance(entry, LLMResponse):
            entry = entry(history)
        if isinstance(entry, str):
            entry = self.answer(entry)

        return LLMResponse(
            content=entry.content,
            tool_calls=list(entry.tool_calls),
            stop_reason=entry.stop_reason,
            model=entry.model or self.model,
            usage=entry.usage
            if entry.usage.total_tokens
            else Usage(input_tokens=len(history), output_tokens=1),
            raw=entry.raw,
        )

    def _echo(self, history: list[Message]) -> LLMResponse:
        """A canned answer for an unscripted provider."""
        last = next((m.content for m in reversed(history) if m.role is Role.USER and m.content), "")
        return LLMResponse(
            content=f"[mock] {last}" if last else "[mock] no input",
            stop_reason=StopReason.END_TURN,
            model=self.model,
            usage=Usage(input_tokens=len(history), output_tokens=1),
        )

    def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        system: str | None = None,
        config: LLMConfig | None = None,
    ) -> Iterator[str]:
        """Yield the scripted text one word at a time."""
        response = self.complete(messages, tools=tools, system=system, config=config)
        for index, word in enumerate(response.content.split()):
            yield word if index == 0 else f" {word}"
