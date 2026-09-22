"""The provider interface every LLM adapter implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..types import LLMResponse, Message


@dataclass(slots=True)
class LLMConfig:
    """Generation settings, shared across providers.

    ``extra`` carries provider-specific parameters that have no portable
    equivalent; each adapter passes through the keys it understands and
    ignores the rest, so one config can be reused across providers.
    """

    model: str | None = None
    max_tokens: int = 16_000
    temperature: float | None = None
    stop_sequences: list[str] = field(default_factory=list)
    timeout: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """A normalized interface over one model vendor.

    Implementations translate :class:`~agentforge.types.Message` and
    provider-neutral tool schemas onto their wire format, and translate
    responses back into :class:`~agentforge.types.LLMResponse`. Engines depend
    only on this contract.
    """

    #: Model used when the caller and the config leave it unset.
    default_model: str = ""

    #: Short identifier for the provider registry, e.g. ``"anthropic"``.
    name: str = "base"

    def __init__(self, *, model: str | None = None, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self.model = model or self.config.model or self.default_model

    @abstractmethod
    def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        system: str | None = None,
        config: LLMConfig | None = None,
    ) -> LLMResponse:
        """Generate one completion.

        Args:
            messages: Conversation history, oldest first. A leading system
                message is honored, but prefer the ``system`` argument.
            tools: Provider-neutral tool schemas from
                :meth:`ToolRegistry.schemas`. Omit to forbid tool use.
            system: System prompt.
            config: Per-call overrides of the provider's config.
        """

    def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        system: str | None = None,
        config: LLMConfig | None = None,
    ) -> Iterator[str]:
        """Yield text deltas as they arrive.

        The default implementation falls back to a single :meth:`complete`
        call, so every provider supports the method even without native
        streaming. Adapters that can stream should override it.
        """
        response = self.complete(messages, tools=tools, system=system, config=config)
        if response.content:
            yield response.content

    def _merged(self, config: LLMConfig | None) -> LLMConfig:
        """Overlay a per-call config on the provider's defaults."""
        if config is None:
            return self.config
        return LLMConfig(
            model=config.model or self.config.model,
            max_tokens=config.max_tokens or self.config.max_tokens,
            temperature=(
                config.temperature if config.temperature is not None else self.config.temperature
            ),
            stop_sequences=config.stop_sequences or self.config.stop_sequences,
            timeout=config.timeout if config.timeout is not None else self.config.timeout,
            extra={**self.config.extra, **config.extra},
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"
