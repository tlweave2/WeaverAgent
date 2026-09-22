"""Provider lookup by name."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..errors import ConfigurationError
from .base import LLMConfig, LLMProvider

_BUILDERS: dict[str, Callable[..., LLMProvider]] = {}


def _load_anthropic(**kwargs: Any) -> LLMProvider:
    from .anthropic_provider import AnthropicProvider

    return AnthropicProvider(**kwargs)


def _load_openai(**kwargs: Any) -> LLMProvider:
    from .openai_provider import OpenAIProvider

    return OpenAIProvider(**kwargs)


def _load_mock(**kwargs: Any) -> LLMProvider:
    from .mock import MockProvider

    return MockProvider(**kwargs)


# Imports are deferred so that a missing optional SDK only matters when that
# provider is actually requested.
_BUILDERS.update(
    {
        "anthropic": _load_anthropic,
        "claude": _load_anthropic,
        "openai": _load_openai,
        "mock": _load_mock,
    }
)


def register_provider(name: str, builder: Callable[..., LLMProvider]) -> None:
    """Add a custom provider under ``name``, making it available to :func:`get_provider`."""
    _BUILDERS[name.lower()] = builder


def available_providers() -> list[str]:
    return sorted(_BUILDERS)


def get_provider(
    name: str = "anthropic",
    *,
    model: str | None = None,
    config: LLMConfig | None = None,
    **kwargs: Any,
) -> LLMProvider:
    """Instantiate a provider by name.

    Args:
        name: One of :func:`available_providers`, or a name added with
            :func:`register_provider`.
        model: Model id. Defaults to the provider's own default.
        config: Generation settings.
        **kwargs: Passed to the provider's constructor.
    """
    builder = _BUILDERS.get(name.lower())
    if builder is None:
        raise ConfigurationError(
            f"unknown provider {name!r}; available: {', '.join(available_providers())}"
        )
    return builder(model=model, config=config, **kwargs)
