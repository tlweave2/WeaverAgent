"""The unified LLM interface layer.

Every provider implements :class:`LLMProvider`, so the rest of the framework
never touches a vendor SDK directly.
"""

from .base import LLMConfig, LLMProvider
from .mock import MockProvider
from .registry import available_providers, get_provider, register_provider

__all__ = [
    "LLMConfig",
    "LLMProvider",
    "MockProvider",
    "available_providers",
    "get_provider",
    "register_provider",
]


def __getattr__(name: str):  # pragma: no cover - thin lazy-import shim
    """Expose the SDK-backed providers lazily.

    Importing them eagerly would make ``weaveragent.llm`` fail to import
    whenever an optional SDK is absent.
    """
    if name == "AnthropicProvider":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider
    if name == "OpenAIProvider":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
