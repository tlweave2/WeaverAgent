"""Exception hierarchy for WeaverAgent."""

from __future__ import annotations


class WeaverAgentError(Exception):
    """Base class for every error raised by WeaverAgent."""


class ConfigurationError(WeaverAgentError):
    """Raised when a provider, engine, or tool is misconfigured."""


class ProviderError(WeaverAgentError):
    """Raised when an LLM provider fails or is unavailable."""


class ProviderNotInstalled(ProviderError):
    """Raised when a provider's optional SDK dependency is missing."""


class ToolError(WeaverAgentError):
    """Raised when a tool cannot be invoked.

    Errors *inside* a tool body are captured into a failed ``ToolResult``
    instead of propagating, so the agent can observe and recover from them.
    """


class ToolNotFound(ToolError):
    """Raised when the model calls a tool that is not in the registry."""


class MemoryError_(WeaverAgentError):
    """Raised when a memory backend fails."""


class ReasoningError(WeaverAgentError):
    """Raised when a reasoning engine cannot complete a run."""


class StepLimitExceeded(ReasoningError):
    """Raised when an engine hits ``max_steps`` without a final answer."""
