"""Exception hierarchy for AgentForge."""

from __future__ import annotations


class AgentForgeError(Exception):
    """Base class for every error raised by AgentForge."""


class ConfigurationError(AgentForgeError):
    """Raised when a provider, engine, or tool is misconfigured."""


class ProviderError(AgentForgeError):
    """Raised when an LLM provider fails or is unavailable."""


class ProviderNotInstalled(ProviderError):
    """Raised when a provider's optional SDK dependency is missing."""


class ToolError(AgentForgeError):
    """Raised when a tool cannot be invoked.

    Errors *inside* a tool body are captured into a failed ``ToolResult``
    instead of propagating, so the agent can observe and recover from them.
    """


class ToolNotFound(ToolError):
    """Raised when the model calls a tool that is not in the registry."""


class MemoryError_(AgentForgeError):
    """Raised when a memory backend fails."""


class ReasoningError(AgentForgeError):
    """Raised when a reasoning engine cannot complete a run."""


class StepLimitExceeded(ReasoningError):
    """Raised when an engine hits ``max_steps`` without a final answer."""
