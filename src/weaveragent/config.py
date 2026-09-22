"""Environment-driven defaults.

Lets a deployment pick a provider, model, engine, and memory backend without
code changes. Used by the CLI and available to applications.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .llm.base import LLMConfig


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(slots=True)
class Settings:
    """Resolved defaults for building an :class:`~weaveragent.agent.Agent`.

    Every field maps to a ``WEAVERAGENT_*`` environment variable, read by
    :meth:`from_env`.
    """

    provider: str = "anthropic"
    model: str | None = None
    engine: str = "react"
    max_steps: int = 10
    max_tokens: int = 16_000
    session_id: str = "default"
    memory_path: str | None = None
    history_limit: int = 20
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, **overrides: Any) -> Settings:
        """Read settings from the environment.

        Recognized variables: ``WEAVERAGENT_PROVIDER``, ``WEAVERAGENT_MODEL``,
        ``WEAVERAGENT_ENGINE``, ``WEAVERAGENT_MAX_STEPS``,
        ``WEAVERAGENT_MAX_TOKENS``, ``WEAVERAGENT_SESSION``,
        ``WEAVERAGENT_MEMORY_PATH``, ``WEAVERAGENT_HISTORY_LIMIT``.

        Keyword arguments win over the environment, so explicit CLI flags and
        application code stay authoritative.
        """
        settings = cls(
            provider=os.environ.get("WEAVERAGENT_PROVIDER", "anthropic"),
            model=os.environ.get("WEAVERAGENT_MODEL") or None,
            engine=os.environ.get("WEAVERAGENT_ENGINE", "react"),
            max_steps=_env_int("WEAVERAGENT_MAX_STEPS", 10),
            max_tokens=_env_int("WEAVERAGENT_MAX_TOKENS", 16_000),
            session_id=os.environ.get("WEAVERAGENT_SESSION", "default"),
            memory_path=os.environ.get("WEAVERAGENT_MEMORY_PATH") or None,
            history_limit=_env_int("WEAVERAGENT_HISTORY_LIMIT", 20),
        )
        for key, value in overrides.items():
            if value is not None and hasattr(settings, key):
                setattr(settings, key, value)
        return settings

    def llm_config(self) -> LLMConfig:
        return LLMConfig(model=self.model, max_tokens=self.max_tokens, extra=dict(self.extra))

    def build_memory(self):
        """A memory store honoring :attr:`memory_path`.

        A path means persistence; no path means in-process only.
        """
        if self.memory_path:
            from .memory.sqlite import SQLiteMemory

            return SQLiteMemory(self.memory_path)
        from .memory.ephemeral import EphemeralMemory

        return EphemeralMemory()

    def build_agent(self, **kwargs: Any):
        """Construct an :class:`~weaveragent.agent.Agent` from these settings."""
        from .agent import Agent

        options: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "engine": self.engine,
            "memory": self.build_memory(),
            "session_id": self.session_id,
            "max_steps": self.max_steps,
            "history_limit": self.history_limit,
            "config": self.llm_config(),
        }
        options.update(kwargs)
        return Agent(**options)
