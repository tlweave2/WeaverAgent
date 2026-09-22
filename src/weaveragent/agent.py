"""The Agent facade."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from .errors import ConfigurationError
from .llm.base import LLMConfig, LLMProvider
from .llm.registry import get_provider
from .memory.base import MemoryStore
from .memory.ephemeral import EphemeralMemory
from .reasoning import ENGINES
from .reasoning.base import ReasoningEngine, Trace
from .tools.base import Tool
from .tools.registry import ToolRegistry
from .types import Message


class Agent:
    """An LLM, a reasoning engine, a tool registry, and a memory store.

    The four layers are independent: any provider works with any engine, any
    tool set, and any memory backend.

    ::

        agent = Agent(provider="anthropic", tools=default_registry())
        print(agent.run("What is 17 * 23, and what time is it?"))

    Args:
        provider: A provider instance, or a name for
            :func:`~weaveragent.llm.registry.get_provider`.
        engine: ``"react"``, ``"plan_execute"``, an engine class, or an engine
            instance.
        tools: Tools to expose. A list of :class:`~weaveragent.tools.base.Tool`
            is accepted and wrapped in a registry.
        memory: Memory backend. Defaults to
            :class:`~weaveragent.memory.ephemeral.EphemeralMemory`.
        model: Model id, when ``provider`` is given as a name.
        session_id: Memory session this agent reads and writes.
        system_prompt: Overrides the engine's default system prompt.
        max_steps: Model round trips allowed per run.
        history_limit: How many past turns to load from memory per run.
        config: Generation settings, when ``provider`` is given as a name.
    """

    def __init__(
        self,
        provider: LLMProvider | str = "anthropic",
        *,
        engine: ReasoningEngine | type[ReasoningEngine] | str = "react",
        tools: ToolRegistry | Sequence[Tool] | None = None,
        memory: MemoryStore | None = None,
        model: str | None = None,
        session_id: str = "default",
        system_prompt: str | None = None,
        max_steps: int = 10,
        history_limit: int = 20,
        config: LLMConfig | None = None,
        **provider_kwargs: Any,
    ) -> None:
        self.provider = (
            get_provider(provider, model=model, config=config, **provider_kwargs)
            if isinstance(provider, str)
            else provider
        )
        self.tools = _as_registry(tools)
        self.memory = memory if memory is not None else EphemeralMemory()
        self.session_id = session_id
        self.history_limit = history_limit
        self.engine = self._build_engine(engine, system_prompt, max_steps)

    def _build_engine(
        self,
        engine: ReasoningEngine | type[ReasoningEngine] | str,
        system_prompt: str | None,
        max_steps: int,
    ) -> ReasoningEngine:
        if isinstance(engine, ReasoningEngine):
            return engine

        if isinstance(engine, str):
            engine_class = ENGINES.get(engine.lower().replace("-", "_"))
            if engine_class is None:
                raise ConfigurationError(
                    f"unknown engine {engine!r}; available: {', '.join(sorted(ENGINES))}"
                )
        elif isinstance(engine, type) and issubclass(engine, ReasoningEngine):
            engine_class = engine
        else:
            raise ConfigurationError(f"cannot use {engine!r} as a reasoning engine")

        return engine_class(
            self.provider,
            self.tools,
            max_steps=max_steps,
            system_prompt=system_prompt,
        )

    # -- running --------------------------------------------------------------

    def run(self, task: str, *, remember: bool = True) -> Trace:
        """Work a task to completion and return the :class:`Trace`.

        The task and the answer are written to memory when ``remember`` is
        true, so the next run sees them as prior context.
        """
        history = self.history() if self.history_limit else []
        trace = self.engine.run(task, history=history)

        if remember:
            self.memory.add_message(Message.user(task), session_id=self.session_id)
            if trace.answer:
                self.memory.add_message(Message.assistant(trace.answer), session_id=self.session_id)
        return trace

    def ask(self, task: str, *, remember: bool = True) -> str:
        """Run a task and return only the answer text."""
        return self.run(task, remember=remember).answer

    def chat(self, message: str) -> str:
        """Alias for :meth:`ask`, for conversational use."""
        return self.ask(message)

    def stream(self, task: str) -> Iterator[str]:
        """Stream a single-turn reply, without tool use or the reasoning loop.

        Use this for conversational replies where latency to first token
        matters. :meth:`run` is the path for anything needing tools.
        """
        messages = [*self.history(), Message.user(task)]
        chunks: list[str] = []
        for chunk in self.provider.stream(messages, system=self.engine._system()):
            chunks.append(chunk)
            yield chunk

        answer = "".join(chunks)
        self.memory.add_message(Message.user(task), session_id=self.session_id)
        if answer:
            self.memory.add_message(Message.assistant(answer), session_id=self.session_id)

    # -- memory ---------------------------------------------------------------

    def history(self, limit: int | None = None) -> list[Message]:
        """Prior turns for this session, oldest first."""
        return self.memory.history(session_id=self.session_id, limit=limit or self.history_limit)

    def remember(self, fact: str, **metadata: Any) -> None:
        """Store a durable fact outside the conversation transcript."""
        self.memory.add_fact(fact, session_id=self.session_id, **metadata)

    def recall(self, query: str, *, limit: int = 5) -> list[str]:
        """Search memory and return the matching contents."""
        return [
            r.content for r in self.memory.search(query, session_id=self.session_id, limit=limit)
        ]

    def reset(self) -> int:
        """Clear this session's memory. Returns the number of records removed."""
        return self.memory.clear(session_id=self.session_id)

    # -- tools ----------------------------------------------------------------

    def add_tool(self, item: Tool) -> Tool:
        """Register a tool. It is visible to the engine on the next run."""
        return self.tools.add(item)

    def close(self) -> None:
        self.memory.close()

    def __enter__(self) -> Agent:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"Agent(provider={self.provider.name!r}, model={self.provider.model!r}, "
            f"engine={self.engine.name!r}, tools={len(self.tools)}, "
            f"memory={type(self.memory).__name__})"
        )


def _as_registry(tools: ToolRegistry | Sequence[Tool] | None) -> ToolRegistry:
    if tools is None:
        return ToolRegistry()
    if isinstance(tools, ToolRegistry):
        return tools
    return ToolRegistry(tools)
