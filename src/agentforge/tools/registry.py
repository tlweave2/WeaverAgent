"""The pluggable tool registry."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any

from ..errors import ToolNotFound
from ..types import ToolCall, ToolResult
from .base import Tool
from .base import tool as tool_decorator


class ToolRegistry:
    """A named collection of tools that an agent can offer to a model.

    Registries compose: build a base registry of shared tools and
    :meth:`subset` or :meth:`merge` it per agent, so one agent's tool surface
    does not dictate another's.
    """

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for item in tools:
            self.add(item)

    def add(self, item: Tool) -> Tool:
        """Register a tool. Replaces any existing tool of the same name."""
        if not isinstance(item, Tool):
            raise TypeError(f"expected a Tool, got {type(item).__name__}; use @tool to wrap it")
        self._tools[item.name] = item
        return item

    def register(
        self,
        func: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Decorator that wraps a function with :func:`tool` and registers it.

        ::

            registry = ToolRegistry()

            @registry.register
            def now() -> str:
                '''Return the current UTC time.'''
                return datetime.now(timezone.utc).isoformat()
        """

        def decorate(fn: Callable[..., Any]) -> Tool:
            return self.add(tool_decorator(fn, **kwargs))

        return decorate if func is None else decorate(func)

    def remove(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            available = ", ".join(sorted(self._tools)) or "none"
            raise ToolNotFound(f"no tool named {name!r}; available: {available}") from None

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """Provider-neutral schemas for every registered tool."""
        return [self._tools[name].to_schema() for name in sorted(self._tools)]

    def describe(self) -> str:
        """A compact catalog for embedding in a system prompt."""
        if not self._tools:
            return "(no tools available)"
        return "\n".join(
            f"- {name}: {self._tools[name].description}" for name in sorted(self._tools)
        )

    def subset(
        self, names: Iterable[str] | None = None, *, tags: Iterable[str] = ()
    ) -> ToolRegistry:
        """A new registry containing only the tools matching ``names`` or ``tags``."""
        wanted = set(names) if names is not None else None
        tag_set = set(tags)
        selected = [
            item
            for item in self._tools.values()
            if (wanted is None or item.name in wanted) and (not tag_set or tag_set & set(item.tags))
        ]
        return ToolRegistry(selected)

    def merge(self, other: ToolRegistry) -> ToolRegistry:
        """A new registry with ``other``'s tools layered on top of this one's."""
        return ToolRegistry([*self._tools.values(), *other._tools.values()])

    def invoke(self, call: ToolCall) -> ToolResult:
        """Execute one tool call.

        An unknown tool name becomes a failed result, not an exception: models
        do occasionally hallucinate a tool, and the error text is usually
        enough for them to correct course on the next step.
        """
        try:
            target = self.get(call.name)
        except ToolNotFound as exc:
            return ToolResult(tool_call_id=call.id, name=call.name, content=str(exc), ok=False)
        return target.invoke(call.id, call.arguments)

    async def ainvoke(self, call: ToolCall) -> ToolResult:
        """Async counterpart to :meth:`invoke`."""
        try:
            target = self.get(call.name)
        except ToolNotFound as exc:
            return ToolResult(tool_call_id=call.id, name=call.name, content=str(exc), ok=False)
        return await target.ainvoke(call.id, call.arguments)

    def invoke_all(self, calls: Iterable[ToolCall]) -> list[ToolResult]:
        """Execute every call, in order, returning one result per call.

        Providers require a result for every call they requested, so results
        are returned even for failures.
        """
        return [self.invoke(call) for call in calls]

    async def ainvoke_all(self, calls: Iterable[ToolCall]) -> list[ToolResult]:
        """Execute calls concurrently, preserving input order in the results."""
        import asyncio

        call_list = list(calls)
        if not call_list:
            return []
        return list(await asyncio.gather(*(self.ainvoke(c) for c in call_list)))

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools[name] for name in sorted(self._tools))

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"ToolRegistry({self.names()})"
