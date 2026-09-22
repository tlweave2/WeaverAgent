"""Tool definitions and JSON Schema inference."""

from __future__ import annotations

import asyncio
import inspect
import time
import types as pytypes
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, get_args, get_origin, get_type_hints

from ..errors import ToolError
from ..types import ToolResult

_JSON_TYPES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _schema_for(annotation: Any) -> dict[str, Any]:
    """Map a Python annotation onto a JSON Schema fragment.

    Unknown annotations degrade to an unconstrained value rather than raising:
    a slightly loose schema still lets the model call the tool, while a hard
    failure at import time would take the whole registry down.
    """
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}

    origin = get_origin(annotation)

    # Optional[X] / X | None -> schema for X (nullability is expressed by
    # omission from `required`, which is what providers actually enforce).
    if origin in (typing.Union, pytypes.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return _schema_for(args[0])
        return {"anyOf": [_schema_for(a) for a in args]}

    if origin in (list, set, tuple):
        args = get_args(annotation)
        item = _schema_for(args[0]) if args else {}
        return {"type": "array", "items": item} if item else {"type": "array"}

    if origin is dict:
        return {"type": "object"}

    if isinstance(annotation, type):
        if issubclass(annotation, bool):
            return {"type": "boolean"}
        for py_type, json_type in _JSON_TYPES.items():
            if issubclass(annotation, py_type):
                return {"type": json_type}

    return {}


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into a summary and per-arg descriptions.

    Tool descriptions are the model's only guide to when a tool applies, so
    the docstring is treated as part of the tool's interface.
    """
    if not doc:
        return "", {}

    lines = inspect.cleandoc(doc).splitlines()
    summary: list[str] = []
    args: dict[str, str] = {}
    in_args = False
    current: str | None = None

    for line in lines:
        stripped = line.strip()
        if stripped.lower() in ("args:", "arguments:", "parameters:"):
            in_args = True
            continue
        if stripped.lower() in ("returns:", "raises:", "yields:", "examples:", "example:"):
            in_args = False
            continue

        if in_args and stripped:
            if ":" in stripped and not line.startswith((" " * 8, "\t\t")):
                name, _, desc = stripped.partition(":")
                current = name.split("(")[0].strip()
                args[current] = desc.strip()
            elif current:
                args[current] = f"{args[current]} {stripped}".strip()
        elif not in_args:
            summary.append(stripped)

    return " ".join(s for s in summary if s).strip(), args


@dataclass(slots=True)
class Tool:
    """A callable the model may invoke, plus the schema that describes it.

    Build these with the :func:`tool` decorator rather than by hand; it derives
    ``parameters`` from the function signature so the schema cannot drift away
    from the implementation.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    is_async: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)

    def to_schema(self) -> dict[str, Any]:
        """Provider-neutral schema; each adapter reshapes this to its wire format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def invoke(self, call_id: str, arguments: dict[str, Any]) -> ToolResult:
        """Run the tool, capturing failures as a failed :class:`ToolResult`.

        An exception in a tool body is information the agent can act on, so it
        becomes an observation instead of unwinding the run.
        """
        started = time.perf_counter()
        try:
            self._validate(arguments)
            output = self.func(**arguments)
            if inspect.isawaitable(output):
                output = asyncio.run(_await(output))
            return ToolResult(
                tool_call_id=call_id,
                name=self.name,
                content=_stringify(output),
                ok=True,
                duration_s=time.perf_counter() - started,
            )
        except Exception as exc:  # noqa: BLE001 - deliberate: surfaced to the model
            return ToolResult(
                tool_call_id=call_id,
                name=self.name,
                content=f"{type(exc).__name__}: {exc}",
                ok=False,
                duration_s=time.perf_counter() - started,
            )

    async def ainvoke(self, call_id: str, arguments: dict[str, Any]) -> ToolResult:
        """Async counterpart to :meth:`invoke`. Sync tools run in a thread."""
        started = time.perf_counter()
        try:
            self._validate(arguments)
            if self.is_async:
                output = await self.func(**arguments)
            else:
                output = await asyncio.to_thread(lambda: self.func(**arguments))
            return ToolResult(
                tool_call_id=call_id,
                name=self.name,
                content=_stringify(output),
                ok=True,
                duration_s=time.perf_counter() - started,
            )
        except Exception as exc:  # noqa: BLE001 - deliberate: surfaced to the model
            return ToolResult(
                tool_call_id=call_id,
                name=self.name,
                content=f"{type(exc).__name__}: {exc}",
                ok=False,
                duration_s=time.perf_counter() - started,
            )

    def _validate(self, arguments: dict[str, Any]) -> None:
        """Check required keys and reject unknown ones.

        Providers do not guarantee schema-valid arguments unless strict mode is
        on, and an unexpected keyword would otherwise surface as an opaque
        ``TypeError``.
        """
        required = set(self.parameters.get("required", []))
        known = set(self.parameters.get("properties", {}))
        given = set(arguments)

        if missing := required - given:
            raise ToolError(f"missing required argument(s): {', '.join(sorted(missing))}")
        if unknown := given - known:
            raise ToolError(f"unexpected argument(s): {', '.join(sorted(unknown))}")


async def _await(awaitable: Any) -> Any:
    return await awaitable


def _stringify(value: Any) -> str:
    """Render a tool's return value as the text the model will read."""
    if isinstance(value, str):
        return value
    if value is None:
        return "(no output)"
    if isinstance(value, (dict, list, tuple, int, float, bool)):
        import json

        return json.dumps(value, default=str, indent=2)
    return str(value)


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    tags: tuple[str, ...] = (),
) -> Any:
    """Turn a function into a :class:`Tool`, inferring its schema.

    Usable bare or with arguments::

        @tool
        def add(a: int, b: int) -> int:
            '''Add two integers.

            Args:
                a: First addend.
                b: Second addend.
            '''
            return a + b

    Args:
        func: The function to wrap.
        name: Override the tool name. Defaults to the function name.
        description: Override the description. Defaults to the docstring summary.
        tags: Free-form labels for selecting subsets of a registry.
    """

    def decorate(fn: Callable[..., Any]) -> Tool:
        summary, arg_docs = _parse_docstring(fn.__doc__)
        signature = inspect.signature(fn)
        try:
            hints = get_type_hints(fn)
        except Exception:  # noqa: BLE001 - unresolvable forward refs shouldn't break import
            hints = {}

        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in signature.parameters.items():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            schema = _schema_for(hints.get(param_name, param.annotation))
            if doc := arg_docs.get(param_name):
                schema["description"] = doc
            properties[param_name] = schema
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        parameters: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        }
        if required:
            parameters["required"] = required

        return Tool(
            name=name or fn.__name__,
            description=description or summary or f"Call {fn.__name__}.",
            parameters=parameters,
            func=fn,
            is_async=inspect.iscoroutinefunction(fn),
            tags=tags,
        )

    return decorate if func is None else decorate(func)
