"""A small set of ready-made tools.

These are deliberately conservative: arithmetic is evaluated from a parsed AST
rather than ``eval``, and file access is confined to an explicit root
directory. Build a registry with :func:`default_registry`, or cherry-pick.
"""

from __future__ import annotations

import ast
import json
import operator
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .base import Tool, tool
from .registry import ToolRegistry

_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Guards against a cheap denial-of-service: `9**9**9` is three characters of
# input and unbounded work.
_MAX_EXPONENT = 1000


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError(f"exponent too large (limit {_MAX_EXPONENT})")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported unary operator: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


@tool(tags=("math",))
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the result.

    Supports + - * / // % ** over numbers. Use this instead of doing
    arithmetic mentally.

    Args:
        expression: The expression to evaluate, e.g. "(17 * 23) + 4".
    """
    tree = ast.parse(expression, mode="eval")
    return str(_eval_node(tree))


@tool(tags=("time",))
def current_time(tz: str = "UTC") -> str:
    """Return the current date and time in ISO 8601 format.

    Args:
        tz: Either "UTC" or "local". Defaults to "UTC".
    """
    if tz.lower() == "local":
        return datetime.now().astimezone().isoformat()
    return datetime.now(timezone.utc).isoformat()


@tool(tags=("data",))
def json_extract(document: str, path: str) -> str:
    """Read a value out of a JSON document using a dotted path.

    Args:
        document: The JSON text to read.
        path: A dotted path such as "user.address.city" or "items.0.name".
    """
    current: Any = json.loads(document)
    for part in filter(None, path.split(".")):
        if isinstance(current, list):
            current = current[int(part)]
        elif isinstance(current, dict):
            current = current[part]
        else:
            raise KeyError(f"cannot descend into {type(current).__name__} at {part!r}")
    return json.dumps(current, default=str)


def make_file_tools(root: str | os.PathLike[str], *, writable: bool = False) -> list[Tool]:
    """Build file tools confined to ``root``.

    Paths are resolved and checked against ``root`` so that ``..`` segments and
    symlinks cannot escape it. Writing is off unless ``writable=True``.

    Args:
        root: Directory the tools may access.
        writable: Whether to also return a write tool.
    """
    base = Path(root).expanduser().resolve()

    def resolve(relative: str) -> Path:
        target = (base / relative).resolve()
        if target != base and base not in target.parents:
            raise ValueError(f"path escapes the allowed root: {relative!r}")
        return target

    @tool(name="read_file", tags=("files",))
    def read_file(path: str, max_bytes: int = 100_000) -> str:
        """Read a UTF-8 text file and return its contents.

        Args:
            path: Path relative to the allowed root directory.
            max_bytes: Stop after this many bytes. Defaults to 100000.
        """
        target = resolve(path)
        data = target.read_bytes()[:max_bytes]
        return data.decode("utf-8", errors="replace")

    @tool(name="list_files", tags=("files",))
    def list_files(path: str = ".") -> str:
        """List the entries of a directory, one per line.

        Args:
            path: Directory relative to the allowed root. Defaults to the root.
        """
        target = resolve(path)
        if not target.is_dir():
            raise NotADirectoryError(f"not a directory: {path}")
        entries = sorted(
            f"{child.name}/" if child.is_dir() else child.name for child in target.iterdir()
        )
        return "\n".join(entries) or "(empty directory)"

    tools = [read_file, list_files]

    if writable:

        @tool(name="write_file", tags=("files",))
        def write_file(path: str, content: str) -> str:
            """Write UTF-8 text to a file, creating parent directories.

            Overwrites the file if it already exists.

            Args:
                path: Path relative to the allowed root directory.
                content: The text to write.
            """
            target = resolve(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"wrote {len(content)} characters to {path}"

        tools.append(write_file)

    return tools


def default_registry(
    *, file_root: str | os.PathLike[str] | None = None, writable_files: bool = False
) -> ToolRegistry:
    """A registry of the general-purpose builtins.

    Args:
        file_root: If given, also include file tools rooted at this directory.
        writable_files: Whether the file tools may write. Requires ``file_root``.
    """
    registry = ToolRegistry([calculator, current_time, json_extract])
    if file_root is not None:
        for item in make_file_tools(file_root, writable=writable_files):
            registry.add(item)
    return registry
