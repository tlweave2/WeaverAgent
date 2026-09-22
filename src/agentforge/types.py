"""Provider-neutral data types.

Every layer of AgentForge speaks these types, never a vendor SDK's types.
That is what makes the reasoning engines, tools, and memory stores portable
across providers: an engine that consumes :class:`LLMResponse` works
identically against Claude, GPT, or the in-process mock provider.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    """Who authored a message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class StopReason(str, Enum):
    """Why the model stopped generating.

    Provider-specific stop reasons are normalized onto these values by each
    adapter, so engines branch on one vocabulary.
    """

    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    REFUSAL = "refusal"
    OTHER = "other"


@dataclass(slots=True)
class ToolCall:
    """A model's request to invoke a tool.

    ``id`` correlates the call with its :class:`ToolResult`; providers require
    the pairing to be exact, so it is carried through untouched.
    """

    name: str
    arguments: dict[str, Any]
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:16]}")

    def __str__(self) -> str:
        return f"{self.name}({json.dumps(self.arguments, default=str)})"


@dataclass(slots=True)
class ToolResult:
    """The outcome of invoking a tool.

    A failed tool produces a result with ``ok=False`` rather than raising, so
    the model can see the error text and try a different approach.
    """

    tool_call_id: str
    name: str
    content: str
    ok: bool = True
    duration_s: float = 0.0

    @property
    def is_error(self) -> bool:
        return not self.ok

    def __str__(self) -> str:
        prefix = "" if self.ok else "ERROR: "
        return f"{prefix}{self.content}"


@dataclass(slots=True)
class Message:
    """One turn in a conversation.

    An assistant message may carry ``tool_calls``; a tool message carries the
    matching ``tool_results``. Both lists are empty for plain text turns.
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    name: str | None = None
    created_at: float = field(default_factory=time.time)

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str = "", tool_calls: list[ToolCall] | None = None) -> Message:
        return cls(role=Role.ASSISTANT, content=content, tool_calls=list(tool_calls or []))

    @classmethod
    def tool(cls, results: list[ToolResult]) -> Message:
        return cls(role=Role.TOOL, tool_results=list(results))

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence. Round-trips through :meth:`from_dict`."""
        return {
            "role": self.role.value,
            "content": self.content,
            "name": self.name,
            "created_at": self.created_at,
            "tool_calls": [
                {"id": c.id, "name": c.name, "arguments": c.arguments} for c in self.tool_calls
            ],
            "tool_results": [
                {
                    "tool_call_id": r.tool_call_id,
                    "name": r.name,
                    "content": r.content,
                    "ok": r.ok,
                    "duration_s": r.duration_s,
                }
                for r in self.tool_results
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        return cls(
            role=Role(data["role"]),
            content=data.get("content", ""),
            name=data.get("name"),
            created_at=data.get("created_at", time.time()),
            tool_calls=[ToolCall(**c) for c in data.get("tool_calls", [])],
            tool_results=[ToolResult(**r) for r in data.get("tool_results", [])],
        )


@dataclass(slots=True)
class Usage:
    """Token accounting for one or more provider calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass(slots=True)
class LLMResponse:
    """A normalized completion from any provider."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: StopReason = StopReason.END_TURN
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    raw: Any = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)

    def as_message(self) -> Message:
        return Message.assistant(self.content, self.tool_calls)
