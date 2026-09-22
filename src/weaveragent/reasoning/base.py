"""The reasoning engine interface and execution trace."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from ..types import Message, ToolCall, ToolResult, Usage

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..llm.base import LLMProvider
    from ..tools.registry import ToolRegistry


class StepType(str, Enum):
    """What happened at one point in a run."""

    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    PLAN = "plan"
    ANSWER = "answer"
    ERROR = "error"


@dataclass(slots=True)
class Step:
    """One entry in a :class:`Trace`."""

    type: StepType
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def render(self) -> str:
        label = self.type.value.capitalize()
        if self.type is StepType.ACTION and self.tool_calls:
            calls = ", ".join(str(c) for c in self.tool_calls)
            return f"{label}: {calls}"
        return f"{label}: {self.content}"


@dataclass(slots=True)
class Trace:
    """The full record of a reasoning run.

    Every engine returns one of these, so a run is inspectable after the fact:
    what the model decided, which tools it called, what came back, and what it
    cost.
    """

    task: str
    steps: list[Step] = field(default_factory=list)
    answer: str = ""
    usage: Usage = field(default_factory=Usage)
    engine: str = ""
    succeeded: bool = False
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def add(self, step: Step) -> Step:
        self.steps.append(step)
        return step

    def add_step(self, type: StepType, content: str, **kwargs: Any) -> Step:
        return self.add(Step(type=type, content=content, **kwargs))

    @property
    def duration_s(self) -> float:
        return (self.finished_at or time.time()) - self.started_at

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Every tool call made during the run, in order."""
        return [call for step in self.steps for call in step.tool_calls]

    def steps_of(self, type: StepType) -> list[Step]:
        return [s for s in self.steps if s.type is type]

    def render(self) -> str:
        """A human-readable transcript of the run."""
        lines = [f"Task: {self.task}", f"Engine: {self.engine}", ""]
        lines.extend(step.render() for step in self.steps)
        lines.extend(
            [
                "",
                f"Answer: {self.answer or '(none)'}",
                f"({len(self.steps)} steps, {self.usage.total_tokens} tokens, "
                f"{self.duration_s:.2f}s)",
            ]
        )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()


class ReasoningEngine(ABC):
    """A strategy for turning a task into an answer using an LLM and tools.

    Engines are interchangeable: an :class:`~weaveragent.agent.Agent` holds one
    and delegates to it, so swapping ReAct for Plan-Execute changes no other
    code.

    Args:
        provider: The LLM to reason with.
        tools: Tools the model may call. An empty registry means no tool use.
        max_steps: Ceiling on model round trips, to bound cost and stop loops.
        system_prompt: Replaces the engine's default system prompt entirely.
    """

    name = "base"

    def __init__(
        self,
        provider: LLMProvider,
        tools: ToolRegistry | None = None,
        *,
        max_steps: int = 10,
        system_prompt: str | None = None,
    ) -> None:
        from ..tools.registry import ToolRegistry

        self.provider = provider
        self.tools = tools if tools is not None else ToolRegistry()
        self.max_steps = max_steps
        self.system_prompt = system_prompt

    @abstractmethod
    def run(self, task: str, *, history: list[Message] | None = None) -> Trace:
        """Work the task to an answer and return the trace.

        Args:
            task: What the agent should do.
            history: Prior conversation turns to condition on.
        """

    def _system(self) -> str:
        """The system prompt for this run."""
        return self.system_prompt or self.default_system_prompt()

    def default_system_prompt(self) -> str:
        return "You are a capable assistant. Use the available tools when they help."

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(provider={self.provider.name!r}, "
            f"tools={len(self.tools)}, max_steps={self.max_steps})"
        )
