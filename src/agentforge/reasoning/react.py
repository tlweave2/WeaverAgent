"""The ReAct reasoning engine."""

from __future__ import annotations

import time

from ..types import LLMResponse, Message, StopReason
from .base import ReasoningEngine, Step, StepType, Trace

_SYSTEM_PROMPT = """You are a reasoning agent that solves tasks by interleaving \
thinking with tool use.

Work in a loop:
1. Reason about what you know and what you still need.
2. If a tool would get you closer, call it. You may call several tools at once \
when they are independent.
3. Read the results and repeat.

Guidelines:
- Prefer calling a tool over guessing at anything a tool can determine.
- If a tool returns an error, read it and adjust; do not retry the same call unchanged.
- When you have enough to answer, stop calling tools and give the answer directly.
- Do not describe a tool call in prose instead of making it."""


class ReActEngine(ReasoningEngine):
    """Reason and act in a loop until the model stops calling tools.

    Tool calls go through the provider's native tool-calling interface rather
    than parsing ``Thought:``/``Action:`` text out of a completion. The
    Thought-Action-Observation structure is still what gets recorded in the
    :class:`~agentforge.reasoning.base.Trace`, but the model never has to
    produce a fragile text format, and malformed actions stop being a failure
    mode.
    """

    name = "react"

    def default_system_prompt(self) -> str:
        prompt = _SYSTEM_PROMPT
        if len(self.tools):
            prompt += f"\n\nAvailable tools:\n{self.tools.describe()}"
        return prompt

    def run(self, task: str, *, history: list[Message] | None = None) -> Trace:
        trace = Trace(task=task, engine=self.name)
        messages: list[Message] = [*(history or []), Message.user(task)]
        schemas = self.tools.schemas() or None

        for step_index in range(self.max_steps):
            response = self.provider.complete(messages, tools=schemas, system=self._system())
            trace.usage = trace.usage + response.usage

            if response.stop_reason is StopReason.REFUSAL:
                trace.add_step(StepType.ERROR, response.content, metadata={"refusal": True})
                trace.answer = response.content
                trace.finished_at = time.time()
                return trace

            if response.content:
                trace.add_step(StepType.THOUGHT, response.content, metadata={"step": step_index})

            if not response.wants_tools:
                trace.answer = response.content
                trace.succeeded = bool(response.content)
                trace.add_step(StepType.ANSWER, response.content)
                trace.finished_at = time.time()
                return trace

            messages.append(response.as_message())
            trace.add(
                Step(
                    type=StepType.ACTION,
                    content=", ".join(str(c) for c in response.tool_calls),
                    tool_calls=list(response.tool_calls),
                    metadata={"step": step_index},
                )
            )

            results = self.tools.invoke_all(response.tool_calls)
            messages.append(Message.tool(results))
            trace.add(
                Step(
                    type=StepType.OBSERVATION,
                    content="\n".join(f"{r.name}: {r}" for r in results),
                    tool_results=results,
                    metadata={"step": step_index, "errors": sum(r.is_error for r in results)},
                )
            )

        # Out of steps. Rather than fail with nothing, ask once for the best
        # answer available from what the run already gathered.
        trace.add_step(
            StepType.ERROR,
            f"reached the {self.max_steps}-step limit; requesting a final answer",
        )
        self._force_answer(messages, trace)
        trace.finished_at = time.time()
        return trace

    def _force_answer(self, messages: list[Message], trace: Trace) -> LLMResponse:
        """Ask for a conclusion with tools withheld, so the model must answer."""
        messages = [
            *messages,
            Message.user(
                "You have reached the step limit and cannot call more tools. "
                "Give the best answer you can from what you have gathered, and "
                "state explicitly what remains unresolved."
            ),
        ]
        response = self.provider.complete(messages, tools=None, system=self._system())
        trace.usage = trace.usage + response.usage
        trace.answer = response.content
        trace.succeeded = False
        trace.add_step(StepType.ANSWER, response.content, metadata={"truncated": True})
        return response
