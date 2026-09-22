"""The Plan-and-Execute reasoning engine."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from ..types import Message, StopReason
from .base import ReasoningEngine, Step, StepType, Trace
from .react import ReActEngine

_PLANNER_PROMPT = """You are a planner. Break the user's task into an ordered \
list of concrete steps that an executor agent can carry out one at a time.

Rules:
- Between 1 and {max_steps} steps. Use the fewest that genuinely do the job.
- Each step must be a single self-contained instruction, not a vague heading.
- Order matters: a later step may rely on an earlier step's result.
- Do not include steps for restating or summarizing the answer; that happens \
automatically at the end.

Respond with JSON only, in exactly this form:
{{"steps": ["first step", "second step"]}}"""

_EXECUTOR_PROMPT = """You are executing one step of a larger plan.

Overall task: {task}

Plan:
{plan}

Results of completed steps:
{completed}

Carry out ONLY the current step. Use tools where they help. Report what you \
found concisely -- a later step may depend on it."""

_SYNTHESIS_PROMPT = """You are answering the user's original task using the \
results gathered by executing a plan.

Task: {task}

Step results:
{results}

Write the final answer. Use the gathered results rather than re-deriving \
anything. If a step failed and that limits the answer, say so plainly."""


@dataclass(slots=True)
class PlanStep:
    """One step of a plan, plus its outcome once executed."""

    description: str
    result: str = ""
    succeeded: bool = False
    executed: bool = False
    sub_steps: list[Step] = field(default_factory=list)


class PlanExecuteEngine(ReasoningEngine):
    """Plan the whole task up front, then execute each step in turn.

    Compared with :class:`~agentforge.reasoning.react.ReActEngine`, this trades
    adaptivity for structure: the model commits to an approach before spending
    tool calls, which suits multi-part tasks where a greedy loop tends to wander.
    Each step is executed by its own bounded ReAct sub-run, so steps can still
    use tools freely.

    Args:
        provider: The LLM to reason with.
        tools: Tools available during execution.
        max_steps: Maximum number of plan steps.
        system_prompt: Overrides the executor's system prompt.
        step_max_iterations: Model round trips allowed per step.
        replan_on_failure: Whether to re-plan the remaining steps after a
            step fails, instead of carrying the failure forward.
    """

    name = "plan_execute"

    def __init__(
        self,
        provider,
        tools=None,
        *,
        max_steps: int = 6,
        system_prompt: str | None = None,
        step_max_iterations: int = 5,
        replan_on_failure: bool = True,
    ) -> None:
        super().__init__(provider, tools, max_steps=max_steps, system_prompt=system_prompt)
        self.step_max_iterations = step_max_iterations
        self.replan_on_failure = replan_on_failure

    def run(self, task: str, *, history: list[Message] | None = None) -> Trace:
        trace = Trace(task=task, engine=self.name)

        plan = self._make_plan(task, trace, history=history)
        if not plan:
            trace.add_step(StepType.ERROR, "planner produced no usable steps")
            trace.finished_at = time.time()
            return trace

        for index, step in enumerate(plan):
            self._execute_step(task, plan, index, trace)

            if not step.succeeded and self.replan_on_failure:
                remaining = plan[index + 1 :]
                if remaining:
                    revised = self._replan(task, plan, index, trace)
                    if revised:
                        plan = plan[: index + 1] + revised

        trace.answer = self._synthesize(task, plan, trace)
        trace.succeeded = any(s.succeeded for s in plan) and bool(trace.answer)
        trace.finished_at = time.time()
        return trace

    # -- planning -------------------------------------------------------------

    def _make_plan(
        self, task: str, trace: Trace, *, history: list[Message] | None = None
    ) -> list[PlanStep]:
        messages = [*(history or []), Message.user(task)]
        response = self.provider.complete(
            messages,
            tools=None,  # the planner decides what to do, it does not act
            system=_PLANNER_PROMPT.format(max_steps=self.max_steps),
        )
        trace.usage = trace.usage + response.usage

        if response.stop_reason is StopReason.REFUSAL:
            trace.add_step(StepType.ERROR, response.content, metadata={"refusal": True})
            return []

        descriptions = _parse_plan(response.content)[: self.max_steps]
        if not descriptions:
            return []

        trace.add_step(
            StepType.PLAN,
            "\n".join(f"{i}. {d}" for i, d in enumerate(descriptions, 1)),
            metadata={"step_count": len(descriptions)},
        )
        return [PlanStep(description=d) for d in descriptions]

    def _replan(
        self, task: str, plan: list[PlanStep], failed_index: int, trace: Trace
    ) -> list[PlanStep]:
        """Rebuild the remaining steps after a failure."""
        done = _format_completed(plan[: failed_index + 1])
        response = self.provider.complete(
            [
                Message.user(
                    f"Task: {task}\n\nSteps attempted so far:\n{done}\n\n"
                    f"Step {failed_index + 1} did not succeed. Give a revised list of "
                    "the REMAINING steps needed to finish the task. If the task can "
                    'already be answered, reply with {"steps": []}.'
                )
            ],
            tools=None,
            system=_PLANNER_PROMPT.format(max_steps=self.max_steps),
        )
        trace.usage = trace.usage + response.usage

        descriptions = _parse_plan(response.content)
        budget = max(0, self.max_steps - (failed_index + 1))
        descriptions = descriptions[:budget]
        if not descriptions:
            return []

        trace.add_step(
            StepType.PLAN,
            "Revised remaining steps:\n"
            + "\n".join(f"{i}. {d}" for i, d in enumerate(descriptions, 1)),
            metadata={"replanned_after": failed_index},
        )
        return [PlanStep(description=d) for d in descriptions]

    # -- execution ------------------------------------------------------------

    def _execute_step(self, task: str, plan: list[PlanStep], index: int, trace: Trace) -> None:
        step = plan[index]
        executor = ReActEngine(
            self.provider,
            self.tools,
            max_steps=self.step_max_iterations,
            system_prompt=self.system_prompt
            or _EXECUTOR_PROMPT.format(
                task=task,
                plan="\n".join(f"{i}. {s.description}" for i, s in enumerate(plan, 1)),
                completed=_format_completed(plan[:index]) or "(none yet)",
            ),
        )

        trace.add_step(
            StepType.THOUGHT,
            f"Executing step {index + 1}/{len(plan)}: {step.description}",
            metadata={"plan_step": index},
        )

        sub_trace = executor.run(step.description)
        trace.usage = trace.usage + sub_trace.usage

        step.executed = True
        step.result = sub_trace.answer
        step.succeeded = sub_trace.succeeded
        step.sub_steps = sub_trace.steps

        # Flatten the sub-run into the parent trace so one trace shows the
        # whole run, tagged with the plan step each entry belongs to.
        for sub_step in sub_trace.steps:
            trace.add(
                Step(
                    type=sub_step.type,
                    content=sub_step.content,
                    tool_calls=sub_step.tool_calls,
                    tool_results=sub_step.tool_results,
                    metadata={**sub_step.metadata, "plan_step": index},
                )
            )

        if not step.succeeded:
            trace.add_step(
                StepType.ERROR,
                f"step {index + 1} did not produce a confident result",
                metadata={"plan_step": index},
            )

    # -- synthesis ------------------------------------------------------------

    def _synthesize(self, task: str, plan: list[PlanStep], trace: Trace) -> str:
        response = self.provider.complete(
            [Message.user(task)],
            tools=None,
            system=_SYNTHESIS_PROMPT.format(task=task, results=_format_completed(plan)),
        )
        trace.usage = trace.usage + response.usage
        trace.add_step(StepType.ANSWER, response.content)
        return response.content


def _format_completed(steps: list[PlanStep]) -> str:
    lines = []
    for index, step in enumerate(steps, 1):
        if not step.executed:
            continue
        status = "ok" if step.succeeded else "failed"
        result = step.result or "(no result)"
        lines.append(f"{index}. [{status}] {step.description}\n   -> {result}")
    return "\n".join(lines)


def _parse_plan(text: str) -> list[str]:
    """Pull an ordered step list out of a planner response.

    The JSON object is tried first, then a bare array, then numbered or
    bulleted lines. Planners drift in and out of strict JSON even when asked
    for it, and a formatting slip should not abort a run.
    """
    if not text:
        return []

    for candidate in _json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            steps = parsed.get("steps") or parsed.get("plan")
        else:
            steps = parsed
        if isinstance(steps, list):
            cleaned = [str(s).strip() for s in steps if str(s).strip()]
            if cleaned:
                return cleaned

    return _parse_plan_lines(text)


def _json_candidates(text: str) -> list[str]:
    """Candidate JSON substrings, widest first, including fenced blocks."""
    candidates = [text.strip()]
    for match in re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL):
        candidates.append(match.strip())
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if 0 <= start < end:
            candidates.append(text[start : end + 1])
    return candidates


def _parse_plan_lines(text: str) -> list[str]:
    steps: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = re.match(r"^(?:\d+[.)]|[-*•])\s+(.*)$", stripped)
        if match:
            content = match.group(1).strip().strip('"').strip(",").strip('"')
            if content:
                steps.append(content)
    return steps
