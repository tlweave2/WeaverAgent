"""Reasoning engines: ReAct and Plan-and-Execute."""

from .base import ReasoningEngine, Step, StepType, Trace
from .plan_execute import PlanExecuteEngine, PlanStep
from .react import ReActEngine

#: Names accepted by :class:`~weaveragent.agent.Agent` and the CLI.
ENGINES: dict[str, type[ReasoningEngine]] = {
    "react": ReActEngine,
    "plan_execute": PlanExecuteEngine,
    "plan-execute": PlanExecuteEngine,
}

__all__ = [
    "ENGINES",
    "PlanExecuteEngine",
    "PlanStep",
    "ReActEngine",
    "ReasoningEngine",
    "Step",
    "StepType",
    "Trace",
]
