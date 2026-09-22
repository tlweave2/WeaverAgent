"""ReAct vs Plan-and-Execute on the same task.

Both engines get the same provider and tools, so the difference in the traces
is purely the reasoning strategy.

    python examples/03_engines_compared.py
"""

from __future__ import annotations

from agentforge import MockProvider, PlanExecuteEngine, ReActEngine
from agentforge.tools import default_registry

TASK = "Compute 17 * 23, then tell me the current time."


def run_react() -> None:
    provider = MockProvider(
        [
            MockProvider.tool_call("calculator", {"expression": "17*23"}, text="First the math."),
            MockProvider.tool_call("current_time", {}, text="Now the time."),
            "17 * 23 = 391, and I have the current time.",
        ]
    )
    print(ReActEngine(provider, default_registry()).run(TASK).render())


def run_plan_execute() -> None:
    provider = MockProvider(
        [
            '{"steps": ["Compute 17 * 23", "Report the current time"]}',
            MockProvider.tool_call("calculator", {"expression": "17*23"}),
            "391",
            MockProvider.tool_call("current_time", {}),
            "the current UTC time",
            "17 * 23 = 391, reported alongside the current UTC time.",
        ]
    )
    print(PlanExecuteEngine(provider, default_registry()).run(TASK).render())


if __name__ == "__main__":
    print("=" * 70, "\nReAct: decide one step at a time\n", "=" * 70, sep="")
    run_react()
    print("\n", "=" * 70, "\nPlan-Execute: commit to a plan, then carry it out\n", "=" * 70, sep="")
    run_plan_execute()
