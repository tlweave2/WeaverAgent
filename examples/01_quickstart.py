"""Quickstart: an agent with tools and memory.

Run against Claude:

    pip install 'agentforge[anthropic]'
    export ANTHROPIC_API_KEY=...
    python examples/01_quickstart.py

Run with no API key at all (uses the mock provider):

    python examples/01_quickstart.py --mock
"""

from __future__ import annotations

import sys

from agentforge import Agent, MockProvider, SQLiteMemory
from agentforge.tools import default_registry


def main() -> None:
    use_mock = "--mock" in sys.argv

    if use_mock:
        provider = MockProvider(
            [
                MockProvider.tool_call("calculator", {"expression": "17*23"}),
                "17 x 23 = 391.",
            ]
        )
    else:
        provider = "anthropic"  # resolved by name; model defaults to claude-opus-5

    agent = Agent(
        provider,
        engine="react",
        tools=default_registry(),
        memory=SQLiteMemory("quickstart_memory.sqlite3"),
        session_id="quickstart",
    )

    with agent:
        trace = agent.run("What is 17 * 23?")

        print(trace.render())
        print(f"\ntool calls: {[str(c) for c in trace.tool_calls]}")
        print(f"tokens: {trace.usage.total_tokens}")

        # Memory persists across processes, so the next run sees this exchange.
        print(f"remembered turns: {len(agent.history())}")


if __name__ == "__main__":
    main()
