"""Memory that survives a restart.

The store is reopened from disk partway through, which is the difference
between SQLiteMemory and EphemeralMemory.

    python examples/04_persistent_memory.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from weaveragent import Agent, MockProvider, SQLiteMemory


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "memory.sqlite3"

        # --- first process ---
        with Agent(
            MockProvider(["Noted -- you prefer metric units."]),
            memory=SQLiteMemory(path),
            session_id="alice",
        ) as agent:
            agent.run("Please use metric units from now on.")
            agent.remember("Alice prefers metric units", source="explicit request")

        # --- a separate process would start here ---
        with Agent(
            MockProvider(["Using metric, as you asked earlier."]),
            memory=SQLiteMemory(path),
            session_id="alice",
        ) as agent:
            print("recovered history:")
            for message in agent.history():
                print(f"  {message.role.value}: {message.content}")

            print(f"\nrecalled facts: {agent.recall('metric')}")
            print(f"\nnew answer: {agent.ask('How tall is a 6 foot person?')}")

        # Sessions are isolated: Bob sees none of Alice's history.
        with Agent(MockProvider(["Hello!"]), memory=SQLiteMemory(path), session_id="bob") as bob:
            print(f"\nbob's history: {bob.history()}")


if __name__ == "__main__":
    main()
