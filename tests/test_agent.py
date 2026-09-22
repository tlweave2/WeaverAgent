"""The Agent facade, settings, and the CLI."""

import sqlite3

import pytest

from weaveragent import Agent, MockProvider, SQLiteMemory
from weaveragent.config import Settings
from weaveragent.errors import ConfigurationError
from weaveragent.reasoning import PlanExecuteEngine, ReActEngine
from weaveragent.tools import ToolRegistry, default_registry, tool


def test_agent_composes_the_four_layers():
    agent = Agent(MockProvider(["hi"]), tools=default_registry())
    assert agent.provider.name == "mock"
    assert isinstance(agent.engine, ReActEngine)
    assert len(agent.tools) == 3
    assert "EphemeralMemory" in repr(agent)


def test_agent_runs_a_task_end_to_end():
    provider = MockProvider([MockProvider.tool_call("calculator", {"expression": "17*23"}), "391."])
    agent = Agent(provider, tools=default_registry())

    trace = agent.run("What is 17 * 23?")
    assert trace.answer == "391."
    assert trace.succeeded
    assert [c.name for c in trace.tool_calls] == ["calculator"]


def test_ask_returns_only_the_answer():
    agent = Agent(MockProvider(["just text"]))
    assert agent.ask("q") == "just text"


def test_engine_selected_by_name():
    assert isinstance(Agent(MockProvider(), engine="react").engine, ReActEngine)
    assert isinstance(Agent(MockProvider(), engine="plan_execute").engine, PlanExecuteEngine)
    # The hyphenated spelling is accepted too.
    assert isinstance(Agent(MockProvider(), engine="plan-execute").engine, PlanExecuteEngine)


def test_engine_accepts_a_class_or_an_instance():
    provider = MockProvider()
    assert isinstance(Agent(provider, engine=PlanExecuteEngine).engine, PlanExecuteEngine)

    prebuilt = ReActEngine(provider, max_steps=42)
    assert Agent(provider, engine=prebuilt).engine is prebuilt


def test_unknown_engine_is_a_configuration_error():
    with pytest.raises(ConfigurationError, match="unknown engine"):
        Agent(MockProvider(), engine="telepathy")
    with pytest.raises(ConfigurationError, match="cannot use"):
        Agent(MockProvider(), engine=42)  # type: ignore[arg-type]


def test_provider_can_be_named():
    assert Agent("mock").provider.name == "mock"


def test_tools_accept_a_list_or_a_registry():
    @tool
    def solo() -> str:
        """A single tool."""
        return "x"

    assert Agent(MockProvider(), tools=[solo]).tools.names() == ["solo"]
    assert isinstance(Agent(MockProvider(), tools=ToolRegistry()).tools, ToolRegistry)
    assert len(Agent(MockProvider()).tools) == 0


def test_add_tool_is_visible_on_the_next_run():
    provider = MockProvider(["a", "b"])
    agent = Agent(provider, tools=default_registry())

    @tool
    def extra() -> str:
        """An extra tool."""
        return "x"

    agent.add_tool(extra)
    agent.run("task")
    assert "extra" in provider.calls[-1]["tools"]


# -- memory integration ------------------------------------------------------


def test_run_writes_the_exchange_to_memory():
    agent = Agent(MockProvider(["answered"]))
    agent.run("asked")
    assert [m.content for m in agent.history()] == ["asked", "answered"]


def test_remember_false_skips_memory():
    agent = Agent(MockProvider(["answered"]))
    agent.run("asked", remember=False)
    assert agent.history() == []


def test_history_is_replayed_into_the_next_run():
    provider = MockProvider(["first answer", "second answer"])
    agent = Agent(provider)

    agent.run("first question")
    agent.run("second question")

    replayed = [m.content for m in provider.calls[1]["messages"]]
    assert replayed == ["first question", "first answer", "second question"]


def test_history_limit_zero_disables_replay():
    provider = MockProvider(["a", "b"])
    agent = Agent(provider, history_limit=0)
    agent.run("one")
    agent.run("two")
    assert [m.content for m in provider.calls[1]["messages"]] == ["two"]


def test_sessions_do_not_leak_between_agents():
    memory = SQLiteMemory(":memory:")
    try:
        first = Agent(MockProvider(["a"]), memory=memory, session_id="alice")
        second = Agent(MockProvider(["b"]), memory=memory, session_id="bob")

        first.run("alice asks")
        second.run("bob asks")

        assert [m.content for m in first.history()] == ["alice asks", "a"]
        assert [m.content for m in second.history()] == ["bob asks", "b"]
    finally:
        memory.close()


def test_remember_and_recall_facts():
    agent = Agent(MockProvider())
    agent.remember("the deploy key lives in vault path kv/prod")
    assert agent.recall("deploy key") == ["the deploy key lives in vault path kv/prod"]
    assert agent.recall("unrelated") == []


def test_reset_clears_the_session():
    agent = Agent(MockProvider(["a"]))
    agent.run("q")
    assert agent.reset() == 2
    assert agent.history() == []


def test_stream_yields_chunks_and_records_the_exchange():
    agent = Agent(MockProvider(["streamed reply here"]))
    chunks = list(agent.stream("question"))

    assert len(chunks) > 1
    assert "".join(chunks) == "streamed reply here"
    assert [m.content for m in agent.history()] == ["question", "streamed reply here"]


def test_agent_context_manager_closes_memory():
    memory = SQLiteMemory(":memory:")
    with Agent(MockProvider(["x"]), memory=memory) as agent:
        agent.run("q")
    # A closed SQLite connection rejects further use.
    with pytest.raises(sqlite3.ProgrammingError):
        len(memory)


# -- settings ----------------------------------------------------------------


def test_settings_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("WEAVERAGENT_PROVIDER", "mock")
    monkeypatch.setenv("WEAVERAGENT_ENGINE", "plan_execute")
    monkeypatch.setenv("WEAVERAGENT_MAX_STEPS", "3")
    monkeypatch.setenv("WEAVERAGENT_SESSION", "from-env")

    settings = Settings.from_env()
    assert settings.provider == "mock"
    assert settings.engine == "plan_execute"
    assert settings.max_steps == 3
    assert settings.session_id == "from-env"


def test_explicit_overrides_beat_the_environment(monkeypatch):
    monkeypatch.setenv("WEAVERAGENT_PROVIDER", "mock")
    assert Settings.from_env(provider="openai").provider == "openai"
    # None means "not specified", so the environment still wins.
    assert Settings.from_env(provider=None).provider == "mock"


def test_malformed_numeric_env_var_falls_back(monkeypatch):
    monkeypatch.setenv("WEAVERAGENT_MAX_STEPS", "not-a-number")
    assert Settings.from_env().max_steps == 10


def test_settings_build_an_agent(monkeypatch, tmp_path):
    monkeypatch.setenv("WEAVERAGENT_PROVIDER", "mock")
    monkeypatch.setenv("WEAVERAGENT_MEMORY_PATH", str(tmp_path / "m.sqlite3"))

    with Settings.from_env().build_agent() as agent:
        assert isinstance(agent.memory, SQLiteMemory)
        assert agent.provider.name == "mock"


def test_settings_without_a_path_use_ephemeral_memory():
    from weaveragent.memory import EphemeralMemory

    assert isinstance(Settings().build_memory(), EphemeralMemory)


# -- CLI ---------------------------------------------------------------------


def test_cli_lists_tools(capsys):
    from weaveragent.cli import main

    assert main(["tools"]) == 0
    assert "calculator" in capsys.readouterr().out


def test_cli_lists_tool_schemas_as_json(capsys):
    import json

    from weaveragent.cli import main

    assert main(["tools", "--json"]) == 0
    schemas = json.loads(capsys.readouterr().out)
    assert {s["name"] for s in schemas} == {"calculator", "current_time", "json_extract"}


def test_cli_lists_providers(capsys):
    from weaveragent.cli import main

    assert main(["providers"]) == 0
    assert "anthropic" in capsys.readouterr().out


def test_cli_tool_spec_selects_a_subset():
    from weaveragent.cli import _build_tools

    assert _build_tools("none").names() == []
    assert _build_tools("calculator").names() == ["calculator"]
    assert len(_build_tools("builtin")) == 3


def test_cli_tool_spec_rejects_an_unmatched_name():
    from weaveragent.cli import _build_tools
    from weaveragent.errors import WeaverAgentError

    with pytest.raises(WeaverAgentError, match="no builtin tools matched"):
        _build_tools("nonexistent")


def test_cli_file_root_adds_file_tools(tmp_path):
    from weaveragent.cli import _build_tools

    registry = _build_tools("none", str(tmp_path))
    assert registry.names() == ["list_files", "read_file"]


def test_cli_run_prints_the_answer(capsys):
    """`run` works against the mock provider with no API key."""
    from weaveragent.cli import main

    assert main(["run", "what", "is", "up", "--provider", "mock", "--tools", "none"]) == 0
    assert capsys.readouterr().out.strip() == "[mock] what is up"


def test_cli_run_can_print_the_full_trace(capsys):
    from weaveragent.cli import main

    assert main(["run", "hello", "--provider", "mock", "--trace"]) == 0
    out = capsys.readouterr().out
    assert "Task: hello" in out
    assert "Engine: react" in out


def test_cli_run_honors_the_engine_flag(capsys):
    from weaveragent.cli import main

    main(["run", "hello", "--provider", "mock", "--engine", "plan_execute", "--trace"])
    assert "Engine: plan_execute" in capsys.readouterr().out


def test_cli_run_persists_memory_to_disk(tmp_path, capsys):
    from weaveragent.cli import main
    from weaveragent.memory import SQLiteMemory

    path = tmp_path / "cli.sqlite3"
    main(["run", "remember this", "--provider", "mock", "--memory", str(path), "--session", "s"])
    capsys.readouterr()

    with SQLiteMemory(path) as memory:
        assert [m.content for m in memory.history(session_id="s")] == [
            "remember this",
            "[mock] remember this",
        ]
