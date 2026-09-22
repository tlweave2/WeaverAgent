"""Memory stores. Both backends are held to the same contract."""

import pytest

from agentforge.memory import EphemeralMemory, MemoryRecord, SQLiteMemory
from agentforge.types import Message, ToolCall, ToolResult


@pytest.fixture(params=["ephemeral", "sqlite"])
def store(request, tmp_path):
    """Each test runs against every backend, so they cannot drift apart."""
    if request.param == "ephemeral":
        yield EphemeralMemory()
    else:
        memory = SQLiteMemory(tmp_path / "memory.sqlite3")
        yield memory
        memory.close()


def test_add_and_recent_returns_oldest_first(store):
    for text in ("one", "two", "three"):
        store.add_message(Message.user(text))
    assert [r.content for r in store.recent()] == ["one", "two", "three"]


def test_recent_respects_limit_and_keeps_the_tail(store):
    for index in range(10):
        store.add_message(Message.user(str(index)))
    assert [r.content for r in store.recent(limit=3)] == ["7", "8", "9"]


def test_sessions_are_isolated(store):
    store.add_message(Message.user("in a"), session_id="a")
    store.add_message(Message.user("in b"), session_id="b")

    assert [r.content for r in store.recent(session_id="a")] == ["in a"]
    assert [r.content for r in store.recent(session_id="b")] == ["in b"]
    assert store.sessions() == ["a", "b"]


def test_facts_are_separate_from_messages(store):
    store.add_message(Message.user("a message"))
    store.add_fact("a durable fact", source="test")

    assert [r.content for r in store.recent(kind="message")] == ["a message"]
    assert [r.content for r in store.recent(kind="fact")] == ["a durable fact"]
    assert len(store.recent(kind=None)) == 2
    assert store.recent(kind="fact")[0].metadata["source"] == "test"


def test_history_rebuilds_messages_with_tool_calls_intact(store):
    store.add_message(Message.user("compute 2+2"))
    store.add_message(Message.assistant("calling", [ToolCall("calc", {"e": "2+2"}, id="c1")]))
    store.add_message(Message.tool([ToolResult("c1", "calc", "4")]))

    history = store.history()
    assert [m.role.value for m in history] == ["user", "assistant", "tool"]
    assert history[1].tool_calls[0].arguments == {"e": "2+2"}
    assert history[2].tool_results[0].content == "4"


def test_search_finds_matching_records(store):
    store.add_message(Message.user("the capital of France is Paris"))
    store.add_message(Message.user("unrelated chatter about weather"))

    hits = store.search("capital France")
    assert len(hits) == 1
    assert "Paris" in hits[0].content


def test_search_is_scoped_to_the_session(store):
    store.add_message(Message.user("secret in session a"), session_id="a")
    assert store.search("secret", session_id="b") == []
    assert len(store.search("secret", session_id="a")) == 1
    assert len(store.search("secret", session_id=None)) == 1


def test_empty_query_returns_nothing(store):
    store.add_message(Message.user("content"))
    assert store.search("   ") == []


def test_clear_one_session_leaves_others(store):
    store.add_message(Message.user("a"), session_id="a")
    store.add_message(Message.user("b1"), session_id="b")
    store.add_message(Message.user("b2"), session_id="b")

    assert store.clear(session_id="b") == 2
    assert store.sessions() == ["a"]


def test_clear_all_sessions(store):
    store.add_message(Message.user("a"), session_id="a")
    store.add_message(Message.user("b"), session_id="b")
    assert store.clear() == 2
    assert store.sessions() == []


def test_search_tolerates_query_operator_characters(store):
    """FTS5 rejects some unescaped syntax; the store must fall back, not raise."""
    store.add_message(Message.user("a note about parentheses"))
    assert store.search('note AND (parentheses OR "unbalanced') is not None


def test_sqlite_persists_across_reopen(tmp_path):
    """This is what separates SQLiteMemory from EphemeralMemory."""
    path = tmp_path / "persist.sqlite3"

    first = SQLiteMemory(path)
    first.add_message(Message.user("remember me"), session_id="s1")
    first.add_fact("a fact worth keeping", session_id="s1")
    first.close()

    second = SQLiteMemory(path)
    try:
        assert [m.content for m in second.history(session_id="s1")] == ["remember me"]
        assert second.recent(session_id="s1", kind="fact")[0].content == "a fact worth keeping"
        assert len(second) == 2
    finally:
        second.close()


def test_sqlite_context_manager_closes():
    with SQLiteMemory(":memory:") as memory:
        memory.add_message(Message.user("x"))
        assert len(memory) == 1


def test_ephemeral_max_records_evicts_oldest():
    memory = EphemeralMemory(max_records=3)
    for index in range(6):
        memory.add_message(Message.user(str(index)))
    assert [r.content for r in memory.recent()] == ["3", "4", "5"]


def test_ephemeral_max_records_is_per_session():
    memory = EphemeralMemory(max_records=2)
    for index in range(3):
        memory.add_message(Message.user(f"a{index}"), session_id="a")
        memory.add_message(Message.user(f"b{index}"), session_id="b")
    assert len(memory.recent(session_id="a")) == 2
    assert len(memory.recent(session_id="b")) == 2


def test_record_without_message_payload_is_skipped_by_history(store):
    store.add(MemoryRecord(content="raw", kind="message"))
    assert store.history() == []
    assert len(store.recent()) == 1
