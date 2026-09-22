"""Tool schema inference, invocation, and the registry."""

import json

import pytest

from weaveragent.errors import ToolNotFound
from weaveragent.tools import ToolRegistry, default_registry, make_file_tools, tool
from weaveragent.types import ToolCall


def test_schema_inferred_from_signature_and_docstring():
    @tool
    def search(query: str, limit: int = 10, exact: bool = False) -> str:
        """Search the index.

        Args:
            query: What to look for.
            limit: Maximum results.
            exact: Whether to require an exact match.
        """
        return query

    schema = search.to_schema()
    assert schema["name"] == "search"
    assert schema["description"] == "Search the index."
    props = schema["parameters"]["properties"]
    assert props["query"] == {"type": "string", "description": "What to look for."}
    assert props["limit"]["type"] == "integer"
    assert props["exact"]["type"] == "boolean"
    # Only the parameter without a default is required.
    assert schema["parameters"]["required"] == ["query"]


def test_optional_and_collection_annotations():
    @tool
    def fn(tags: list[str], note: str | None = None, mapping: dict | None = None) -> str:
        """Do a thing."""
        return "ok"

    props = fn.parameters["properties"]
    assert props["tags"] == {"type": "array", "items": {"type": "string"}}
    assert props["note"] == {"type": "string"}  # Optional[str] unwraps to str
    assert props["mapping"] == {"type": "object"}
    assert fn.parameters["required"] == ["tags"]


def test_decorator_accepts_overrides():
    @tool(name="renamed", description="Custom.", tags=("x",))
    def original() -> str:
        """Ignored."""
        return "v"

    assert original.name == "renamed"
    assert original.description == "Custom."
    assert original.tags == ("x",)


def test_tool_error_becomes_failed_result_not_exception():
    @tool
    def explode() -> str:
        """Always fails."""
        raise RuntimeError("boom")

    result = explode.invoke("c1", {})
    assert result.is_error
    assert "RuntimeError: boom" in result.content
    assert result.tool_call_id == "c1"


def test_missing_and_unknown_arguments_are_rejected():
    @tool
    def needs(value: str) -> str:
        """Needs a value."""
        return value

    assert "missing required argument" in needs.invoke("c", {}).content
    assert "unexpected argument" in needs.invoke("c", {"value": "v", "other": 1}).content


def test_non_string_return_is_serialized():
    @tool
    def data() -> dict:
        """Return a mapping."""
        return {"a": [1, 2]}

    assert json.loads(data.invoke("c", {}).content) == {"a": [1, 2]}


def test_registry_add_get_remove_and_describe():
    registry = ToolRegistry()

    @registry.register
    def ping() -> str:
        """Reply with pong."""
        return "pong"

    assert registry.names() == ["ping"]
    assert "ping" in registry
    assert "Reply with pong." in registry.describe()
    assert len(registry) == 1

    registry.remove("ping")
    assert registry.names() == []
    with pytest.raises(ToolNotFound):
        registry.get("ping")


def test_registry_rejects_undecorated_function():
    with pytest.raises(TypeError, match="use @tool"):
        ToolRegistry().add(lambda: None)  # type: ignore[arg-type]


def test_unknown_tool_call_returns_failed_result():
    result = ToolRegistry().invoke(ToolCall("ghost", {}))
    assert result.is_error
    assert "no tool named 'ghost'" in result.content


def test_invoke_all_returns_one_result_per_call():
    registry = default_registry()
    calls = [
        ToolCall("calculator", {"expression": "1+1"}),
        ToolCall("calculator", {"expression": "bad!"}),
        ToolCall("missing", {}),
    ]
    results = registry.invoke_all(calls)
    assert len(results) == 3
    assert [r.ok for r in results] == [True, False, False]
    assert [r.tool_call_id for r in results] == [c.id for c in calls]


def test_subset_and_merge():
    registry = default_registry()
    assert registry.subset(["calculator"]).names() == ["calculator"]
    assert registry.subset(tags=["math"]).names() == ["calculator"]

    other = ToolRegistry([tool(lambda: "x", name="extra")])
    assert "extra" in registry.merge(other)
    assert "extra" not in registry  # merge does not mutate


@pytest.mark.parametrize(
    "expression,expected",
    [("(17*23)+4", "395"), ("2**10", "1024"), ("-7 + 3", "-4"), ("7/2", "3.5")],
)
def test_calculator_arithmetic(expression, expected):
    registry = default_registry()
    assert registry.invoke(ToolCall("calculator", {"expression": expression})).content == expected


@pytest.mark.parametrize(
    "expression", ["__import__('os').system('echo hi')", "open('/etc/passwd')", "9**9**9"]
)
def test_calculator_refuses_non_arithmetic(expression):
    """The calculator parses an AST rather than using eval, so code cannot run."""
    result = default_registry().invoke(ToolCall("calculator", {"expression": expression}))
    assert result.is_error


def test_json_extract_dotted_path():
    doc = json.dumps({"user": {"roles": [{"name": "admin"}]}})
    registry = default_registry()
    out = registry.invoke(ToolCall("json_extract", {"document": doc, "path": "user.roles.0.name"}))
    assert json.loads(out.content) == "admin"


def test_file_tools_are_confined_to_root(tmp_path):
    (tmp_path / "inside.txt").write_text("visible", encoding="utf-8")
    (tmp_path.parent / "outside.txt").write_text("secret", encoding="utf-8")

    registry = ToolRegistry(make_file_tools(tmp_path, writable=True))
    assert registry.names() == ["list_files", "read_file", "write_file"]

    ok = registry.invoke(ToolCall("read_file", {"path": "inside.txt"}))
    assert ok.content == "visible"

    escaped = registry.invoke(ToolCall("read_file", {"path": "../outside.txt"}))
    assert escaped.is_error
    assert "escapes the allowed root" in escaped.content


def test_file_tools_are_read_only_by_default(tmp_path):
    registry = ToolRegistry(make_file_tools(tmp_path))
    assert "write_file" not in registry


def test_write_file_round_trips(tmp_path):
    registry = ToolRegistry(make_file_tools(tmp_path, writable=True))
    registry.invoke(ToolCall("write_file", {"path": "sub/new.txt", "content": "hello"}))
    assert (tmp_path / "sub" / "new.txt").read_text(encoding="utf-8") == "hello"


@pytest.mark.asyncio
async def test_async_tool_and_concurrent_invocation():
    import asyncio

    @tool
    async def slow(value: str) -> str:
        """Sleep briefly then echo."""
        await asyncio.sleep(0.01)
        return value

    registry = ToolRegistry([slow])
    assert slow.is_async

    results = await registry.ainvoke_all(
        [ToolCall("slow", {"value": "a"}), ToolCall("slow", {"value": "b"})]
    )
    assert [r.content for r in results] == ["a", "b"]


def test_sync_tool_works_through_async_path():
    import asyncio

    registry = default_registry()
    result = asyncio.run(registry.ainvoke(ToolCall("calculator", {"expression": "6*7"})))
    assert result.content == "42"
