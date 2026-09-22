# WeaverAgent

Universal AI agent framework featuring a unified LLM interface layer, ReAct and
Plan-Execute reasoning engines, a pluggable tool registry, and persistent memory
management.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

The four layers are independent. Any provider works with any reasoning engine,
any tool set, and any memory backend — swapping one changes no other code.

```python
from weaveragent import Agent
from weaveragent.tools import default_registry

agent = Agent(provider="anthropic", tools=default_registry())
print(agent.ask("What is 17 * 23?"))
```

## Install

```bash
pip install -e '.[anthropic]'     # Claude
pip install -e '.[openai]'        # GPT / any OpenAI-compatible endpoint
pip install -e '.[all,dev]'       # everything, plus pytest and ruff
```

The core package has **no required dependencies** — provider SDKs are optional
extras, imported only when that provider is actually used.

Credentials come from the environment (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or
an `ant auth login` profile); nothing needs to be passed in code.

## Try it without an API key

The built-in `mock` provider runs the whole stack in-process:

```bash
python -m weaveragent.cli run "hello" --provider mock --trace
python examples/03_engines_compared.py
pytest                             # 143 tests, no network, no key
```

## The four layers

### 1. Unified LLM interface

Every provider implements one interface and speaks one set of types, so nothing
above this layer touches a vendor SDK.

```python
from weaveragent.llm import get_provider, LLMConfig

claude = get_provider("anthropic", model="claude-opus-5")
gpt    = get_provider("openai", model="gpt-4o")
local  = get_provider("openai", base_url="http://localhost:11434/v1")  # Ollama, vLLM, ...

response = claude.complete(messages, tools=registry.schemas(), system="Be brief.")
response.content      # str
response.tool_calls   # list[ToolCall]  — same shape from every provider
response.stop_reason  # StopReason enum  — normalized vocabulary
response.usage        # Usage, addable across calls
```

The adapters absorb the differences that would otherwise leak upward:

| Concern | Handled by the adapter |
|---|---|
| Tool schemas | Neutral `parameters` → Anthropic `input_schema` / OpenAI `function` envelope |
| Tool results | Batched into one turn for Anthropic, split per-call for OpenAI |
| Stop reasons | Vendor strings → `StopReason` enum |
| Refusals | `stop_reason: "refusal"` is read *before* content, so a refusal never reads as an empty answer |
| Thinking | Adaptive thinking enabled on models that accept it, omitted on those that don't |
| Sampling | `temperature` dropped, with a warning, on models that reject it rather than 400-ing |
| Malformed tool arguments | Reported to the model as a failed call instead of crashing the run |

Provider-specific parameters with no portable equivalent go through
`LLMConfig.extra`:

```python
LLMConfig(extra={"output_config": {"effort": "high"},
                 "thinking": {"type": "adaptive", "display": "summarized"}})
```

Register your own provider with `register_provider("name", builder)`.

### 2. Reasoning engines

**ReAct** — reason, act, observe, repeat, until the model stops calling tools.
Tool calls go through the provider's native tool-calling interface rather than
parsing `Thought:`/`Action:` text out of a completion, so malformed actions stop
being a failure mode. At the step limit it makes one final tools-withheld call so
a truncated run still returns its best answer, flagged `truncated`.

**Plan-Execute** — plan the whole task up front, then execute each step in its
own bounded ReAct sub-run, then synthesize. Trades adaptivity for structure,
which suits multi-part tasks where a greedy loop wanders. Re-plans the remaining
steps when one fails.

```python
from weaveragent import Agent

Agent(provider, engine="react", tools=registry, max_steps=10)
Agent(provider, engine="plan_execute", tools=registry, max_steps=6)
```

Both return a `Trace` — the full record of the run:

```
Task: Compute 17 * 23, then tell me the current time.
Engine: plan_execute

Plan: 1. Compute 17 * 23
2. Report the current time
Thought: Executing step 1/2: Compute 17 * 23
Action: calculator({"expression": "17*23"})
Observation: calculator: 391
...
(12 steps, 16 tokens, 0.00s)
```

```python
trace.answer        # final text
trace.succeeded     # whether it finished, rather than hitting a limit
trace.steps         # list[Step] — THOUGHT / ACTION / OBSERVATION / PLAN / ANSWER / ERROR
trace.tool_calls    # every call made, in order
trace.usage         # accumulated tokens across the whole run
```

Subclass `ReasoningEngine` to add a strategy; `Agent` accepts a name, a class, or
an instance.

### 3. Pluggable tool registry

Schemas are derived from type hints and the docstring, so what the model sees
cannot drift away from the implementation.

```python
from weaveragent.tools import ToolRegistry, tool

@tool(tags=("search",))
def search(query: str, limit: int = 10) -> str:
    """Search the document index.

    Args:
        query: What to look for.
        limit: Maximum number of results.
    """
    return do_search(query, limit)

registry = ToolRegistry([search])
```

That yields:

```json
{"name": "search",
 "description": "Search the document index.",
 "parameters": {"type": "object",
                "properties": {"query": {"type": "string", "description": "What to look for."},
                               "limit": {"type": "integer", "description": "Maximum number of results."}},
                "required": ["query"],
                "additionalProperties": false}}
```

Registries compose — `subset(names=...)`, `subset(tags=...)`, and `merge(other)`
let one base registry serve several agents with different tool surfaces. Async
tools are supported (`ainvoke_all` runs a round of calls concurrently).

**A tool that raises becomes a failed `ToolResult`, not an exception.** The error
text goes back to the model as an observation, so it can adjust — which is the
behavior that makes multi-step runs survive a bad argument. Arguments are
validated against the schema before the call, since providers don't guarantee
schema-valid input unless strict mode is on.

Built-ins: `calculator` (parses an AST — `eval` is never used, and code can't
run through it), `current_time`, `json_extract`, and optional file tools confined
to an explicit root directory, read-only unless `writable=True`.

### 4. Persistent memory

```python
from weaveragent.memory import EphemeralMemory, SQLiteMemory

Agent(provider, memory=SQLiteMemory("memory.sqlite3"), session_id="alice")
```

Both backends implement one contract and are tested against the same suite.
`SQLiteMemory` survives process restarts and uses FTS5 for ranked search when the
Python build provides it, falling back to `LIKE` when it doesn't.

```python
agent.history()                      # prior turns, tool calls and results intact
agent.remember("prefers metric")     # a durable fact, outside the transcript
agent.recall("units")                # search
agent.reset()                        # clear this session
```

Sessions are isolated, so one store serves many users. Conversation turns
(`kind="message"`) and durable facts (`kind="fact"`) are kept separate, so
recalling history doesn't drag in every stored fact. Subclass `MemoryStore` to
plug in a vector store or Redis.

## CLI

```bash
weaveragent run "What is 17 * 23?" --trace     # run one task
weaveragent chat --memory ~/.weaveragent.db     # interactive, persistent
weaveragent tools --json                       # inspect generated schemas
weaveragent providers                          # list providers
```

Configurable by environment: `WEAVERAGENT_PROVIDER`, `WEAVERAGENT_MODEL`,
`WEAVERAGENT_ENGINE`, `WEAVERAGENT_MAX_STEPS`, `WEAVERAGENT_MAX_TOKENS`,
`WEAVERAGENT_SESSION`, `WEAVERAGENT_MEMORY_PATH`, `WEAVERAGENT_HISTORY_LIMIT`.
Explicit flags and code always win over the environment.

## Architecture

```
                        ┌─────────────┐
                        │    Agent    │   composes the four layers
                        └──────┬──────┘
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
    ┌────────────────┐ ┌──────────────┐ ┌────────────────┐
    │   Reasoning    │ │    Tools     │ │     Memory     │
    │  ReAct         │ │  registry    │ │  Ephemeral     │
    │  Plan-Execute  │ │  @tool       │ │  SQLite        │
    └───────┬────────┘ └──────────────┘ └────────────────┘
            │
            ▼
    ┌───────────────────────────────────────────┐
    │          LLM interface layer              │
    │  Anthropic  │  OpenAI-compatible │ Mock   │
    └───────────────────────────────────────────┘
```

Everything crossing a layer boundary is a plain dataclass from
`weaveragent.types` — `Message`, `ToolCall`, `ToolResult`, `LLMResponse`,
`Usage`, `StopReason`. No vendor type reaches the reasoning, tool, or memory
layers, which is what makes them portable.

```
src/weaveragent/
├── agent.py          Agent facade
├── config.py         environment-driven Settings
├── types.py          provider-neutral dataclasses
├── errors.py         exception hierarchy
├── cli.py            command line interface
├── llm/              base · anthropic · openai · mock · registry
├── reasoning/        base (Trace/Step) · react · plan_execute
├── tools/            base (@tool, schema inference) · registry · builtin
└── memory/           base · ephemeral · sqlite
```

## Examples

| File | Shows |
|---|---|
| `examples/01_quickstart.py` | An agent with tools and persistent memory |
| `examples/02_custom_tools.py` | Registering tools, generated schemas, tag subsets |
| `examples/03_engines_compared.py` | ReAct vs Plan-Execute on the same task |
| `examples/04_persistent_memory.py` | Memory recovered after a restart |

## Development

```bash
pip install -e '.[all,dev]'
pytest                    # 143 tests: no network, no API key
ruff check . && ruff format --check .
```

Tests run against the mock provider, so the reasoning engines are exercised
deterministically — including refusals, tool failures, step-limit truncation,
re-planning, and planner output that isn't valid JSON. Both memory backends run
against a shared suite so they can't drift apart.

## License

MIT — see [LICENSE](LICENSE).
