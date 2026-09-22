"""WeaverAgent -- a universal AI agent framework.

Four independent layers:

* :mod:`weaveragent.llm` -- a unified interface over model providers.
* :mod:`weaveragent.reasoning` -- ReAct and Plan-and-Execute engines.
* :mod:`weaveragent.tools` -- a pluggable tool registry.
* :mod:`weaveragent.memory` -- ephemeral and persistent memory stores.

:class:`~weaveragent.agent.Agent` composes them::

    from weaveragent import Agent
    from weaveragent.tools import default_registry

    agent = Agent(provider="anthropic", tools=default_registry())
    print(agent.ask("What is 17 * 23?"))
"""

from .agent import Agent
from .errors import (
    ConfigurationError,
    ProviderError,
    ProviderNotInstalled,
    ReasoningError,
    StepLimitExceeded,
    ToolError,
    ToolNotFound,
    WeaverAgentError,
)
from .llm import LLMConfig, LLMProvider, MockProvider, get_provider
from .memory import EphemeralMemory, MemoryRecord, MemoryStore, SQLiteMemory
from .reasoning import PlanExecuteEngine, ReActEngine, ReasoningEngine, Step, StepType, Trace
from .tools import Tool, ToolRegistry, default_registry, tool
from .types import (
    LLMResponse,
    Message,
    Role,
    StopReason,
    ToolCall,
    ToolResult,
    Usage,
)

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "ConfigurationError",
    "default_registry",
    "EphemeralMemory",
    "get_provider",
    "LLMConfig",
    "LLMProvider",
    "LLMResponse",
    "MemoryRecord",
    "MemoryStore",
    "Message",
    "MockProvider",
    "PlanExecuteEngine",
    "ProviderError",
    "ProviderNotInstalled",
    "ReActEngine",
    "ReasoningEngine",
    "ReasoningError",
    "Role",
    "SQLiteMemory",
    "Step",
    "StepLimitExceeded",
    "StepType",
    "StopReason",
    "Tool",
    "tool",
    "ToolCall",
    "ToolError",
    "ToolNotFound",
    "ToolRegistry",
    "ToolResult",
    "Trace",
    "Usage",
    "__version__",
    "WeaverAgentError",
]
