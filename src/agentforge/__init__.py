"""AgentForge -- a universal AI agent framework.

Four independent layers:

* :mod:`agentforge.llm` -- a unified interface over model providers.
* :mod:`agentforge.reasoning` -- ReAct and Plan-and-Execute engines.
* :mod:`agentforge.tools` -- a pluggable tool registry.
* :mod:`agentforge.memory` -- ephemeral and persistent memory stores.

:class:`~agentforge.agent.Agent` composes them::

    from agentforge import Agent
    from agentforge.tools import default_registry

    agent = Agent(provider="anthropic", tools=default_registry())
    print(agent.ask("What is 17 * 23?"))
"""

from .agent import Agent
from .errors import (
    AgentForgeError,
    ConfigurationError,
    ProviderError,
    ProviderNotInstalled,
    ReasoningError,
    StepLimitExceeded,
    ToolError,
    ToolNotFound,
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
    "AgentForgeError",
    "ConfigurationError",
    "EphemeralMemory",
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
    "ToolCall",
    "ToolError",
    "ToolNotFound",
    "ToolRegistry",
    "ToolResult",
    "Trace",
    "Usage",
    "__version__",
    "default_registry",
    "get_provider",
    "tool",
]
