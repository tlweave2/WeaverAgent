"""Command line interface: ``weaveragent``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__
from .config import Settings
from .errors import WeaverAgentError
from .llm.registry import available_providers
from .reasoning import ENGINES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="weaveragent",
        description="Run a WeaverAgent agent from the command line.",
    )
    parser.add_argument("--version", action="version", version=f"weaveragent {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run a single task and print the answer")
    run.add_argument("task", nargs="+", help="the task for the agent")
    run.add_argument(
        "-p", "--provider", default=None, choices=available_providers(), help="LLM provider"
    )
    run.add_argument("-m", "--model", default=None, help="model id")
    run.add_argument(
        "-e", "--engine", default=None, choices=sorted(ENGINES), help="reasoning engine"
    )
    run.add_argument("--max-steps", type=int, default=None, help="model round trips allowed")
    run.add_argument("--session", default=None, help="memory session id")
    run.add_argument("--memory", default=None, help="SQLite path for persistent memory")
    run.add_argument(
        "--tools",
        default="builtin",
        help="'builtin', 'none', or a comma-separated subset of the builtin tool names",
    )
    run.add_argument(
        "--file-root",
        default=None,
        help="expose read-only file tools rooted at this directory",
    )
    run.add_argument("--trace", action="store_true", help="print the full reasoning trace")

    chat = subparsers.add_parser("chat", help="start an interactive session")
    for name, kwargs in (
        ("-p/--provider", {"default": None, "choices": available_providers()}),
        ("-m/--model", {"default": None}),
        ("-e/--engine", {"default": None, "choices": sorted(ENGINES)}),
        ("--session", {"default": None}),
        ("--memory", {"default": None}),
    ):
        chat.add_argument(*name.split("/"), **kwargs)
    chat.add_argument("--tools", default="builtin")

    tools_cmd = subparsers.add_parser("tools", help="list the built-in tools")
    tools_cmd.add_argument("--json", action="store_true", help="print full JSON schemas")

    subparsers.add_parser("providers", help="list the available providers")

    return parser


def _build_tools(spec: str, file_root: str | None = None):
    from .tools.builtin import default_registry, make_file_tools
    from .tools.registry import ToolRegistry

    if spec == "none":
        registry = ToolRegistry()
    elif spec == "builtin":
        registry = default_registry()
    else:
        wanted = [name.strip() for name in spec.split(",") if name.strip()]
        registry = default_registry().subset(wanted)
        if not len(registry):
            raise WeaverAgentError(
                f"no builtin tools matched {spec!r}; "
                f"available: {', '.join(default_registry().names())}"
            )

    if file_root:
        for item in make_file_tools(file_root):
            registry.add(item)
    return registry


def _agent_from_args(args: argparse.Namespace):
    settings = Settings.from_env(
        provider=getattr(args, "provider", None),
        model=getattr(args, "model", None),
        engine=getattr(args, "engine", None),
        max_steps=getattr(args, "max_steps", None),
        session_id=getattr(args, "session", None),
        memory_path=getattr(args, "memory", None),
    )
    tools = _build_tools(getattr(args, "tools", "builtin"), getattr(args, "file_root", None))
    return settings.build_agent(tools=tools)


def _cmd_run(args: argparse.Namespace) -> int:
    task = " ".join(args.task)
    with _agent_from_args(args) as agent:
        trace = agent.run(task)
        print(trace.render() if args.trace else (trace.answer or "(no answer)"))
    return 0 if trace.succeeded else 1


def _cmd_chat(args: argparse.Namespace) -> int:
    with _agent_from_args(args) as agent:
        print(f"{agent!r}\nType /exit to quit, /reset to clear memory.\n")
        while True:
            try:
                line = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not line:
                continue
            if line in ("/exit", "/quit"):
                return 0
            if line == "/reset":
                print(f"cleared {agent.reset()} record(s)\n")
                continue
            try:
                print(f"\nagent> {agent.ask(line)}\n")
            except WeaverAgentError as exc:
                print(f"\nerror: {exc}\n", file=sys.stderr)


def _cmd_tools(args: argparse.Namespace) -> int:
    from .tools.builtin import default_registry

    registry = default_registry()
    if args.json:
        import json

        print(json.dumps(registry.schemas(), indent=2))
    else:
        print(registry.describe())
    return 0


def _cmd_providers(_: argparse.Namespace) -> int:
    print("\n".join(available_providers()))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "run": _cmd_run,
        "chat": _cmd_chat,
        "tools": _cmd_tools,
        "providers": _cmd_providers,
    }
    try:
        return handlers[args.command](args)
    except WeaverAgentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
