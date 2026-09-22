"""Registering your own tools.

Schemas are derived from type hints and the docstring, so the description the
model sees cannot drift away from the implementation.

    python examples/02_custom_tools.py
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from weaveragent import Agent, MockProvider
from weaveragent.tools import ToolRegistry, tool


@tool(tags=("text",))
def word_frequency(text: str, top_n: int = 5) -> dict:
    """Count the most frequent words in a piece of text.

    Args:
        text: The text to analyze.
        top_n: How many of the most frequent words to return.
    """
    from collections import Counter

    counts = Counter(word.strip(".,!?;:\"'").lower() for word in text.split())
    counts.pop("", None)
    return dict(counts.most_common(top_n))


@tool(tags=("network",))
def http_get_json(url: str, timeout: float = 10.0) -> dict:
    """Fetch a URL and parse the response as JSON.

    Args:
        url: An http:// or https:// URL.
        timeout: Seconds to wait before giving up.
    """
    if not url.startswith(("http://", "https://")):
        raise ValueError("only http:// and https:// URLs are allowed")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed: {exc.reason}") from exc


registry = ToolRegistry([word_frequency, http_get_json])


# The decorator form also works directly on a registry.
@registry.register
def reverse(text: str) -> str:
    """Reverse a string.

    Args:
        text: The text to reverse.
    """
    return text[::-1]


def main() -> None:
    print("Registered tools:")
    print(registry.describe())

    print("\nGenerated schema for word_frequency:")
    print(json.dumps(registry.get("word_frequency").to_schema(), indent=2))

    # Tags let one registry serve several agents with different tool surfaces.
    print(f"\noffline-only subset: {registry.subset(tags=['text']).names()}")

    agent = Agent(
        MockProvider(
            [
                MockProvider.tool_call(
                    "word_frequency", {"text": "the cat sat on the mat the end", "top_n": 2}
                ),
                "The most frequent words are 'the' (3) and 'cat' (1).",
            ]
        ),
        tools=registry,
    )
    print(f"\n{agent.ask('What are the most common words in that sentence?')}")


if __name__ == "__main__":
    main()
