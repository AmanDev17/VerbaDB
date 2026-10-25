"""A scripted stand-in for the model, so the agent loop is tested with no network and no API key."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple, Union

from verbadb.llm import FunctionCall, Turn


def call(tool: str, /, **args: Any) -> FunctionCall:
    """A scripted tool call; ``tool`` is positional-only so a tool argument may itself be called ``name``."""
    return FunctionCall(name=tool, args=args, id=None)


def model_turn(text: str = "", *calls: FunctionCall) -> Turn:
    raw = {"role": "model", "text": text, "calls": [(c.name, c.args) for c in calls]}
    return Turn(text=text, calls=list(calls), raw=raw, finish_reason="STOP", usage={"total_tokens": 100})


class FakeLLM:
    model = "fake-model"

    def __init__(self, script: Sequence[Union[Turn, Exception]]) -> None:
        self.script: List[Union[Turn, Exception]] = list(script)
        self.requests: List[Dict[str, Any]] = []

    def user_message(self, text: str) -> Any:
        return {"role": "user", "text": text}

    def tool_results(self, results: Sequence[Tuple[FunctionCall, Dict[str, Any]]]) -> Any:
        return {"role": "tool", "results": [(c.name, r) for c, r in results]}

    def generate(self, system: str, tools: Sequence[Any], history: Sequence[Any]) -> Turn:
        self.requests.append({"system": system, "tools": [t.name for t in tools], "history": list(history)})
        if not self.script:
            raise AssertionError("the agent asked the model more times than the test scripted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
