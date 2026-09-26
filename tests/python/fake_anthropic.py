"""A fake Anthropic client that accepts exactly what the installed SDK accepts.

Every call is bound against the real method signature from the installed `anthropic` package
(`inspect.signature(...).bind(...)`), so an argument the SDK would reject (an unknown keyword such
as `temperature=` in SDK 1.x, or a missing required one) raises the same TypeError offline that
the real client raises. When the SDK is upgraded, the fake follows automatically.

The response itself is supplied by the test (`respond`), since no network call is made.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from anthropic.resources.messages import Messages

CREATE_SIGNATURE = inspect.signature(Messages.create)
COUNT_TOKENS_SIGNATURE = inspect.signature(Messages.count_tokens)


def check_call(signature: inspect.Signature, kwargs: dict[str, Any]) -> None:
    """Raise TypeError if the real SDK method would reject these keyword arguments."""
    signature.bind(object(), **kwargs)  # object() stands in for `self`


class FakeResponse:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def to_dict(self) -> dict[str, Any]:
        return self._data


class FakeMessages:
    def __init__(
        self,
        respond: Callable[[dict[str, Any]], dict[str, Any]],
        count: Callable[[dict[str, Any]], int] | None = None,
    ) -> None:
        self.respond = respond
        self.count = count or (lambda kwargs: 0)
        self.calls: list[dict[str, Any]] = []
        self.count_calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeResponse:
        check_call(CREATE_SIGNATURE, kwargs)
        self.calls.append(kwargs)
        return FakeResponse(self.respond(kwargs))

    def count_tokens(self, **kwargs: Any) -> Any:
        check_call(COUNT_TOKENS_SIGNATURE, kwargs)
        self.count_calls.append(kwargs)
        return SimpleNamespace(input_tokens=self.count(kwargs))


class FakeAnthropic:
    """Stands in for anthropic.Anthropic(): `client.messages.create` / `.count_tokens`."""

    def __init__(
        self,
        respond: Callable[[dict[str, Any]], dict[str, Any]],
        count: Callable[[dict[str, Any]], int] | None = None,
    ) -> None:
        self.messages = FakeMessages(respond, count)
