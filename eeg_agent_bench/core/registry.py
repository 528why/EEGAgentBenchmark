"""Generic registry utilities."""

from __future__ import annotations

from typing import Any, TypeVar, Generic

T = TypeVar("T")


class Registry(Generic[T]):
    """A simple name -> object registry."""

    def __init__(self, name: str = "default"):
        self._name = name
        self._items: dict[str, T] = {}

    def register(self, name: str, item: T) -> None:
        if name in self._items:
            raise ValueError(f"[{self._name}] '{name}' is already registered.")
        self._items[name] = item

    def get(self, name: str) -> T:
        if name not in self._items:
            raise KeyError(f"[{self._name}] '{name}' is not registered. Available: {list(self._items.keys())}")
        return self._items[name]

    def list_names(self) -> list[str]:
        return list(self._items.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)
