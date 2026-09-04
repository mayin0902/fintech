from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class QueryResult:
    """A database result with stable, serializable structure."""

    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    latency_ms: float = 0.0

    @classmethod
    def from_sequences(
        cls,
        columns: Sequence[str],
        rows: Sequence[Sequence[Any]],
        *,
        latency_ms: float = 0.0,
    ) -> QueryResult:
        return cls(tuple(columns), tuple(tuple(row) for row in rows), latency_ms)


@dataclass(frozen=True)
class ExpectedResult:
    """Gold execution result; row order matters only when explicitly requested."""

    result: QueryResult
    ordered: bool = False


@dataclass(frozen=True)
class ExecutionOutcome:
    """Result of the guarded database execution boundary."""

    ok: bool
    result: QueryResult | None = None
    error_kind: str | None = None
    error_message: str | None = None


class SQLExecutor(Protocol):
    def execute(self, sql: str) -> ExecutionOutcome: ...


class SQLGenerator(Protocol):
    def generate(self, messages: Sequence[dict[str, str]]) -> str: ...
