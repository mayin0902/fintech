from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .contracts import ExpectedResult, QueryResult, SQLExecutor
from .reward import ExecutionReward


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        final = completion[-1]
        if isinstance(final, dict):
            return str(final.get("content", ""))
    return str(completion)


@dataclass
class TRLExecutionReward:
    """Pickle-friendly TRL reward callable backed by the same DB verifier."""

    executor: SQLExecutor
    max_workers: int = 8

    def __call__(
        self,
        completions: list[Any],
        expected_columns: list[list[str]],
        expected_rows: list[list[list[Any]]],
        ordered: list[bool] | None = None,
        **_: Any,
    ) -> list[float]:
        order_flags = ordered or [False] * len(completions)
        jobs = list(zip(completions, expected_columns, expected_rows, order_flags, strict=True))

        def score_one(job: tuple[Any, list[str], list[list[Any]], bool]) -> float:
            completion, columns, rows, is_ordered = job
            expected = ExpectedResult(
                QueryResult.from_sequences(columns, rows),
                ordered=bool(is_ordered),
            )
            return (
                ExecutionReward(self.executor).score(_completion_text(completion), expected).total
            )

        # DB reward is I/O-bound. Bound concurrency to protect PostgreSQL while
        # avoiding a serial reward bottleneck across each GRPO completion group.
        workers = max(1, min(self.max_workers, len(jobs)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(score_one, jobs))
