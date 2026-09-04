from __future__ import annotations

from dataclasses import dataclass

from .contracts import ExpectedResult, SQLExecutor
from .evaluation import Comparison, compare_results
from .sql import extract_sql, validate_readonly


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    format_score: float
    readonly_score: float
    executable_score: float
    column_score: float
    row_count_score: float
    row_overlap_score: float
    exact_match: bool
    feedback: str
    comparison: Comparison | None = None


@dataclass(frozen=True)
class ExecutionReward:
    """Dense shaping anchored by exact PostgreSQL execution equivalence."""

    executor: SQLExecutor
    non_exact_cap: float = 0.85

    def score(self, completion: str, expected: ExpectedResult) -> RewardBreakdown:
        fenced = "```sql" in completion.casefold() and completion.rstrip().endswith("```")
        format_score = 0.05 if fenced else 0.0
        validation = validate_readonly(completion)
        if not validation.ok:
            return RewardBreakdown(
                total=format_score,
                format_score=format_score,
                readonly_score=0.0,
                executable_score=0.0,
                column_score=0.0,
                row_count_score=0.0,
                row_overlap_score=0.0,
                exact_match=False,
                feedback=validation.reason or "invalid_sql",
            )

        outcome = self.executor.execute(extract_sql(completion))
        if not outcome.ok or outcome.result is None:
            total = format_score + 0.10
            return RewardBreakdown(
                total=total,
                format_score=format_score,
                readonly_score=0.10,
                executable_score=0.0,
                column_score=0.0,
                row_count_score=0.0,
                row_overlap_score=0.0,
                exact_match=False,
                feedback=f"{outcome.error_kind}:{outcome.error_message}",
            )

        comparison = compare_results(expected, outcome.result)
        if comparison.exact_match:
            return RewardBreakdown(
                total=1.0,
                format_score=format_score,
                readonly_score=0.10,
                executable_score=0.15,
                column_score=0.15,
                row_count_score=0.15,
                row_overlap_score=0.40,
                exact_match=True,
                feedback="exact_match",
                comparison=comparison,
            )

        column_score = 0.15 if comparison.column_match else 0.0
        row_count_score = 0.15 if comparison.row_count_match else 0.0
        row_overlap_score = 0.40 * comparison.row_overlap
        total = min(
            format_score + 0.10 + 0.15 + column_score + row_count_score + row_overlap_score,
            self.non_exact_cap,
        )
        return RewardBreakdown(
            total=round(total, 6),
            format_score=format_score,
            readonly_score=0.10,
            executable_score=0.15,
            column_score=column_score,
            row_count_score=row_count_score,
            row_overlap_score=round(row_overlap_score, 6),
            exact_match=False,
            feedback=comparison.reason,
            comparison=comparison,
        )
