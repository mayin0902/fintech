from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from typing import Any

from .contracts import ExpectedResult, QueryResult


@dataclass(frozen=True)
class Comparison:
    exact_match: bool
    column_match: bool
    row_count_match: bool
    row_overlap: float
    reason: str


def normalize_cell(value: Any) -> Any:
    """Normalize driver-specific values without turning numbers into lossy floats."""

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, Decimal):
        return value.normalize()
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return Decimal(str(value)).normalize()
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return Decimal(stripped).normalize()
        except (InvalidOperation, ValueError):
            return stripped
    return str(value)


def normalize_row(row: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(normalize_cell(value) for value in row)


def _multiset_overlap(expected: list[tuple[Any, ...]], actual: list[tuple[Any, ...]]) -> float:
    if not expected and not actual:
        return 1.0
    expected_counts = Counter(expected)
    actual_counts = Counter(actual)
    intersection = sum((expected_counts & actual_counts).values())
    union = sum((expected_counts | actual_counts).values())
    return intersection / union if union else 0.0


def compare_results(expected: ExpectedResult, actual: QueryResult) -> Comparison:
    """Compare typed results as an ordered sequence or unordered row multiset."""

    expected_result = expected.result
    expected_columns = tuple(column.casefold() for column in expected_result.columns)
    actual_columns = tuple(column.casefold() for column in actual.columns)
    column_match = expected_columns == actual_columns
    row_count_match = len(expected_result.rows) == len(actual.rows)

    expected_rows = [normalize_row(row) for row in expected_result.rows]
    actual_rows = [normalize_row(row) for row in actual.rows]
    if expected.ordered:
        rows_equal = expected_rows == actual_rows
        overlap = sum(
            left == right for left, right in zip(expected_rows, actual_rows, strict=False)
        )
        overlap /= max(len(expected_rows), len(actual_rows), 1)
    else:
        rows_equal = Counter(expected_rows) == Counter(actual_rows)
        overlap = _multiset_overlap(expected_rows, actual_rows)

    exact_match = column_match and rows_equal
    if exact_match:
        reason = "exact_match"
    elif not column_match:
        reason = "column_mismatch"
    elif not row_count_match:
        reason = "row_count_mismatch"
    else:
        reason = "value_mismatch"
    return Comparison(exact_match, column_match, row_count_match, overlap, reason)
