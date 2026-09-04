from text2sql_feedback import ExpectedResult, QueryResult, compare_results


def test_unordered_comparison_preserves_duplicate_rows() -> None:
    expected = ExpectedResult(QueryResult.from_sequences(["amount"], [[1], [1], [2]]))
    actual = QueryResult.from_sequences(["amount"], [[2], [1], [1]])
    assert compare_results(expected, actual).exact_match


def test_multiset_detects_missing_duplicate() -> None:
    expected = ExpectedResult(QueryResult.from_sequences(["amount"], [[1], [1], [2]]))
    actual = QueryResult.from_sequences(["amount"], [[1], [2], [2]])
    comparison = compare_results(expected, actual)
    assert not comparison.exact_match
    assert comparison.reason == "value_mismatch"
    assert comparison.row_overlap == 0.5


def test_ordered_result_is_sequence_sensitive() -> None:
    expected = ExpectedResult(
        QueryResult.from_sequences(["id"], [[1], [2]]),
        ordered=True,
    )
    actual = QueryResult.from_sequences(["id"], [[2], [1]])
    assert not compare_results(expected, actual).exact_match
