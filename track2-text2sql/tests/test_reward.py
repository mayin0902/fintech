from dataclasses import dataclass

from text2sql_feedback import ExecutionReward, ExpectedResult, QueryResult
from text2sql_feedback.contracts import ExecutionOutcome


@dataclass
class StaticExecutor:
    outcome: ExecutionOutcome

    def execute(self, sql: str) -> ExecutionOutcome:
        return self.outcome


def test_exact_execution_match_is_one() -> None:
    result = QueryResult.from_sequences(["count"], [[3]])
    scorer = ExecutionReward(StaticExecutor(ExecutionOutcome(True, result=result)))
    reward = scorer.score(
        "```sql\nSELECT COUNT(*) AS count FROM loans;\n```", ExpectedResult(result)
    )
    assert reward.total == 1.0
    assert reward.exact_match


def test_executable_mismatch_gets_shaping_but_is_capped() -> None:
    actual = QueryResult.from_sequences(["count"], [[2]])
    expected = ExpectedResult(QueryResult.from_sequences(["count"], [[3]]))
    scorer = ExecutionReward(StaticExecutor(ExecutionOutcome(True, result=actual)))
    reward = scorer.score("```sql\nSELECT COUNT(*) AS count FROM loans;\n```", expected)
    assert 0.0 < reward.total <= scorer.non_exact_cap
    assert not reward.exact_match
    assert reward.feedback == "value_mismatch"


def test_unsafe_sql_never_reaches_executor() -> None:
    class ExplodingExecutor:
        def execute(self, sql: str) -> ExecutionOutcome:
            raise AssertionError("unsafe SQL must not reach the database")

    scorer = ExecutionReward(ExplodingExecutor())
    expected = ExpectedResult(QueryResult.from_sequences([], []))
    reward = scorer.score("DROP TABLE loans", expected)
    assert reward.total == 0.0
    assert reward.feedback == "not_select_or_with"
