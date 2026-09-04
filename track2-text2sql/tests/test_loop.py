from collections.abc import Sequence
from dataclasses import dataclass, field

from text2sql_feedback import ExecutionReward, ExpectedResult, QueryResult
from text2sql_feedback.contracts import ExecutionOutcome
from text2sql_feedback.loop import ReflectionLoop


@dataclass
class SequenceGenerator:
    outputs: list[str]
    observed_messages: list[list[dict[str, str]]] = field(default_factory=list)

    def generate(self, messages: Sequence[dict[str, str]]) -> str:
        self.observed_messages.append(list(messages))
        return self.outputs[len(self.observed_messages) - 1]


class MappingExecutor:
    def execute(self, sql: str) -> ExecutionOutcome:
        value = 3 if "approved" in sql else 5
        return ExecutionOutcome(True, QueryResult.from_sequences(["count"], [[value]]))


def test_reflection_loop_uses_feedback_and_recovers() -> None:
    generator = SequenceGenerator(
        [
            "```sql\nSELECT COUNT(*) AS count FROM loans;\n```",
            "```sql\nSELECT COUNT(*) AS count FROM loans WHERE status = 'approved';\n```",
        ]
    )
    expected = ExpectedResult(QueryResult.from_sequences(["count"], [[3]]))
    loop = ReflectionLoop(generator, ExecutionReward(MappingExecutor()), max_attempts=2)
    result = loop.solve(
        question="How many approved loans?", schema="loans(status text)", expected=expected
    )
    assert result.solved
    assert len(result.attempts) == 2
    assert "feedback=value_mismatch" in generator.observed_messages[1][-1]["content"]
