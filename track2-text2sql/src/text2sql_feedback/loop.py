from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .contracts import ExpectedResult, SQLGenerator
from .reward import ExecutionReward, RewardBreakdown
from .sql import extract_sql

SYSTEM_PROMPT = """You are a PostgreSQL Text-to-SQL agent.
Use only tables and columns from the supplied schema. Return one read-only SELECT
or WITH query in a lowercase ```sql fenced block, with no prose."""


@dataclass(frozen=True)
class Attempt:
    index: int
    completion: str
    sql: str
    reward: RewardBreakdown


@dataclass(frozen=True)
class LoopResult:
    solved: bool
    final_sql: str
    attempts: tuple[Attempt, ...]


@dataclass
class ReflectionLoop:
    """Generate, execute, reflect on structured feedback, and retry."""

    generator: SQLGenerator
    reward: ExecutionReward
    max_attempts: int = 3

    def solve(
        self,
        *,
        question: str,
        schema: str,
        expected: ExpectedResult,
        history: Sequence[str] = (),
    ) -> LoopResult:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self._task_prompt(question=question, schema=schema, history=history),
            },
        ]
        attempts: list[Attempt] = []
        for index in range(1, self.max_attempts + 1):
            completion = self.generator.generate(messages)
            reward = self.reward.score(completion, expected)
            sql = extract_sql(completion)
            attempts.append(Attempt(index, completion, sql, reward))
            if reward.exact_match:
                return LoopResult(True, sql, tuple(attempts))

            messages.extend(
                [
                    {"role": "assistant", "content": completion},
                    {"role": "user", "content": self._repair_prompt(reward)},
                ]
            )
        return LoopResult(False, attempts[-1].sql if attempts else "", tuple(attempts))

    @staticmethod
    def _task_prompt(*, question: str, schema: str, history: Sequence[str]) -> str:
        sections = [f"Database schema:\n{schema}"]
        if history:
            sections.append("Conversation history:\n" + "\n".join(history))
        sections.append(f"Current question:\n{question}")
        return "\n\n".join(sections)

    @staticmethod
    def _repair_prompt(reward: RewardBreakdown) -> str:
        detail = reward.feedback.replace("\n", " ")[:500]
        return (
            "The database verifier rejected the previous SQL. "
            f"feedback={detail}; reward={reward.total:.3f}. "
            "Diagnose the schema, join, filter, aggregation, or projection mistake and return "
            "a corrected SQL query only. Do not guess or reveal expected result rows."
        )
