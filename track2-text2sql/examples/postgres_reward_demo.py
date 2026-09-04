from __future__ import annotations

import os

from text2sql_feedback import ExecutionReward, ExpectedResult, QueryResult
from text2sql_feedback.executor import PostgresExecutor


def main() -> None:
    executor = PostgresExecutor(
        os.getenv(
            "TEXT2SQL_DSN",
            "postgresql://text2sql:text2sql@localhost:5432/text2sql",
        )
    )
    expected = ExpectedResult(
        QueryResult.from_sequences(
            ["city", "approved_amount"],
            [["Shanghai", "48000.00"]],
        )
    )
    completion = """```sql
SELECT c.city, SUM(l.amount) AS approved_amount
FROM finance_demo.customers c
JOIN finance_demo.loans l ON l.customer_id = c.customer_id
WHERE l.status = 'approved'
GROUP BY c.city;
```"""
    breakdown = ExecutionReward(executor).score(completion, expected)
    print(breakdown)


if __name__ == "__main__":
    main()
