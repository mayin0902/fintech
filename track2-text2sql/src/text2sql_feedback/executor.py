from __future__ import annotations

import time
from dataclasses import dataclass

from .contracts import ExecutionOutcome, QueryResult
from .sql import validate_readonly


@dataclass(frozen=True)
class PostgresExecutor:
    """Execute generated SQL inside a read-only, time-bounded PostgreSQL transaction."""

    dsn: str
    statement_timeout_ms: int = 5_000
    timezone: str = "Asia/Shanghai"

    def execute(self, sql: str) -> ExecutionOutcome:
        validation = validate_readonly(sql)
        if not validation.ok:
            return ExecutionOutcome(
                ok=False,
                error_kind="unsafe_sql",
                error_message=validation.reason,
            )

        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("Install the postgres extra: pip install -e '.[postgres]'") from exc

        started = time.perf_counter()
        try:
            with psycopg.connect(self.dsn) as connection, connection.transaction():
                connection.execute("SET TRANSACTION READ ONLY")
                connection.execute(
                    "SELECT set_config('statement_timeout', %s, true)",
                    (f"{self.statement_timeout_ms}ms",),
                )
                connection.execute(
                    "SELECT set_config('TimeZone', %s, true)",
                    (self.timezone,),
                )
                cursor = connection.execute(validation.sql)
                rows = cursor.fetchall()
                columns = tuple(column.name for column in cursor.description or ())
            latency_ms = (time.perf_counter() - started) * 1_000
            return ExecutionOutcome(
                ok=True,
                result=QueryResult.from_sequences(columns, rows, latency_ms=latency_ms),
            )
        except Exception as exc:  # database drivers expose backend-specific subclasses
            message = str(exc).strip().splitlines()[0][:500]
            lowered = message.lower()
            kind = "timeout" if "statement timeout" in lowered else "execution_error"
            return ExecutionOutcome(ok=False, error_kind=kind, error_message=message)
