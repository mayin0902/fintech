from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SchemaSnapshot:
    """Compact schema representation used for schema grounding."""

    text: str


def inspect_postgres_schema(connection: Any, schema_name: str) -> SchemaSnapshot:
    """Read tables, columns, primary keys, and foreign keys from PostgreSQL catalogs."""

    columns = connection.execute(
        """
        SELECT table_name, column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = %s
        ORDER BY table_name, ordinal_position
        """,
        (schema_name,),
    ).fetchall()
    foreign_keys = connection.execute(
        """
        SELECT tc.table_name, kcu.column_name,
               ccu.table_name AS foreign_table_name,
               ccu.column_name AS foreign_column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.constraint_schema = kcu.constraint_schema
        JOIN information_schema.constraint_column_usage ccu
          ON ccu.constraint_name = tc.constraint_name
         AND ccu.constraint_schema = tc.constraint_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_schema = %s
        ORDER BY tc.table_name, kcu.column_name
        """,
        (schema_name,),
    ).fetchall()

    table_lines: dict[str, list[str]] = {}
    for table, column, data_type, nullable in columns:
        suffix = "" if nullable == "YES" else " NOT NULL"
        table_lines.setdefault(table, []).append(f"  {column} {data_type}{suffix}")

    parts: list[str] = []
    for table, definitions in table_lines.items():
        parts.append(f"{schema_name}.{table}(\n" + ",\n".join(definitions) + "\n)")
    if foreign_keys:
        parts.append("FOREIGN KEYS:")
        parts.extend(
            f"  {schema_name}.{table}.{column} -> {schema_name}.{foreign_table}.{foreign_column}"
            for table, column, foreign_table, foreign_column in foreign_keys
        )
    return SchemaSnapshot("\n\n".join(parts))
