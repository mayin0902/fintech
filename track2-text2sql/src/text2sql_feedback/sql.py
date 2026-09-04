from __future__ import annotations

import re
from dataclasses import dataclass

_FENCED_SQL = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9$]*")

# This is a fast application guard, not the security boundary. The executor also
# uses a read-only transaction, timeout, and should connect through a least-privilege role.
_DENIED_TOKENS = {
    "alter",
    "analyze",
    "call",
    "comment",
    "copy",
    "create",
    "delete",
    "do",
    "drop",
    "execute",
    "grant",
    "insert",
    "lock",
    "merge",
    "prepare",
    "reassign",
    "refresh",
    "reindex",
    "reset",
    "revoke",
    "security",
    "set",
    "truncate",
    "update",
    "vacuum",
}
_DENIED_FUNCTIONS = {
    "dblink",
    "lo_export",
    "lo_import",
    "pg_ls_dir",
    "pg_read_binary_file",
    "pg_read_file",
    "pg_sleep",
}


@dataclass(frozen=True)
class SQLValidation:
    ok: bool
    sql: str
    reason: str | None = None


def extract_sql(text: str | None) -> str:
    """Extract the final fenced SQL block, or return stripped raw text."""

    if not text:
        return ""
    blocks = _FENCED_SQL.findall(text)
    return (blocks[-1] if blocks else text).strip()


def _mask_literals_and_comments(sql: str) -> str:
    """Replace quoted/comment content while preserving statement punctuation."""

    output: list[str] = []
    index = 0
    state = "code"
    dollar_tag = ""
    while index < len(sql):
        char = sql[index]
        nxt = sql[index + 1] if index + 1 < len(sql) else ""

        if state == "code":
            if char == "'":
                state = "single"
                output.append(" ")
            elif char == '"':
                state = "double"
                output.append(" ")
            elif char == "-" and nxt == "-":
                state = "line_comment"
                output.extend((" ", " "))
                index += 1
            elif char == "/" and nxt == "*":
                state = "block_comment"
                output.extend((" ", " "))
                index += 1
            elif char == "$":
                tag_match = re.match(r"\$[A-Za-z_0-9]*\$", sql[index:])
                if tag_match:
                    dollar_tag = tag_match.group(0)
                    state = "dollar"
                    output.extend(" " * len(dollar_tag))
                    index += len(dollar_tag) - 1
                else:
                    output.append(char)
            else:
                output.append(char)
        elif state == "single":
            output.append(" ")
            if char == "'" and nxt == "'":
                output.append(" ")
                index += 1
            elif char == "'":
                state = "code"
        elif state == "double":
            output.append(" ")
            if char == '"' and nxt == '"':
                output.append(" ")
                index += 1
            elif char == '"':
                state = "code"
        elif state == "line_comment":
            output.append("\n" if char == "\n" else " ")
            if char == "\n":
                state = "code"
        elif state == "block_comment":
            output.append(" ")
            if char == "*" and nxt == "/":
                output.append(" ")
                index += 1
                state = "code"
        elif state == "dollar":
            if sql.startswith(dollar_tag, index):
                output.extend(" " * len(dollar_tag))
                index += len(dollar_tag) - 1
                state = "code"
            else:
                output.append(" ")
        index += 1

    return "".join(output)


def validate_readonly(text: str | None) -> SQLValidation:
    """Accept exactly one SELECT/WITH statement and reject obvious side effects."""

    sql = extract_sql(text).strip()
    if not sql:
        return SQLValidation(False, "", "empty_sql")

    masked = _mask_literals_and_comments(sql)
    statements = [part.strip() for part in masked.split(";") if part.strip()]
    if len(statements) != 1:
        return SQLValidation(False, sql, "multiple_statements")

    tokens = [token.lower() for token in _WORD.findall(masked)]
    if not tokens or tokens[0] not in {"select", "with"}:
        return SQLValidation(False, sql, "not_select_or_with")

    denied = sorted(set(tokens).intersection(_DENIED_TOKENS))
    if denied:
        return SQLValidation(False, sql, f"denied_keyword:{denied[0]}")

    lowered = masked.lower()
    for function in sorted(_DENIED_FUNCTIONS):
        if re.search(rf"\b{re.escape(function)}\s*\(", lowered):
            return SQLValidation(False, sql, f"denied_function:{function}")

    return SQLValidation(True, sql.rstrip(";") + ";")
