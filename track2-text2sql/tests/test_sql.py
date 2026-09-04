from text2sql_feedback.sql import extract_sql, validate_readonly


def test_extracts_last_fenced_block() -> None:
    text = "draft ```sql\nSELECT 0;\n``` final ```sql\nSELECT 1;\n```"
    assert extract_sql(text) == "SELECT 1;"


def test_accepts_one_readonly_statement() -> None:
    result = validate_readonly("SELECT * FROM loans WHERE status = 'approved';")
    assert result.ok
    assert result.sql.endswith(";")


def test_semicolon_and_keyword_inside_literal_are_not_an_injection() -> None:
    result = validate_readonly("SELECT 'x; DROP TABLE loans' AS example;")
    assert result.ok


def test_rejects_multiple_statements() -> None:
    result = validate_readonly("SELECT 1; DROP TABLE loans;")
    assert not result.ok
    assert result.reason == "multiple_statements"


def test_rejects_data_modifying_cte() -> None:
    result = validate_readonly(
        "WITH changed AS (DELETE FROM loans RETURNING *) SELECT * FROM changed"
    )
    assert not result.ok
    assert result.reason == "denied_keyword:delete"


def test_rejects_side_effect_function() -> None:
    result = validate_readonly("SELECT pg_sleep(10)")
    assert not result.ok
    assert result.reason == "denied_function:pg_sleep"
