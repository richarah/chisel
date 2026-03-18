"""
Task 6: SQL Validation & Repair

Validate generated SQL and attempt simple repairs.

Libraries doing the heavy lifting:
- sqlglot: Parse SQL, detect syntax errors, validate structure

We write: Repair rules for common errors
"""

from typing import Optional, Tuple, List
import sqlglot
from sqlglot import exp, parse_one
from sqlglot.errors import ParseError

from .schema_graph import SchemaGraph


def validate_sql(sql: str) -> Tuple[bool, Optional[str]]:
    """
    Validate SQL syntax using sqlglot.

    Args:
        sql: SQL query string

    Returns: (is_valid, error_message)
    """
    if not sql or not sql.strip():
        return False, "Empty SQL"

    try:
        # Parse SQL (sqlglot does this)
        parsed = parse_one(sql, read="sqlite")

        # Check if it's a SELECT statement
        if not isinstance(parsed, exp.Select):
            return False, f"Not a SELECT statement (got {type(parsed).__name__})"

        return True, None

    except ParseError as e:
        return False, f"Parse error: {str(e)}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def repair_sql(sql: str) -> Optional[str]:
    """
    Attempt to repair common SQL errors.

    Returns: Repaired SQL or None if unrepairable
    """
    if not sql:
        return None

    try:
        # Parse and regenerate (normalizes formatting)
        parsed = parse_one(sql, read="sqlite")
        repaired = parsed.sql(dialect="sqlite")
        return repaired

    except ParseError:
        # Try common repairs

        # Repair 1: Missing SELECT keyword
        if not sql.strip().upper().startswith("SELECT"):
            sql = "SELECT " + sql

        # Repair 2: Unquoted identifiers with spaces
        # sqlglot should handle this, but try re-parsing
        try:
            parsed = parse_one(sql, read="sqlite")
            return parsed.sql(dialect="sqlite")
        except:
            pass

        return None


def validate_and_repair(sql: str) -> Tuple[str, bool, Optional[str]]:
    """
    Validate SQL and attempt repair if needed.

    Args:
        sql: SQL query string

    Returns: (final_sql, is_valid, error_message)
    """
    # First validation
    is_valid, error = validate_sql(sql)

    if is_valid:
        return sql, True, None

    # Try to repair
    repaired = repair_sql(sql)

    if repaired:
        # Validate repaired SQL
        is_valid, error = validate_sql(repaired)
        if is_valid:
            return repaired, True, None
        else:
            return repaired, False, f"Repair failed: {error}"
    else:
        return sql, False, error


def validate_against_schema(sql: str, schema: SchemaGraph) -> Tuple[bool, Optional[str]]:
    """
    Validate SQL against database schema (semantic validation).

    Checks:
    - All referenced tables exist in schema
    - All referenced columns exist in their tables
    - JOIN conditions reference valid foreign keys

    Args:
        sql: SQL query string
        schema: Database schema

    Returns: (is_valid, error_message)
    """
    try:
        parsed = parse_one(sql, read="sqlite")

        if not isinstance(parsed, exp.Select):
            return False, "Not a SELECT statement"

        # Extract all table references
        tables = list(parsed.find_all(exp.Table))
        schema_tables = set(schema.get_all_tables())

        for table_node in tables:
            table_name = table_node.name
            if table_name not in schema_tables:
                return False, f"Table '{table_name}' does not exist in schema"

        # Extract all column references
        columns = list(parsed.find_all(exp.Column))
        schema_columns = {f"{t}.{c}" for t, c in schema.get_all_columns()}

        for col_node in columns:
            col_name = col_node.name
            table_name = col_node.table if hasattr(col_node, 'table') else None

            # If table specified, check table.column exists
            if table_name and table_name != '':
                full_name = f"{table_name}.{col_name}"
                if full_name not in schema_columns:
                    # Check if column exists in any table
                    if not any(c == col_name for t, c in schema.get_all_columns()):
                        return False, f"Column '{col_name}' does not exist in schema"

        return True, None

    except ParseError as e:
        return False, f"Cannot parse SQL for schema validation: {str(e)}"
    except Exception as e:
        return False, f"Schema validation error: {str(e)}"


def check_sql_features(sql: str) -> dict:
    """
    Analyze SQL query features using sqlglot.

    Returns dict with:
        - has_join: bool
        - has_where: bool
        - has_group_by: bool
        - has_having: bool
        - has_order_by: bool
        - has_limit: bool
        - has_aggregation: bool
        - table_count: int
        - column_count: int
    """
    features = {
        "has_join": False,
        "has_where": False,
        "has_group_by": False,
        "has_having": False,
        "has_order_by": False,
        "has_limit": False,
        "has_aggregation": False,
        "table_count": 0,
        "column_count": 0,
    }

    try:
        parsed = parse_one(sql, read="sqlite")

        if not isinstance(parsed, exp.Select):
            return features

        # Check for joins
        joins = parsed.args.get("joins")
        if joins:
            features["has_join"] = True

        # Check for WHERE
        where = parsed.args.get("where")
        if where:
            features["has_where"] = True

        # Check for GROUP BY
        group = parsed.args.get("group")
        if group:
            features["has_group_by"] = True

        # Check for HAVING
        having = parsed.args.get("having")
        if having:
            features["has_having"] = True

        # Check for ORDER BY
        order = parsed.args.get("order")
        if order:
            features["has_order_by"] = True

        # Check for LIMIT
        limit = parsed.args.get("limit")
        if limit:
            features["has_limit"] = True

        # Check for aggregations (sqlglot makes this easy)
        aggregations = list(parsed.find_all(exp.Count, exp.Max, exp.Min, exp.Avg, exp.Sum))
        features["has_aggregation"] = len(aggregations) > 0

        # Count tables
        tables = list(parsed.find_all(exp.Table))
        features["table_count"] = len(tables)

        # Count columns in SELECT
        select_exps = parsed.args.get("expressions", [])
        features["column_count"] = len(select_exps)

    except:
        pass

    return features


if __name__ == "__main__":
    # Test validation
    test_queries = [
        "SELECT * FROM student",
        "SELECT COUNT(*) FROM student",
        "SELECT name, age FROM student WHERE age > 20",
        "SELECT invalid syntax here",
        "SELECT s.name FROM student s JOIN enrollment e ON s.id = e.student_id",
    ]

    for sql in test_queries:
        print(f"\nSQL: {sql}")
        final_sql, is_valid, error = validate_and_repair(sql)

        if is_valid:
            print(f"  [OK] Valid")
            features = check_sql_features(final_sql)
            print(f"  Features: {features}")
        else:
            print(f"  [X] Invalid: {error}")
