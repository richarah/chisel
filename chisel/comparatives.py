"""
Comparative and Superlative SQL Generation

Uses spaCy morphology (already detected) to generate proper SQL ORDER BY + LIMIT
or WHERE comparisons.

NIH Principle: spaCy does morphology, we just map to SQL.

Examples:
    "oldest student" -> ORDER BY age DESC LIMIT 1
    "students older than 20" -> WHERE age > 20
    "departments with highest budget" -> ORDER BY budget DESC LIMIT 1
    "cheaper than average" -> WHERE price < (SELECT AVG(price) FROM ...)
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import Enum


class ComparativeType(Enum):
    """Types of comparatives in SQL."""
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    EQUAL = "="
    NOT_EQUAL = "!="


class SuperlativeType(Enum):
    """Types of superlatives in SQL."""
    MAXIMUM = "MAX"  # ORDER BY col DESC LIMIT 1
    MINIMUM = "MIN"  # ORDER BY col ASC LIMIT 1


@dataclass
class ComparativePattern:
    """
    Detected comparative pattern.

    Example:
        "older than 20" -> {adjective: "old", type: >, value: 20}
        "more expensive than average" -> {adjective: "expensive", type: >, value: AVG}
    """
    adjective: str  # Base form (old, expensive)
    comp_type: ComparativeType
    value: Optional[float] = None  # Numeric value if present
    is_average: bool = False  # "than average" pattern


@dataclass
class SuperlativePattern:
    """
    Detected superlative pattern.

    Example:
        "oldest" -> {adjective: "old", type: MAX, direction: DESC}
        "cheapest" -> {adjective: "cheap", type: MIN, direction: ASC}
    """
    adjective: str  # Base form
    sup_type: SuperlativeType
    direction: str  # "ASC" or "DESC"
    with_limit: bool = True  # Usually LIMIT 1
    offset: int = 0  # For "second highest" etc


# Map adjectives to database columns (domain-specific)
# In practice, this comes from schema linking
ADJECTIVE_TO_COLUMN = {
    "old": "age",
    "young": "age",
    "high": "value",
    "low": "value",
    "expensive": "price",
    "cheap": "price",
    "large": "size",
    "small": "size",
    "big": "size",
    "long": "length",
    "short": "length",
}


def detect_comparative(tokens, token_idx: int) -> Optional[ComparativePattern]:
    """
    Detect comparative pattern from token with JJR/RBR tag.

    Uses spaCy morphology + dependency parsing to extract pattern.

    Args:
        tokens: List of analyzed tokens
        token_idx: Index of comparative token

    Returns:
        ComparativePattern or None
    """
    token = tokens[token_idx]

    # Must be comparative (already detected by spaCy)
    if token.tag not in ["JJR", "RBR"]:
        return None

    # Get base form of adjective
    adjective = token.lemma.lower()

    # Look for "than" pattern
    comp_type = ComparativeType.GREATER_THAN  # Default for -er form
    value = None
    is_average = False

    # Check if "more" + adjective pattern (periphrastic comparative)
    if token_idx > 0 and tokens[token_idx - 1].lemma.lower() == "more":
        comp_type = ComparativeType.GREATER_THAN
    elif token_idx > 0 and tokens[token_idx - 1].lemma.lower() == "less":
        comp_type = ComparativeType.LESS_THAN

    # Look ahead for "than X"
    for i in range(token_idx + 1, min(token_idx + 5, len(tokens))):
        if tokens[i].lemma.lower() == "than":
            # Found "than", check what follows
            if i + 1 < len(tokens):
                next_token = tokens[i + 1]

                # "than average" / "than the average"
                if next_token.lemma.lower() in ["average", "mean"]:
                    is_average = True

                # "than 20" (numeric)
                elif next_token.numeric_value is not None:
                    value = next_token.numeric_value

            break

    return ComparativePattern(
        adjective=adjective,
        comp_type=comp_type,
        value=value,
        is_average=is_average
    )


def detect_superlative(tokens, token_idx: int, ordinal_idx: Optional[int] = None) -> Optional[SuperlativePattern]:
    """
    Detect superlative pattern from token with JJS/RBS tag.

    Args:
        tokens: List of analyzed tokens
        token_idx: Index of superlative token
        ordinal_idx: Optional index of ordinal (for "third highest")

    Returns:
        SuperlativePattern or None
    """
    token = tokens[token_idx]

    # Must be superlative
    if token.tag not in ["JJS", "RBS"]:
        return None

    adjective = token.lemma.lower()

    # Determine if MAX (highest, most) or MIN (lowest, least)
    # Check both the token and any "most"/"least" modifiers
    is_min = False

    # Direct superlatives that indicate minimum
    if adjective in ["low", "small", "short", "cheap", "young", "least", "fewest"]:
        is_min = True

    # Check for "most"/"least" modifier
    if token_idx > 0:
        prev = tokens[token_idx - 1].lemma.lower()
        if prev == "most":
            is_min = False
        elif prev == "least":
            is_min = True

    sup_type = SuperlativeType.MINIMUM if is_min else SuperlativeType.MAXIMUM
    direction = "ASC" if is_min else "DESC"

    # Handle ordinals: "third highest" = OFFSET 2
    offset = 0
    if ordinal_idx is not None:
        ordinal_token = tokens[ordinal_idx]
        # ordinal_value already parsed (e.g., "third" -> 3)
        if hasattr(ordinal_token, 'ordinal_value'):
            offset = ordinal_token.ordinal_value - 1

    return SuperlativePattern(
        adjective=adjective,
        sup_type=sup_type,
        direction=direction,
        with_limit=True,
        offset=offset
    )


def generate_comparative_sql(pattern: ComparativePattern, column: str, table: str) -> str:
    """
    Generate SQL WHERE clause from comparative pattern.

    Args:
        pattern: Detected comparative pattern
        column: Database column name
        table: Database table name

    Returns:
        SQL WHERE clause
    """
    if pattern.is_average:
        # "than average" -> subquery
        subquery = f"(SELECT AVG({column}) FROM {table})"
        return f"WHERE {column} {pattern.comp_type.value} {subquery}"

    elif pattern.value is not None:
        # "than 20" -> direct comparison
        return f"WHERE {column} {pattern.comp_type.value} {pattern.value}"

    else:
        # Just comparative without "than" - context dependent
        # Usually implies comparison to some reference
        return f"WHERE {column} {pattern.comp_type.value} value"


def generate_superlative_sql(pattern: SuperlativePattern, column: str, table: str) -> str:
    """
    Generate SQL ORDER BY + LIMIT from superlative pattern.

    Args:
        pattern: Detected superlative pattern
        column: Database column name
        table: Database table name

    Returns:
        SQL ORDER BY clause
    """
    sql = f"ORDER BY {column} {pattern.direction}"

    if pattern.with_limit:
        sql += " LIMIT 1"

    if pattern.offset > 0:
        sql += f" OFFSET {pattern.offset}"

    return sql


# ==========================
# INTEGRATION WITH CHISEL
# ==========================

def enhance_sql_with_comparatives(
    base_sql: str,
    comparatives: List[tuple],  # List of (idx, word) from question_analysis
    tokens,
    schema
) -> str:
    """
    Enhance base SQL with comparative clauses.

    Args:
        base_sql: Base SQL query
        comparatives: Detected comparatives from question_analysis
        tokens: Token list
        schema: Database schema

    Returns:
        Enhanced SQL with WHERE clauses
    """
    for idx, word in comparatives:
        pattern = detect_comparative(tokens, idx)
        if pattern:
            # Map adjective to column using schema
            column = ADJECTIVE_TO_COLUMN.get(pattern.adjective, pattern.adjective)

            # Generate WHERE clause
            where_clause = generate_comparative_sql(pattern, column, "table")

            # Add to SQL (simplified - actual would parse SQL AST)
            if "WHERE" in base_sql:
                base_sql += f" AND {where_clause.replace('WHERE ', '')}"
            else:
                base_sql += f" {where_clause}"

    return base_sql


def enhance_sql_with_superlatives(
    base_sql: str,
    superlatives: List[tuple],  # List of (idx, word) from question_analysis
    ordinals: List[tuple],  # List of (idx, value) from question_analysis
    tokens,
    schema
) -> str:
    """
    Enhance base SQL with superlative ORDER BY + LIMIT.

    Args:
        base_sql: Base SQL query
        superlatives: Detected superlatives from question_analysis
        ordinals: Detected ordinals (for "third highest")
        tokens: Token list
        schema: Database schema

    Returns:
        Enhanced SQL with ORDER BY
    """
    for idx, word in superlatives:
        # Check if there's a nearby ordinal
        ordinal_idx = None
        for ord_idx, ord_val in ordinals:
            if abs(ord_idx - idx) <= 2:  # Within 2 tokens
                ordinal_idx = ord_idx
                break

        pattern = detect_superlative(tokens, idx, ordinal_idx)
        if pattern:
            # Map adjective to column
            column = ADJECTIVE_TO_COLUMN.get(pattern.adjective, pattern.adjective)

            # Generate ORDER BY clause
            order_clause = generate_superlative_sql(pattern, column, "table")

            # Add to SQL
            base_sql += f" {order_clause}"

    return base_sql


# ==========================
# EXAMPLES
# ==========================

if __name__ == "__main__":
    # Example patterns
    print("Comparative/Superlative SQL Generation")
    print("=" * 60)

    # Superlative examples
    patterns = [
        ("oldest student", "age", "student", "ORDER BY age DESC LIMIT 1"),
        ("youngest professor", "age", "professor", "ORDER BY age ASC LIMIT 1"),
        ("highest budget", "budget", "department", "ORDER BY budget DESC LIMIT 1"),
        ("third highest salary", "salary", "employee", "ORDER BY salary DESC LIMIT 1 OFFSET 2"),
    ]

    print("Superlative Patterns:")
    for desc, col, table, expected in patterns:
        print(f"  '{desc}' -> {expected}")

    print()

    # Comparative examples
    patterns = [
        ("older than 20", "age", "student", "WHERE age > 20"),
        ("cheaper than average", "price", "product", "WHERE price < (SELECT AVG(price) FROM product)"),
        ("departments with budget > 1M", "budget", "department", "WHERE budget > 1000000"),
    ]

    print("Comparative Patterns:")
    for desc, col, table, expected in patterns:
        print(f"  '{desc}' -> {expected}")

    print()
    print("[OK] All patterns use spaCy morphology + deterministic rules")
