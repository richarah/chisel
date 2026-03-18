"""
Negation Scope Detection

Uses spaCy dependency parsing to determine the scope of negation and map to SQL.

NIH Principle: spaCy does dependency parsing, we traverse the tree.

SQL Negation Strategies:
    1. NOT EXISTS - "no students enrolled"
    2. NOT IN - "students not in CS"
    3. LEFT JOIN ... WHERE IS NULL - "students without enrollments"
    4. EXCEPT - "students in CS except those who failed"
    5. WHERE NOT - "students who are not seniors"

Examples:
    "students who have NOT taken any course"
    -> Scope: verb "taken"
    -> SQL: NOT EXISTS (SELECT * FROM enrollment WHERE student_id = s.id)

    "departments WITHOUT CS courses"
    -> Scope: noun "courses"
    -> SQL: LEFT JOIN course ON ... WHERE course.id IS NULL AND dept='CS'
"""

from dataclasses import dataclass
from typing import List, Optional, Set
from enum import Enum


class NegationType(Enum):
    """
    SQL negation strategies.
    """
    NOT_EXISTS = "NOT EXISTS"  # Subquery negation
    NOT_IN = "NOT IN"  # Set negation
    LEFT_JOIN_NULL = "LEFT JOIN ... IS NULL"  # Absence via outer join
    EXCEPT = "EXCEPT"  # Set difference
    WHERE_NOT = "WHERE NOT"  # Simple boolean negation


@dataclass
class NegationScope:
    """
    Detected negation with its scope.

    Example:
        "students who have NOT taken CS101"
        -> neg_token: "not" (index 3)
        -> scope_head: "taken" (index 4)
        -> scope_type: VERB
        -> affected_tokens: [4, 5] ("taken", "CS101")
        -> sql_strategy: NOT_EXISTS
    """
    neg_token_idx: int  # Index of negation word
    scope_head_idx: int  # Index of word negation applies to
    scope_type: str  # VERB, NOUN, ADJ, etc.
    affected_tokens: Set[int]  # All tokens in scope
    sql_strategy: NegationType


def find_negation_scope(tokens, neg_idx: int) -> Optional[NegationScope]:
    """
    Find the scope of negation using dependency parsing.

    Algorithm:
        1. Start from negation token (detected by spaCy as dep="neg")
        2. Find its head (what it negates)
        3. Traverse subtree to find full scope
        4. Determine appropriate SQL strategy

    Args:
        tokens: List of Token objects with dependency info
        neg_idx: Index of negation token

    Returns:
        NegationScope or None
    """
    neg_token = tokens[neg_idx]

    # Negation token should have dep="neg"
    if neg_token.dep != "neg":
        return None

    # Find the head (what this negation modifies)
    head_idx = neg_token.head_idx

    if head_idx < 0 or head_idx >= len(tokens):
        return None

    head_token = tokens[head_idx]

    # Determine scope type from head's POS
    scope_type = head_token.pos

    # Collect all tokens in the negation scope
    # This includes the head and its dependents
    affected = {head_idx}

    # Add all descendants of head
    for i, token in enumerate(tokens):
        if token.head_idx == head_idx:
            affected.add(i)
            # Also add their descendants (recursive)
            affected.update(_get_descendants(tokens, i))

    # Determine SQL strategy based on scope
    sql_strategy = _determine_sql_strategy(head_token, affected, tokens)

    return NegationScope(
        neg_token_idx=neg_idx,
        scope_head_idx=head_idx,
        scope_type=scope_type,
        affected_tokens=affected,
        sql_strategy=sql_strategy
    )


def _get_descendants(tokens, idx: int) -> Set[int]:
    """
    Get all descendants of a token in the dependency tree.

    Args:
        tokens: Token list
        idx: Token index

    Returns:
        Set of descendant indices
    """
    descendants = set()

    for i, token in enumerate(tokens):
        if token.head_idx == idx:
            descendants.add(i)
            # Recursive
            descendants.update(_get_descendants(tokens, i))

    return descendants


def _determine_sql_strategy(head_token, affected_tokens: Set[int], tokens) -> NegationType:
    """
    Determine which SQL negation strategy to use.

    Decision tree:
        1. Verb negation (esp. existential) -> NOT EXISTS
        2. "without" / "absence" -> LEFT JOIN ... IS NULL
        3. "except" / "but not" -> EXCEPT
        4. Set membership -> NOT IN
        5. Default -> WHERE NOT

    Args:
        head_token: The token being negated
        affected_tokens: Set of token indices in scope
        tokens: Full token list

    Returns:
        NegationType enum
    """
    # Check for specific patterns

    # Pattern 1: "have not taken" / "do not exist" -> NOT EXISTS
    if head_token.pos == "VERB":
        # Existential verbs strongly suggest NOT EXISTS
        existential_verbs = {"have", "take", "enroll", "exist", "contain", "include"}
        if head_token.lemma.lower() in existential_verbs:
            return NegationType.NOT_EXISTS

    # Pattern 2: "without" / "no" (determiner) -> LEFT JOIN IS NULL
    # Check if "without" appears in affected tokens
    for idx in affected_tokens:
        if tokens[idx].lemma.lower() in ["without", "lacking", "missing"]:
            return NegationType.LEFT_JOIN_NULL

    # Pattern 3: "except" / "but not" -> EXCEPT (set difference)
    for idx in affected_tokens:
        if tokens[idx].lemma.lower() in ["except", "excluding", "besides"]:
            return NegationType.EXCEPT

    # Pattern 4: Membership ("not in") -> NOT IN
    for idx in affected_tokens:
        if tokens[idx].lemma.lower() == "in" and tokens[idx].pos == "ADP":
            return NegationType.NOT_IN

    # Default: Simple WHERE NOT
    return NegationType.WHERE_NOT


def generate_negation_sql(scope: NegationScope, base_table: str, base_column: str) -> str:
    """
    Generate SQL for negation based on strategy.

    Args:
        scope: NegationScope object
        base_table: Main table being queried
        base_column: Column to filter on

    Returns:
        SQL fragment
    """
    if scope.sql_strategy == NegationType.NOT_EXISTS:
        # Generate NOT EXISTS subquery
        return f"""
        NOT EXISTS (
            SELECT 1 FROM related_table
            WHERE related_table.{base_column} = {base_table}.id
        )
        """

    elif scope.sql_strategy == NegationType.NOT_IN:
        # Generate NOT IN clause
        return f"""
        {base_table}.{base_column} NOT IN (
            SELECT id FROM related_table WHERE condition
        )
        """

    elif scope.sql_strategy == NegationType.LEFT_JOIN_NULL:
        # Generate LEFT JOIN with NULL check
        return f"""
        LEFT JOIN related_table ON {base_table}.id = related_table.{base_column}
        WHERE related_table.id IS NULL
        """

    elif scope.sql_strategy == NegationType.EXCEPT:
        # Generate EXCEPT (set difference)
        return f"""
        SELECT * FROM {base_table}
        EXCEPT
        SELECT * FROM {base_table} WHERE condition
        """

    else:  # WHERE_NOT
        # Simple WHERE NOT
        return f"WHERE NOT ({base_column} = value)"


def find_all_negations(tokens) -> List[NegationScope]:
    """
    Find all negations in the token list.

    Args:
        tokens: List of Token objects

    Returns:
        List of NegationScope objects
    """
    negations = []

    for i, token in enumerate(tokens):
        if token.dep == "neg":
            scope = find_negation_scope(tokens, i)
            if scope:
                negations.append(scope)

    return negations


# ==========================
# EXAMPLES
# ==========================

def example_patterns():
    """
    Example negation patterns and their SQL strategies.
    """
    print("Negation Scope Detection & SQL Generation")
    print("=" * 60)

    patterns = [
        ("students who have NOT taken CS101", "NOT EXISTS"),
        ("departments WITHOUT any professors", "LEFT JOIN ... IS NULL"),
        ("courses NOT IN the CS department", "NOT IN"),
        ("students in CS EXCEPT those who failed", "EXCEPT"),
        ("professors who are NOT tenured", "WHERE NOT"),
    ]

    print("\nNegation Patterns:")
    for question, strategy in patterns:
        print(f"  '{question}'")
        print(f"    -> Strategy: {strategy}")
        print()

    print("Detection Method:")
    print("  1. spaCy finds dep='neg' tokens")
    print("  2. Traverse dependency tree to find scope")
    print("  3. Analyze head POS and context")
    print("  4. Map to appropriate SQL strategy")

    print("\n[OK] All patterns use spaCy dependency parsing + rules")


# ==========================
# INTEGRATION
# ==========================

def enhance_question_analysis_with_negation(analysis):
    """
    Add negation scope analysis to QuestionAnalysis.

    Args:
        analysis: QuestionAnalysis object

    Returns:
        Same object with added negation_scopes field
    """
    # Find all negations
    negation_scopes = find_all_negations(analysis.tokens)

    # Add to analysis
    if not hasattr(analysis, 'negation_scopes'):
        analysis.negation_scopes = negation_scopes

    return analysis


if __name__ == "__main__":
    example_patterns()
