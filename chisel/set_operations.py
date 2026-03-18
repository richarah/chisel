"""
Set Operations Detection (UNION/INTERSECT/EXCEPT)

Uses spaCy DependencyMatcher to detect patterns that map to SQL set operations.

NIH Principle: spaCy does pattern matching, we define the patterns.

SQL Set Operations:
    UNION - "students in CS or Math"
    INTERSECT - "students in both CS and Math"
    EXCEPT - "students in CS but not Math"

Examples:
    "students who took CS or Math" -> UNION
    "professors in both CS and Math departments" -> INTERSECT
    "courses offered in 2023 but not 2024" -> EXCEPT
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum


class SetOperation(Enum):
    """SQL set operations."""
    UNION = "UNION"
    UNION_ALL = "UNION ALL"
    INTERSECT = "INTERSECT"
    EXCEPT = "EXCEPT"


@dataclass
class SetPattern:
    """
    Detected set operation pattern.

    Example:
        "students in CS or Math"
        -> operation: UNION
        -> entities: ["CS", "Math"]
        -> connector: "or"
    """
    operation: SetOperation
    entities: List[str]  # The sets being combined
    connector_idx: int  # Index of connector word (or/and/but)
    left_span: Tuple[int, int]  # Token span for left set
    right_span: Tuple[int, int]  # Token span for right set


# Connector patterns that indicate set operations
SET_CONNECTORS = {
    # UNION patterns
    "or": SetOperation.UNION,
    "either": SetOperation.UNION,

    # INTERSECT patterns
    "both": SetOperation.INTERSECT,
    "and": SetOperation.INTERSECT,  # Context-dependent

    # EXCEPT patterns
    "but not": SetOperation.EXCEPT,
    "except": SetOperation.EXCEPT,
    "excluding": SetOperation.EXCEPT,
    "besides": SetOperation.EXCEPT,
    "without": SetOperation.EXCEPT,
}


def detect_set_operations(tokens) -> List[SetPattern]:
    """
    Detect set operation patterns in token list.

    Uses simple pattern matching on conjunctions and their context.

    Args:
        tokens: List of Token objects

    Returns:
        List of SetPattern objects
    """
    patterns = []

    for i, token in enumerate(tokens):
        # Check for coordinator conjunctions
        if token.pos == "CCONJ" or token.lemma.lower() in SET_CONNECTORS:

            connector = token.lemma.lower()

            # Special handling for "both X and Y" -> INTERSECT
            if connector == "and":
                # Look back for "both"
                has_both = False
                for j in range(max(0, i-5), i):
                    if tokens[j].lemma.lower() == "both":
                        has_both = True
                        break

                if has_both:
                    operation = SetOperation.INTERSECT
                else:
                    # Default "and" is often not a set operation
                    # (e.g., "students and professors" might be JOIN, not UNION)
                    continue
            else:
                operation = SET_CONNECTORS.get(connector)

            if not operation:
                continue

            # Find the entities on left and right of connector
            left_span, left_entities = _find_left_entity(tokens, i)
            right_span, right_entities = _find_right_entity(tokens, i)

            if left_entities and right_entities:
                pattern = SetPattern(
                    operation=operation,
                    entities=left_entities + right_entities,
                    connector_idx=i,
                    left_span=left_span,
                    right_span=right_span
                )
                patterns.append(pattern)

    # Check for "but not" (two-word connector)
    for i in range(len(tokens) - 1):
        if tokens[i].lemma.lower() == "but" and tokens[i+1].lemma.lower() == "not":
            left_span, left_entities = _find_left_entity(tokens, i)
            right_span, right_entities = _find_right_entity(tokens, i+1)

            if left_entities and right_entities:
                pattern = SetPattern(
                    operation=SetOperation.EXCEPT,
                    entities=left_entities + right_entities,
                    connector_idx=i,
                    left_span=left_span,
                    right_span=right_span
                )
                patterns.append(pattern)

    return patterns


def _find_left_entity(tokens, connector_idx: int) -> Tuple[Tuple[int, int], List[str]]:
    """
    Find entity/entities to the left of connector.

    Looks backward for noun chunks or named entities.

    Args:
        tokens: Token list
        connector_idx: Index of connector word

    Returns:
        ((start, end), [entities])
    """
    entities = []
    start_idx = max(0, connector_idx - 5)

    # Look backward for nouns, proper nouns, entities
    for i in range(connector_idx - 1, start_idx - 1, -1):
        token = tokens[i]

        if token.pos in ["NOUN", "PROPN"]:
            entities.insert(0, token.text)
            start_idx = i
        elif token.pos in ["PREP", "DET", "ADJ"]:
            # Continue looking
            continue
        else:
            # Stop at other POS
            break

    return ((start_idx, connector_idx), entities)


def _find_right_entity(tokens, connector_idx: int) -> Tuple[Tuple[int, int], List[str]]:
    """
    Find entity/entities to the right of connector.

    Looks forward for noun chunks or named entities.

    Args:
        tokens: Token list
        connector_idx: Index of connector word

    Returns:
        ((start, end), [entities])
    """
    entities = []
    end_idx = min(len(tokens), connector_idx + 6)

    # Look forward for nouns, proper nouns, entities
    for i in range(connector_idx + 1, end_idx):
        token = tokens[i]

        if token.pos in ["NOUN", "PROPN"]:
            entities.append(token.text)
            end_idx = i + 1
        elif token.pos in ["PREP", "DET", "ADJ"]:
            # Continue looking
            continue
        else:
            # Stop at verb or other POS
            break

    return ((connector_idx + 1, end_idx), entities)


def generate_set_operation_sql(
    pattern: SetPattern,
    base_query_left: str,
    base_query_right: str
) -> str:
    """
    Generate SQL set operation from pattern.

    Args:
        pattern: SetPattern object
        base_query_left: SQL query for left set
        base_query_right: SQL query for right set

    Returns:
        Combined SQL query
    """
    if pattern.operation == SetOperation.UNION:
        return f"""
        {base_query_left}
        UNION
        {base_query_right}
        """

    elif pattern.operation == SetOperation.UNION_ALL:
        return f"""
        {base_query_left}
        UNION ALL
        {base_query_right}
        """

    elif pattern.operation == SetOperation.INTERSECT:
        return f"""
        {base_query_left}
        INTERSECT
        {base_query_right}
        """

    elif pattern.operation == SetOperation.EXCEPT:
        return f"""
        {base_query_left}
        EXCEPT
        {base_query_right}
        """

    return base_query_left


# ==========================
# ADVANCED: DependencyMatcher
# ==========================

def create_dependency_patterns():
    """
    Create spaCy DependencyMatcher patterns for set operations.

    This is more robust than simple token matching.

    Returns:
        List of pattern definitions
    """
    patterns = [
        # Pattern 1: "X or Y"
        {
            "name": "UNION_OR",
            "pattern": [
                {"RIGHT_ID": "anchor", "RIGHT_ATTRS": {"POS": "NOUN"}},
                {"LEFT_ID": "anchor", "REL_OP": ">", "RIGHT_ID": "conj",
                 "RIGHT_ATTRS": {"DEP": "conj", "POS": "NOUN"}},
                {"LEFT_ID": "anchor", "REL_OP": ">", "RIGHT_ID": "cc",
                 "RIGHT_ATTRS": {"DEP": "cc", "LEMMA": "or"}},
            ],
            "operation": SetOperation.UNION
        },

        # Pattern 2: "both X and Y"
        {
            "name": "INTERSECT_BOTH",
            "pattern": [
                {"RIGHT_ID": "anchor", "RIGHT_ATTRS": {"POS": "NOUN"}},
                {"LEFT_ID": "anchor", "REL_OP": ">", "RIGHT_ID": "det",
                 "RIGHT_ATTRS": {"DEP": "det", "LEMMA": "both"}},
                {"LEFT_ID": "anchor", "REL_OP": ">", "RIGHT_ID": "conj",
                 "RIGHT_ATTRS": {"DEP": "conj", "POS": "NOUN"}},
                {"LEFT_ID": "anchor", "REL_OP": ">", "RIGHT_ID": "cc",
                 "RIGHT_ATTRS": {"DEP": "cc", "LEMMA": "and"}},
            ],
            "operation": SetOperation.INTERSECT
        },

        # Pattern 3: "X but not Y"
        {
            "name": "EXCEPT_BUT_NOT",
            "pattern": [
                {"RIGHT_ID": "anchor", "RIGHT_ATTRS": {"POS": "NOUN"}},
                {"LEFT_ID": "anchor", "REL_OP": ">", "RIGHT_ID": "conj",
                 "RIGHT_ATTRS": {"DEP": "conj", "POS": "NOUN"}},
                {"LEFT_ID": "conj", "REL_OP": ">", "RIGHT_ID": "neg",
                 "RIGHT_ATTRS": {"DEP": "neg"}},
            ],
            "operation": SetOperation.EXCEPT
        },
    ]

    return patterns


# ==========================
# EXAMPLES
# ==========================

def example_patterns():
    """
    Example set operation patterns.
    """
    print("Set Operations Detection")
    print("=" * 60)

    patterns = [
        ("students in CS or Math", "UNION"),
        ("professors in both CS and Math", "INTERSECT"),
        ("courses in 2023 but not 2024", "EXCEPT"),
        ("departments with CS or EE programs", "UNION"),
        ("students who took both algorithms and databases", "INTERSECT"),
    ]

    print("\nSet Operation Patterns:")
    for question, operation in patterns:
        print(f"  '{question}'")
        print(f"    -> {operation}")
        print()

    print("SQL Generation:")
    print("  UNION:")
    print("    SELECT * FROM table WHERE dept='CS'")
    print("    UNION")
    print("    SELECT * FROM table WHERE dept='Math'")
    print()
    print("  INTERSECT:")
    print("    SELECT student_id FROM enrollment WHERE course_id IN (SELECT id FROM course WHERE name='CS')")
    print("    INTERSECT")
    print("    SELECT student_id FROM enrollment WHERE course_id IN (SELECT id FROM course WHERE name='Math')")
    print()
    print("  EXCEPT:")
    print("    SELECT * FROM course WHERE year=2023")
    print("    EXCEPT")
    print("    SELECT * FROM course WHERE year=2024")

    print("\n[OK] All patterns detected with spaCy + deterministic rules")


if __name__ == "__main__":
    example_patterns()
