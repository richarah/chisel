"""
Task 4: Skeleton Prediction

Determine which SQL clauses are needed based on detected signals.

This is pure rule-based logic - no libraries needed.
We write rules that map linguistic patterns -> SQL structure.

Rules based on:
- Standard SQL-92 semantics
- Common English question patterns
- Linguistic analysis from spaCy
"""

from dataclasses import dataclass
from typing import Set, Optional
from .question_analysis import QuestionAnalysis


@dataclass
class SQLSkeleton:
    """
    SQL query structure prediction.

    Flags indicating which clauses/features are needed.
    """
    # SELECT clause
    select_distinct: bool = False
    select_columns: bool = True  # Default: select columns

    # Aggregation
    use_count: bool = False
    use_max: bool = False
    use_min: bool = False
    use_avg: bool = False
    use_sum: bool = False

    # FROM clause
    needs_join: bool = False  # Will be determined by number of tables

    # WHERE clause
    needs_where: bool = False

    # GROUP BY clause
    needs_group_by: bool = False

    # HAVING clause
    needs_having: bool = False

    # ORDER BY clause
    needs_order_by: bool = False
    order_desc: bool = True  # Default descending for superlatives

    # LIMIT clause
    needs_limit: bool = False
    limit_value: Optional[int] = None

    # Set operations
    use_union: bool = False
    use_intersect: bool = False
    use_except: bool = False

    # Subqueries
    needs_subquery: bool = False
    use_exists: bool = False
    use_not_exists: bool = False

    def has_aggregation(self) -> bool:
        """Check if any aggregation is needed."""
        return any([
            self.use_count, self.use_max, self.use_min,
            self.use_avg, self.use_sum
        ])


def predict_skeleton(analysis: QuestionAnalysis) -> SQLSkeleton:
    """
    Predict SQL skeleton from question analysis.

    Rules based on SQL semantics and English grammar patterns.
    """
    skeleton = SQLSkeleton()
    signals = analysis.sql_signals

    # ==========================
    # AGGREGATION DETECTION
    # ==========================
    # Rule: COUNT signal or "how many" -> COUNT(*)
    if "COUNT" in signals or analysis.question_type == "count":
        skeleton.use_count = True

    # Rule: MAX/MIN signals or superlatives -> MAX/MIN
    if "MAX" in signals or (analysis.superlatives and "most" in analysis.original_text.lower()):
        skeleton.use_max = True

    if "MIN" in signals or (analysis.superlatives and "least" in analysis.original_text.lower()):
        skeleton.use_min = True

    # Rule: AVG signal -> AVG
    if "AVG" in signals:
        skeleton.use_avg = True

    # Rule: SUM signal -> SUM
    if "SUM" in signals:
        skeleton.use_sum = True

    # ==========================
    # SELECT CLAUSE
    # ==========================
    # Rule: DISTINCT signal -> SELECT DISTINCT
    if "DISTINCT" in signals:
        skeleton.select_distinct = True

    # ==========================
    # WHERE CLAUSE
    # ==========================
    # Rule: Comparison signals -> WHERE needed
    if analysis.has_comparison():
        skeleton.needs_where = True

    # Rule: Quoted values, entities, or numbers -> likely WHERE
    if analysis.quoted_values or analysis.entities or analysis.numeric_values:
        skeleton.needs_where = True

    # Rule: Negation -> WHERE with NOT/!=
    if analysis.negations:
        skeleton.needs_where = True

    # ==========================
    # GROUP BY CLAUSE
    # ==========================
    # Rule: GROUP_BY signal -> GROUP BY needed
    if "GROUP_BY" in signals:
        skeleton.needs_group_by = True

    # Rule: Aggregation + non-aggregated column -> GROUP BY needed
    # (This will be refined in slot filling when we know actual columns)
    if skeleton.has_aggregation():
        # Check for "each", "every", "per" patterns
        for token in analysis.tokens:
            if token.lemma in ["each", "every", "per"]:
                skeleton.needs_group_by = True
                break

    # ==========================
    # HAVING CLAUSE
    # ==========================
    # Rule: Aggregation + comparison -> might need HAVING
    # Example: "departments with more than 10 students"
    if skeleton.has_aggregation() and analysis.has_comparison():
        skeleton.needs_having = True

    # ==========================
    # ORDER BY CLAUSE
    # ==========================
    # Rule: Superlative -> ORDER BY + LIMIT 1
    if analysis.superlatives:
        skeleton.needs_order_by = True
        skeleton.needs_limit = True
        skeleton.limit_value = 1

        # Determine order direction
        superlative_text = " ".join([s[1] for s in analysis.superlatives])
        if any(w in superlative_text for w in ["most", "biggest", "largest", "highest", "greatest"]):
            skeleton.order_desc = True
        elif any(w in superlative_text for w in ["least", "smallest", "lowest", "fewest"]):
            skeleton.order_desc = False

    # Rule: ORDER signals
    if "ORDER_DESC" in signals:
        skeleton.needs_order_by = True
        skeleton.order_desc = True

    if "ORDER_ASC" in signals:
        skeleton.needs_order_by = True
        skeleton.order_desc = False

    # Rule: LIMIT signal
    if "LIMIT" in signals:
        skeleton.needs_limit = True

        # Try to extract limit value from question
        for idx, num_val in analysis.numeric_values:
            # Check if this number is near "top", "first", etc.
            if idx > 0:
                prev_token = analysis.tokens[idx - 1].lemma
                if prev_token in ["top", "first", "last"]:
                    skeleton.limit_value = int(num_val)
                    break

    # ==========================
    # SET OPERATIONS
    # ==========================
    # Rule: UNION signal -> UNION
    if "UNION" in signals:
        skeleton.use_union = True

    # Rule: INTERSECT signal -> INTERSECT
    if "INTERSECT" in signals:
        skeleton.use_intersect = True

    # Rule: EXCEPT signal -> EXCEPT
    if "EXCEPT" in signals:
        skeleton.use_except = True

    # ==========================
    # SUBQUERIES
    # ==========================
    # Rule: EXISTS signals -> subquery with EXISTS
    if "EXISTS" in signals:
        skeleton.needs_subquery = True
        skeleton.use_exists = True

    if "NOT_EXISTS" in signals:
        skeleton.needs_subquery = True
        skeleton.use_not_exists = True

    # Rule: Comparison with aggregation might need subquery
    # Example: "students with GPA higher than average"
    if skeleton.has_aggregation() and analysis.comparatives:
        skeleton.needs_subquery = True

    # ==========================
    # SPECIAL PATTERNS
    # ==========================
    # Rule: "how many X have Y" -> COUNT with GROUP BY
    # Example: "How many students does each department have?"
    if skeleton.use_count and skeleton.needs_group_by:
        # This is COUNT(...) ... GROUP BY pattern
        pass

    # Rule: Count with comparison -> HAVING
    # Example: "departments with more than 10 students"
    if skeleton.use_count and analysis.has_comparison():
        skeleton.needs_having = True
        skeleton.needs_group_by = True

    return skeleton


def refine_skeleton_with_links(
    skeleton: SQLSkeleton,
    analysis: QuestionAnalysis,
    num_tables: int
) -> SQLSkeleton:
    """
    Refine skeleton prediction based on schema linking results.

    Args:
        skeleton: Initial skeleton prediction
        analysis: Question analysis
        num_tables: Number of tables involved (from schema linking)
    """
    # Rule: Multiple tables -> need JOINs
    if num_tables > 1:
        skeleton.needs_join = True

    # Rule: No aggregation but GROUP BY signal -> might be DISTINCT instead
    if skeleton.needs_group_by and not skeleton.has_aggregation():
        skeleton.select_distinct = True
        skeleton.needs_group_by = False

    return skeleton


if __name__ == "__main__":
    # Test
    from .question_analysis import analyze_question

    test_questions = [
        "How many students are there?",
        "What is the name of the student with the highest GPA?",
        "List all courses that have more than 100 students.",
        "Which department has the most professors?",
        "Show me distinct course titles.",
        "Find students who are enrolled in both CS101 and CS102.",
    ]

    for q in test_questions:
        print(f"\nQuestion: {q}")
        analysis = analyze_question(q)
        skeleton = predict_skeleton(analysis)

        print(f"  Signals: {analysis.sql_signals}")
        print(f"  Aggregation: {skeleton.has_aggregation()}")
        if skeleton.use_count:
            print(f"    -> COUNT")
        if skeleton.use_max or skeleton.use_min:
            print(f"    -> MAX/MIN")
        print(f"  WHERE: {skeleton.needs_where}")
        print(f"  GROUP BY: {skeleton.needs_group_by}")
        print(f"  ORDER BY: {skeleton.needs_order_by} ({'DESC' if skeleton.order_desc else 'ASC'})")
        print(f"  LIMIT: {skeleton.needs_limit} ({skeleton.limit_value})")
        print(f"  DISTINCT: {skeleton.select_distinct}")
