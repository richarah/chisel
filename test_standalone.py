#!/usr/bin/env python3
"""
Standalone test for CHISEL - no Spider data required.

This tests all components with in-memory schema data.
"""

import sys
from pathlib import Path

# Add chisel to path
sys.path.insert(0, str(Path(__file__).parent))

from chisel import (
    ChiselPipeline,
    SchemaGraph,
    analyze_question,
    link_question_to_schema,
    predict_skeleton,
    fill_sql_skeleton,
    validate_sql
)

# Define test schema in Spider format
TEST_SCHEMA = {
    "db_id": "university",
    "table_names_original": ["Student", "Course", "Enrollment", "Department"],
    "column_names_original": [
        [-1, "*"],
        [0, "student_id"],
        [0, "name"],
        [0, "age"],
        [0, "gpa"],
        [0, "dept_id"],
        [1, "course_id"],
        [1, "title"],
        [1, "credits"],
        [1, "dept_id"],
        [2, "enrollment_id"],
        [2, "student_id"],
        [2, "course_id"],
        [2, "grade"],
        [3, "dept_id"],
        [3, "dept_name"],
        [3, "building"],
    ],
    "column_types": [
        "text", "number", "text", "number", "number", "number",  # Student
        "number", "text", "number", "number",  # Course
        "number", "number", "number", "text",  # Enrollment
        "number", "text", "text"  # Department
    ],
    "primary_keys": [1, 6, 10, 14],
    "foreign_keys": [
        [5, 14],   # Student.dept_id -> Department.dept_id
        [9, 14],   # Course.dept_id -> Department.dept_id
        [11, 1],   # Enrollment.student_id -> Student.student_id
        [12, 6],   # Enrollment.course_id -> Course.course_id
    ]
}

def test_schema_graph():
    """Test schema graph construction."""
    print("\n=== Testing Schema Graph ===")
    schema = SchemaGraph.from_spider_json(TEST_SCHEMA)

    # Test tables
    tables = schema.get_all_tables()
    print(f"[OK] Loaded {len(tables)} tables: {tables}")
    assert len(tables) == 4
    assert "student" in tables

    # Test FK graph
    assert schema.fk_graph.has_edge("student", "department")
    assert schema.fk_graph.has_edge("enrollment", "student")
    print("[OK] FK graph built correctly")

    # Test join path
    path = schema.find_join_path(["student", "course"])
    print(f"[OK] Join path (student -> course): {len(path)} joins")
    assert len(path) > 0

    return schema


def test_question_analysis():
    """Test question analysis."""
    print("\n=== Testing Question Analysis ===")

    test_cases = [
        ("How many students are there?", "count", {"COUNT"}),
        ("What is the name of the student with the highest GPA?", "superlative", {"ORDER_DESC", "LIMIT"}),
        ("List all courses", "select", set()),
    ]

    for question, expected_type, expected_signals in test_cases:
        analysis = analyze_question(question)
        print(f"\nQuestion: {question}")
        print(f"  Type: {analysis.question_type} (expected: {expected_type})")
        print(f"  Signals: {analysis.sql_signals}")
        print(f"  Content words: {analysis.content_words[:5]}")
        print(f"  Noun chunks: {len(analysis.noun_chunks)}")

        # Check basic properties
        assert analysis.question_type == expected_type
        assert expected_signals.issubset(analysis.sql_signals) or len(expected_signals) == 0
        print("  [OK] PASS")


def test_schema_linking(schema):
    """Test schema linking."""
    print("\n=== Testing Schema Linking ===")

    question = "What are the names of students in the Computer Science department?"
    analysis = analyze_question(question)
    links = link_question_to_schema(analysis, schema)

    print(f"\nQuestion: {question}")
    print(f"Found {len(links)} links:")

    for link in links[:10]:
        print(f"  '{link.question_phrase}' -> {link.schema_element} ({link.link_type.value}, {link.score:.1f})")

    # Should find student table and name column
    table_links = [l for l in links if l.element_type == "table" and "student" in l.schema_element]
    column_links = [l for l in links if l.element_type == "column" and "name" in l.column_name]

    assert len(table_links) > 0, "Should find student table"
    assert len(column_links) > 0, "Should find name column"
    print("[OK] PASS")


def test_skeleton_prediction():
    """Test SQL skeleton prediction."""
    print("\n=== Testing Skeleton Prediction ===")

    test_cases = [
        ("How many students are there?", lambda s: s.use_count),
        ("What is the student with the highest GPA?", lambda s: s.needs_order_by and s.needs_limit),
        ("List students who are older than 20", lambda s: s.needs_where),
    ]

    for question, check_fn in test_cases:
        analysis = analyze_question(question)
        skeleton = predict_skeleton(analysis)

        print(f"\nQuestion: {question}")
        print(f"  Aggregation: {skeleton.has_aggregation()}")
        print(f"  WHERE: {skeleton.needs_where}")
        print(f"  ORDER BY: {skeleton.needs_order_by}")
        print(f"  LIMIT: {skeleton.needs_limit}")

        assert check_fn(skeleton), f"Skeleton check failed for: {question}"
        print("  [OK] PASS")


def test_slot_filling(schema):
    """Test SQL slot filling."""
    print("\n=== Testing Slot Filling ===")

    question = "How many students are there?"
    analysis = analyze_question(question)
    links = link_question_to_schema(analysis, schema)
    skeleton = predict_skeleton(analysis)

    filled = fill_sql_skeleton(skeleton, analysis, links, schema)

    if filled:
        sql = filled.to_sql()
        print(f"\nQuestion: {question}")
        print(f"SQL: {sql}")

        # Should be a COUNT query
        assert "COUNT" in sql.upper()
        assert "student" in sql.lower()
        print("[OK] PASS")
    else:
        print("[X] FAIL: Could not fill SQL skeleton")
        sys.exit(1)


def test_validation():
    """Test SQL validation."""
    print("\n=== Testing SQL Validation ===")

    valid_queries = [
        "SELECT * FROM student",
        "SELECT COUNT(*) FROM student WHERE age > 20",
        "SELECT name FROM student ORDER BY gpa DESC LIMIT 1",
    ]

    invalid_queries = [
        "SELECT invalid syntax here",
        "",
        "not a query",
    ]

    for sql in valid_queries:
        is_valid, error = validate_sql(sql)
        print(f"'{sql}' -> {'VALID' if is_valid else f'INVALID: {error}'}")
        assert is_valid, f"Should be valid: {sql}"

    for sql in invalid_queries:
        is_valid, error = validate_sql(sql)
        print(f"'{sql}' -> {'VALID' if is_valid else 'INVALID (expected)'}")
        assert not is_valid, f"Should be invalid: {sql}"

    print("[OK] PASS")


def main():
    """Run all tests."""
    print("="*80)
    print("CHISEL Standalone Test Suite")
    print("="*80)

    try:
        # Test each component
        schema = test_schema_graph()
        test_question_analysis()
        test_schema_linking(schema)
        test_skeleton_prediction()
        test_slot_filling(schema)
        test_validation()

        print("\n" + "="*80)
        print("[OK] ALL TESTS PASSED")
        print("="*80)
        print("\nCHISEL is working correctly!")
        print("\nNext steps:")
        print("1. Download Spider: https://yale-lily.github.io/spider")
        print("2. Extract to data/spider/")
        print("3. Run: python evaluation/evaluate.py")

        return 0

    except Exception as e:
        print(f"\n[X] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
