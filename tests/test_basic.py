"""
Basic tests for CHISEL components.
"""

import pytest
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chisel.schema_graph import SchemaGraph
from chisel.question_analysis import analyze_question, expand_with_wordnet
from chisel.schema_linking import link_question_to_schema, get_wordnet_similarity
from chisel.skeleton_prediction import predict_skeleton
from chisel.validation import validate_sql, check_sql_features


# Sample schema for testing
SAMPLE_SCHEMA = {
    "db_id": "test_db",
    "table_names_original": ["Student", "Course", "Enrollment"],
    "column_names_original": [
        [-1, "*"],
        [0, "student_id"],
        [0, "name"],
        [0, "age"],
        [1, "course_id"],
        [1, "title"],
        [2, "enrollment_id"],
        [2, "student_id"],
        [2, "course_id"]
    ],
    "column_types": ["text", "number", "text", "number", "number", "text", "number", "number", "number"],
    "primary_keys": [1, 4, 6],
    "foreign_keys": [[7, 1], [8, 4]]
}


def test_schema_graph():
    """Test schema graph construction."""
    schema = SchemaGraph.from_spider_json(SAMPLE_SCHEMA)

    # Check tables loaded
    assert len(schema.get_all_tables()) == 3
    assert "student" in schema.get_all_tables()

    # Check FK graph
    assert schema.fk_graph.has_edge("enrollment", "student")
    assert schema.fk_graph.has_edge("enrollment", "course")

    # Check join path
    path = schema.find_join_path(["student", "course"])
    assert len(path) > 0


def test_question_analysis():
    """Test question analysis."""
    question = "How many students are there?"
    analysis = analyze_question(question)

    # Check basic parsing
    assert len(analysis.tokens) > 0
    assert analysis.question_type == "count"
    assert "COUNT" in analysis.sql_signals


def test_wordnet_expansion():
    """Test WordNet expansion."""
    seeds = ["big", "large"]
    expanded = expand_with_wordnet(seeds)

    # Should expand to more synonyms
    assert len(expanded) >= len(seeds)
    assert "big" in expanded


def test_schema_linking():
    """Test schema linking."""
    schema = SchemaGraph.from_spider_json(SAMPLE_SCHEMA)
    question = "What are the names of students?"
    analysis = analyze_question(question)

    links = link_question_to_schema(analysis, schema)

    # Should find some links
    assert len(links) > 0

    # Should link "students" to "student" table
    table_links = [l for l in links if l.element_type == "table"]
    student_links = [l for l in table_links if "student" in l.schema_element]
    assert len(student_links) > 0


def test_wordnet_similarity():
    """Test WordNet similarity."""
    # Similar words
    sim1 = get_wordnet_similarity("student", "pupil")
    assert sim1 > 0.5  # Should be similar

    # Dissimilar words
    sim2 = get_wordnet_similarity("student", "car")
    assert sim2 < 0.5  # Should be dissimilar


def test_skeleton_prediction():
    """Test SQL skeleton prediction."""
    # COUNT question
    question = "How many students are there?"
    analysis = analyze_question(question)
    skeleton = predict_skeleton(analysis)

    assert skeleton.use_count is True

    # Superlative question
    question2 = "What is the oldest student?"
    analysis2 = analyze_question(question2)
    skeleton2 = predict_skeleton(analysis2)

    assert skeleton2.needs_order_by is True
    assert skeleton2.needs_limit is True


def test_sql_validation():
    """Test SQL validation."""
    # Valid SQL
    valid_sql = "SELECT * FROM student"
    is_valid, error = validate_sql(valid_sql)
    assert is_valid is True
    assert error is None

    # Invalid SQL
    invalid_sql = "SELECT invalid syntax"
    is_valid, error = validate_sql(invalid_sql)
    assert is_valid is False


def test_sql_features():
    """Test SQL feature detection."""
    sql = "SELECT COUNT(*) FROM student WHERE age > 20 ORDER BY age LIMIT 10"
    features = check_sql_features(sql)

    assert features["has_where"] is True
    assert features["has_order_by"] is True
    assert features["has_limit"] is True
    assert features["has_aggregation"] is True


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
