"""
Pytest configuration and shared fixtures for CHISEL tests.

Fixtures:
- nlp: Shared spaCy model (loaded once per session)
- sample_schema: Standard test database schema
- sample_questions: Common test questions
"""

import pytest
import spacy
from chisel.schema_graph import SchemaGraph


@pytest.fixture(scope="session")
def nlp():
    """
    Load spaCy model once for entire test session.

    This is expensive, so we cache it at session scope.
    """
    return spacy.load("en_core_web_sm")


@pytest.fixture
def sample_schema():
    """
    Create a standard test database schema.

    Schema represents a simple university database:
    - student(id, name, age, gpa, dept_id)
    - course(id, title, credits, dept_id)
    - enrollment(student_id, course_id, grade, semester)
    - department(id, name, budget)
    - professor(id, name, dept_id, salary)
    """
    schema = SchemaGraph(
        db_id="university",
        tables=["student", "course", "enrollment", "department", "professor"],
        columns={
            "student": ["id", "name", "age", "gpa", "dept_id"],
            "course": ["id", "title", "credits", "dept_id"],
            "enrollment": ["student_id", "course_id", "grade", "semester"],
            "department": ["id", "name", "budget"],
            "professor": ["id", "name", "dept_id", "salary"],
        },
        primary_keys={
            "student": "id",
            "course": "id",
            "enrollment": None,
            "department": "id",
            "professor": "id",
        },
        foreign_keys=[
            ("student", "dept_id", "department", "id"),
            ("course", "dept_id", "department", "id"),
            ("enrollment", "student_id", "student", "id"),
            ("enrollment", "course_id", "course", "id"),
            ("professor", "dept_id", "department", "id"),
        ],
    )
    return schema


@pytest.fixture
def sample_questions():
    """
    Provide common test questions with expected features.

    Returns dict with questions categorized by type:
    - simple: Basic SELECT queries
    - aggregation: COUNT, MAX, AVG, etc.
    - superlative: "highest", "oldest", etc.
    - comparative: "more than", "less than"
    - quantifier: "all", "some", "no"
    - negation: "not", "without"
    - multi_table: Requires joins
    """
    return {
        "simple": [
            "What are the names of all students?",
            "Show me student ages",
            "List all courses",
        ],
        "aggregation": [
            "How many students are there?",
            "What is the average GPA?",
            "What is the total budget?",
        ],
        "superlative": [
            "Who is the oldest student?",
            "What is the highest salary?",
            "Which course has the most credits?",
        ],
        "comparative": [
            "Students with GPA above 3.5",
            "Professors earning more than 100000",
            "Departments with budget less than 50000",
        ],
        "quantifier": [
            "Show me all students",
            "Find some courses in CS",
            "Students who took no courses",
        ],
        "negation": [
            "Students who did not enroll",
            "Professors with no publications",
            "Departments without budget",
        ],
        "multi_table": [
            "Show students enrolled in courses",
            "Which students are in computer science department?",
            "Professors teaching courses",
        ],
    }


@pytest.fixture
def simple_fk_graph():
    """
    Simple foreign key graph for join inference tests.

    Structure:
    student --[dept_id]--> department
    course --[dept_id]--> department
    enrollment --[student_id]--> student
    enrollment --[course_id]--> course
    professor --[dept_id]--> department
    """
    return [
        ("student", "dept_id", "department", "id"),
        ("course", "dept_id", "department", "id"),
        ("enrollment", "student_id", "student", "id"),
        ("enrollment", "course_id", "course", "id"),
        ("professor", "dept_id", "department", "id"),
    ]


# Test markers for categorizing tests
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (slower, requires multiple modules)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (takes >1 second)"
    )
    config.addinivalue_line(
        "markers", "requires_spacy: mark test as requiring spaCy model"
    )
    config.addinivalue_line(
        "markers", "requires_nltk: mark test as requiring NLTK data"
    )
