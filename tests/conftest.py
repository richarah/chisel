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
    # Create schema using Spider JSON format
    spider_db = {
        "db_id": "university",
        "table_names_original": ["student", "course", "enrollment", "department", "professor"],
        "column_names_original": [
            [-1, "*"],  # Special * column
            [0, "id"], [0, "name"], [0, "age"], [0, "gpa"], [0, "dept_id"],
            [1, "id"], [1, "title"], [1, "credits"], [1, "dept_id"],
            [2, "student_id"], [2, "course_id"], [2, "grade"], [2, "semester"],
            [3, "id"], [3, "name"], [3, "budget"],
            [4, "id"], [4, "name"], [4, "dept_id"], [4, "salary"],
        ],
        "column_types": [
            "text",  # *
            "number", "text", "number", "number", "number",  # student
            "number", "text", "number", "number",  # course
            "number", "number", "text", "number",  # enrollment
            "number", "text", "number",  # department
            "number", "text", "number", "number",  # professor
        ],
        "primary_keys": [1, 6, 14, 17],  # student.id, course.id, department.id, professor.id
        "foreign_keys": [
            [5, 14],   # student.dept_id -> department.id
            [9, 14],   # course.dept_id -> department.id
            [10, 1],   # enrollment.student_id -> student.id
            [11, 6],   # enrollment.course_id -> course.id
            [19, 14],  # professor.dept_id -> department.id
        ],
    }

    schema = SchemaGraph.from_spider_json(spider_db)
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
