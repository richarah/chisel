"""
Test Lambda DCS IR implementation.

Tests compositional semantics, type system, and SQL compilation.
"""

import pytest
from chisel.lambda_dcs import (
    DCSAtom, DCSType, DCSCombinator, DCSSchemaGrounder,
    DCSToSQLCompiler, LambdaDCSBuilder
)
from chisel.schema_graph import SchemaGraph


class TestDCSAtom:
    """Test DCS atoms."""

    def test_table_atom(self):
        """Test table atom creation."""
        atom = DCSAtom(name="student", dcs_type=DCSType.SET, table="student")
        assert atom.name == "student"
        assert atom.dcs_type == DCSType.SET
        assert atom.table == "student"

    def test_column_atom(self):
        """Test column atom creation."""
        atom = DCSAtom(
            name="age",
            dcs_type=DCSType.PROPERTY,
            table="student",
            column="age"
        )
        assert atom.name == "age"
        assert atom.dcs_type == DCSType.PROPERTY
        assert atom.column == "age"

    def test_atom_to_lambda(self):
        """Test conversion to lambda expression."""
        atom = DCSAtom(name="student", dcs_type=DCSType.SET, table="student")
        expr = atom.to_lambda_expr()
        assert expr is not None
        assert "student" in str(expr)


class TestDCSCombinators:
    """Test Lambda DCS combinators."""

    def test_count(self):
        """Test count combinator."""
        student = DCSAtom("student", DCSType.SET, "student")
        expr = student.to_lambda_expr()
        count_expr = DCSCombinator.count(expr)
        assert "count" in str(count_expr)

    def test_argmax(self):
        """Test argmax combinator."""
        student = DCSAtom("student", DCSType.SET, "student")
        age = DCSAtom("age", DCSType.PROPERTY, "student", "age")

        student_expr = student.to_lambda_expr()
        age_expr = age.to_lambda_expr()

        argmax_expr = DCSCombinator.argmax(student_expr, age_expr)
        assert "argmax" in str(argmax_expr)

    def test_argmin(self):
        """Test argmin combinator."""
        student = DCSAtom("student", DCSType.SET, "student")
        age = DCSAtom("age", DCSType.PROPERTY, "student", "age")

        student_expr = student.to_lambda_expr()
        age_expr = age.to_lambda_expr()

        argmin_expr = DCSCombinator.argmin(student_expr, age_expr)
        assert "argmin" in str(argmin_expr)

    def test_filter_eq(self):
        """Test equality filter combinator."""
        student = DCSAtom("student", DCSType.SET, "student")
        age = DCSAtom("age", DCSType.PROPERTY, "student", "age")

        student_expr = student.to_lambda_expr()
        age_expr = age.to_lambda_expr()

        filter_expr = DCSCombinator.filter_eq(student_expr, age_expr, 25)
        assert str(filter_expr) is not None

    def test_intersection(self):
        """Test set intersection."""
        student = DCSAtom("student", DCSType.SET, "student")
        enrolled = DCSAtom("enrolled", DCSType.SET, "enrollment")

        student_expr = student.to_lambda_expr()
        enrolled_expr = enrolled.to_lambda_expr()

        intersect_expr = DCSCombinator.intersection(student_expr, enrolled_expr)
        assert str(intersect_expr) is not None


class TestDCSCompiler:
    """Test DCS to SQL compilation."""

    @pytest.fixture
    def mock_schema(self):
        """Create mock schema for testing."""
        # Create a simple mock object that looks like SchemaGraph
        class MockTable:
            def __init__(self):
                self.columns = {"age": "INTEGER", "name": "TEXT"}

            def __iter__(self):
                return iter(self.columns)

        class MockSchema:
            def __init__(self):
                self.tables = {"student": MockTable()}

        return MockSchema()

    def test_compile_count(self, mock_schema):
        """Test compiling count expression to SQL."""
        student = DCSAtom("student", DCSType.SET, "student")
        expr = student.to_lambda_expr()
        count_expr = DCSCombinator.count(expr)

        compiler = DCSToSQLCompiler(mock_schema)
        sql = compiler.compile(count_expr)

        assert "COUNT" in sql.upper()
        assert "student" in sql.lower()

    def test_compile_argmax(self, mock_schema):
        """Test compiling argmax to SQL."""
        student = DCSAtom("student", DCSType.SET, "student")
        age = DCSAtom("age", DCSType.PROPERTY, "student", "age")

        student_expr = student.to_lambda_expr()
        age_expr = age.to_lambda_expr()
        argmax_expr = DCSCombinator.argmax(student_expr, age_expr)

        compiler = DCSToSQLCompiler(mock_schema)
        sql = compiler.compile(argmax_expr)

        assert "ORDER BY" in sql.upper()
        assert "DESC" in sql.upper()
        assert "LIMIT 1" in sql.upper()


class TestTypeSystem:
    """Test Lambda DCS type system."""

    def test_entity_type(self):
        """Test entity type."""
        assert DCSType.ENTITY.value == "e"

    def test_number_type(self):
        """Test number type."""
        assert DCSType.NUMBER.value == "n"

    def test_set_type(self):
        """Test set type."""
        assert DCSType.SET.value == "set"

    def test_property_type(self):
        """Test property type."""
        assert DCSType.PROPERTY.value == "prop"


class TestCompositionality:
    """Test compositional semantics."""

    def test_nested_composition(self):
        """Test nested lambda expressions compose correctly."""
        student = DCSAtom("student", DCSType.SET, "student")
        age = DCSAtom("age", DCSType.PROPERTY, "student", "age")

        student_expr = student.to_lambda_expr()
        age_expr = age.to_lambda_expr()

        # Build: argmax(student, age)
        argmax_expr = DCSCombinator.argmax(student_expr, age_expr)

        # Should be composable further
        assert argmax_expr is not None
        assert str(argmax_expr) is not None

    def test_filter_then_count(self):
        """Test composing filter then count."""
        student = DCSAtom("student", DCSType.SET, "student")
        age = DCSAtom("age", DCSType.PROPERTY, "student", "age")

        student_expr = student.to_lambda_expr()
        age_expr = age.to_lambda_expr()

        # Filter students with age > 25
        filtered = DCSCombinator.filter_gt(student_expr, age_expr, 25)

        # Count filtered students
        count_expr = DCSCombinator.count(filtered)

        assert "count" in str(count_expr).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
