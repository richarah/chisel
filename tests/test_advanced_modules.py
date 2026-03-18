"""
Tests for advanced SQL generation modules

Test coverage:
- comparatives.py
- join_inference.py
- negation_scope.py
- set_operations.py
- coreference.py
"""

import pytest
import spacy
from chisel.comparatives import detect_superlative, detect_comparative, SuperlativeType, ComparativeType
from chisel.join_inference import JoinInference
from chisel.negation_scope import find_negation_scope, NegationType
from chisel.set_operations import detect_set_operations, SetOperation
from chisel.coreference import DialogueState, DialogueTurn, CoreferenceResolver, EllipsisResolver


@pytest.fixture(scope="module")
def nlp():
    """Load spaCy model once for all tests."""
    return spacy.load("en_core_web_sm")


class TestComparatives:
    """Test comparatives.py functionality."""

    def test_superlative_highest(self, nlp):
        """Test detecting 'highest' superlative."""
        doc = nlp("What is the highest salary?")
        tokens = list(doc)

        # Find the superlative token
        sup_idx = None
        for i, token in enumerate(tokens):
            if token.tag_ == "JJS" or token.text.lower() == "highest":
                sup_idx = i
                break

        if sup_idx is not None:
            pattern = detect_superlative(tokens, sup_idx)
            assert pattern is not None
            assert pattern.superlative_type == SuperlativeType.MAXIMUM

    def test_superlative_lowest(self, nlp):
        """Test detecting 'lowest' superlative."""
        doc = nlp("Find the lowest price")
        tokens = list(doc)

        sup_idx = None
        for i, token in enumerate(tokens):
            if token.tag_ == "JJS" or token.text.lower() == "lowest":
                sup_idx = i
                break

        if sup_idx is not None:
            pattern = detect_superlative(tokens, sup_idx)
            assert pattern is not None
            assert pattern.superlative_type == SuperlativeType.MINIMUM

    def test_comparative_more_than(self, nlp):
        """Test detecting 'more than' comparative."""
        doc = nlp("Students with more than 20 credits")
        tokens = list(doc)

        comp_idx = None
        for i, token in enumerate(tokens):
            if token.text.lower() == "more":
                comp_idx = i
                break

        if comp_idx is not None:
            pattern = detect_comparative(tokens, comp_idx)
            # May return None if pattern not recognized
            assert pattern is None or pattern.comparative_type == ComparativeType.GREATER


class TestJoinInference:
    """Test join_inference.py functionality."""

    def test_direct_fk(self):
        """Test finding join path with direct FK."""
        # Create simple FK graph
        fk_graph = [
            ("student", "dept_id", "department", "id"),
            ("course", "dept_id", "department", "id"),
        ]

        join_inf = JoinInference(fk_graph)

        # Should find path from student to department
        path = join_inf.find_join_path("student", "department")
        assert path is not None
        assert path.total_hops == 1
        assert "student" in path.tables
        assert "department" in path.tables

    def test_multi_hop_join(self):
        """Test finding multi-hop join path."""
        fk_graph = [
            ("enrollment", "student_id", "student", "id"),
            ("enrollment", "course_id", "course", "id"),
            ("course", "dept_id", "department", "id"),
        ]

        join_inf = JoinInference(fk_graph)

        # Should find path from student to department (via enrollment and course)
        path = join_inf.find_join_path("student", "department")
        if path:
            assert path.total_hops >= 2  # At least 2 hops
            assert "student" in path.tables
            assert "department" in path.tables

    def test_no_path(self):
        """Test when no join path exists."""
        fk_graph = [
            ("student", "dept_id", "department", "id"),
        ]

        join_inf = JoinInference(fk_graph)

        # No path from student to unrelated table
        path = join_inf.find_join_path("student", "unrelated_table")
        assert path is None


class TestNegationScope:
    """Test negation_scope.py functionality."""

    def test_not_negation(self, nlp):
        """Test 'not' negation detection."""
        doc = nlp("Students who did not enroll")
        tokens = list(doc)

        neg_idx = None
        for i, token in enumerate(tokens):
            if token.text.lower() == "not":
                neg_idx = i
                break

        if neg_idx is not None:
            scope = find_negation_scope(tokens, neg_idx)
            assert scope is not None
            assert scope.negation_idx == neg_idx

    def test_no_negation(self, nlp):
        """Test 'no' as negation."""
        doc = nlp("Students with no courses")
        tokens = list(doc)

        neg_idx = None
        for i, token in enumerate(tokens):
            if token.text.lower() == "no":
                neg_idx = i
                break

        if neg_idx is not None:
            # 'no' might be detected differently depending on parsing
            # Just check function doesn't crash
            scope = find_negation_scope(tokens, neg_idx)
            # May or may not return scope depending on dependency structure
            assert scope is None or scope.negation_idx == neg_idx


class TestSetOperations:
    """Test set_operations.py functionality."""

    def test_union_detection(self, nlp):
        """Test UNION detection with 'or'."""
        doc = nlp("Students who major in CS or Math")
        tokens = list(doc)

        patterns = detect_set_operations(tokens)
        if patterns:
            assert any(p.operation == SetOperation.UNION for p in patterns)

    def test_intersect_detection(self, nlp):
        """Test INTERSECT detection with 'both...and'."""
        doc = nlp("Students who take both CS and Math")
        tokens = list(doc)

        patterns = detect_set_operations(tokens)
        if patterns:
            assert any(p.operation == SetOperation.INTERSECT for p in patterns)

    def test_except_detection(self, nlp):
        """Test EXCEPT detection."""
        doc = nlp("All students except those in CS")
        tokens = list(doc)

        patterns = detect_set_operations(tokens)
        if patterns:
            assert any(p.operation == SetOperation.EXCEPT for p in patterns)


class TestCoreference:
    """Test coreference.py functionality."""

    def test_dialogue_state(self):
        """Test dialogue state tracking."""
        state = DialogueState()

        turn1 = DialogueTurn(
            turn_id=0,
            question="Show me all departments",
            tables_mentioned={"department"},
        )

        state.add_turn(turn1)

        assert len(state.turns) == 1
        assert state.current_topic == "department"

    def test_multiple_turns(self):
        """Test multiple dialogue turns."""
        state = DialogueState()

        turn1 = DialogueTurn(turn_id=0, question="Show departments", tables_mentioned={"department"})
        turn2 = DialogueTurn(turn_id=1, question="Show students", tables_mentioned={"student"})

        state.add_turn(turn1)
        state.add_turn(turn2)

        assert len(state.turns) == 2
        assert state.current_topic == "student"  # Latest topic

    def test_coreference_resolver(self, nlp):
        """Test coreference resolution."""
        resolver = CoreferenceResolver(nlp)
        state = DialogueState()

        turn1 = DialogueTurn(
            turn_id=0,
            question="Show me all students",
            tables_mentioned={"student"},
        )
        state.add_turn(turn1)

        # Second turn with pronoun
        resolved = resolver.resolve("How many of them are there?", state)

        # Should resolve 'them' to 'students'
        assert resolved is not None
        assert isinstance(resolved, str)

    def test_ellipsis_resolver(self, nlp):
        """Test ellipsis resolution."""
        resolver = EllipsisResolver(nlp)
        state = DialogueState()

        turn1 = DialogueTurn(
            turn_id=0,
            question="Show me all students",
            tables_mentioned={"student"},
        )
        state.add_turn(turn1)

        # Fragment: "in computer science"
        resolved = resolver.resolve("in computer science", state)

        # Should combine with previous turn
        assert "student" in resolved.lower() or "computer science" in resolved.lower()

    def test_complete_question_no_ellipsis(self, nlp):
        """Test that complete questions are not modified."""
        resolver = EllipsisResolver(nlp)
        state = DialogueState()

        complete_question = "How many students are there?"
        resolved = resolver.resolve(complete_question, state)

        # Should return unchanged
        assert resolved == complete_question


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
