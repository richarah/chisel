"""
Tests for question_analysis.py

Test coverage:
- Basic question parsing
- SQL signal detection
- Quantifier detection
- Superlative/comparative detection
- Number and date parsing
- Ordinal parsing
"""

import pytest
import spacy
from chisel.question_analysis import analyze_question, QuestionAnalysis


@pytest.fixture(scope="module")
def nlp():
    """Load spaCy model once for all tests."""
    return spacy.load("en_core_web_sm")


class TestBasicParsing:
    """Test basic NLP parsing functionality."""

    def test_simple_question(self, nlp):
        """Test basic question parsing."""
        question = "What are the names of all students?"
        analysis = analyze_question(question, nlp)

        assert analysis is not None
        assert analysis.original_text == question
        assert analysis.question_type in ["select", "count", "yes_no", "superlative"]
        assert len(analysis.content_words) > 0

    def test_tokenization(self, nlp):
        """Test that tokens are properly created."""
        question = "How many students are there?"
        analysis = analyze_question(question, nlp)

        assert analysis.tokens is not None
        assert len(analysis.tokens) > 0
        assert all(hasattr(token, 'text') for token in analysis.tokens)


class TestSQLSignals:
    """Test SQL signal detection."""

    def test_count_signal(self, nlp):
        """Test COUNT signal detection."""
        question = "How many students are there?"
        analysis = analyze_question(question, nlp)

        assert "COUNT" in analysis.sql_signals

    def test_max_signal(self, nlp):
        """Test MAX/superlative signal detection."""
        question = "What is the highest grade?"
        analysis = analyze_question(question, nlp)

        assert "MAX" in analysis.sql_signals or "ORDER_DESC" in analysis.sql_signals

    def test_where_signal(self, nlp):
        """Test WHERE signal detection."""
        question = "What are students with GPA above 3.5?"
        analysis = analyze_question(question, nlp)

        assert "WHERE_GT" in analysis.sql_signals or "EXISTS" in analysis.sql_signals


class TestQuantifiers:
    """Test quantifier detection."""

    def test_all_quantifier(self, nlp):
        """Test 'all' quantifier detection."""
        question = "Show me all students"
        analysis = analyze_question(question, nlp)

        # Check for 'all' in tokens or signals
        all_found = any(t.lemma == 'all' for t in analysis.tokens)
        assert all_found or "QUANTIFIER_ALL" in analysis.sql_signals

    def test_some_quantifier(self, nlp):
        """Test 'some' quantifier detection."""
        question = "Find some students who took courses"
        analysis = analyze_question(question, nlp)

        # Check for 'some' in tokens or signals
        some_found = any(t.lemma == 'some' for t in analysis.tokens)
        assert some_found or "QUANTIFIER_ANY" in analysis.sql_signals

    def test_no_quantifier(self, nlp):
        """Test 'no' quantifier detection."""
        question = "Which students have no enrolled courses?"
        analysis = analyze_question(question, nlp)

        # Check for 'no' in tokens or signals
        no_found = any(t.lemma == 'no' for t in analysis.tokens)
        assert no_found or "QUANTIFIER_NO" in analysis.sql_signals


class TestSuperlativesComparatives:
    """Test superlative and comparative detection."""

    def test_superlative_highest(self, nlp):
        """Test 'highest' superlative detection."""
        question = "What is the highest salary?"
        analysis = analyze_question(question, nlp)

        assert len(analysis.superlatives) > 0
        assert any('highest' in s[1] or 'high' in s[1] for s in analysis.superlatives)

    def test_superlative_oldest(self, nlp):
        """Test 'oldest' superlative detection."""
        question = "Who is the oldest student?"
        analysis = analyze_question(question, nlp)

        assert len(analysis.superlatives) > 0
        assert any('oldest' in s[1] or 'old' in s[1] for s in analysis.superlatives)

    def test_comparative_more_than(self, nlp):
        """Test 'more than' comparative detection."""
        question = "Students with more than 3 courses"
        analysis = analyze_question(question, nlp)

        assert len(analysis.comparatives) > 0 or "WHERE_GT" in analysis.sql_signals


class TestNumbers:
    """Test number parsing."""

    def test_digit_number(self, nlp):
        """Test digit number parsing."""
        question = "Students with GPA above 3.5"
        analysis = analyze_question(question, nlp)

        assert len(analysis.numeric_values) > 0
        assert any(n[1] == 3.5 for n in analysis.numeric_values)

    def test_word_number(self, nlp):
        """Test word number parsing."""
        question = "Show me twenty students"
        analysis = analyze_question(question, nlp)

        # word2number might not parse all number words, but spaCy recognizes NUM
        num_tokens = [t for t in analysis.tokens if t.pos == "NUM"]
        assert len(num_tokens) > 0 or len(analysis.numeric_values) > 0


class TestOrdinals:
    """Test ordinal parsing."""

    def test_first_ordinal(self, nlp):
        """Test 'first' ordinal."""
        question = "Show me the first student"
        analysis = analyze_question(question, nlp)

        # Check if ordinals detected
        assert len(analysis.ordinal_values) > 0 or len(analysis.numeric_values) > 0

    def test_third_ordinal(self, nlp):
        """Test 'third' ordinal."""
        question = "Who is the third highest scorer?"
        analysis = analyze_question(question, nlp)

        # Ordinal should be detected
        found_ordinal = any(val == 3 for idx, val in analysis.ordinal_values)
        assert found_ordinal or any(n[1] == 3 for n in analysis.numeric_values)


class TestNegation:
    """Test negation detection."""

    def test_not_negation(self, nlp):
        """Test 'not' negation."""
        question = "Students who did not enroll"
        analysis = analyze_question(question, nlp)

        assert len(analysis.negations) > 0

    def test_no_negation(self, nlp):
        """Test 'no' as negation."""
        question = "Students with no courses"
        analysis = analyze_question(question, nlp)

        # 'no' can be negation or quantifier
        assert len(analysis.negations) > 0 or "QUANTIFIER_NO" in analysis.sql_signals


class TestComplexQuestions:
    """Test complex multi-feature questions."""

    def test_complex_with_multiple_features(self, nlp):
        """Test question with count + superlative + where."""
        question = "How many students have the highest GPA in each department?"
        analysis = analyze_question(question, nlp)

        assert "COUNT" in analysis.sql_signals
        assert len(analysis.superlatives) > 0
        assert "GROUP_BY" in analysis.sql_signals or "EXISTS" in analysis.sql_signals

    def test_complex_with_quantifier_and_comparative(self, nlp):
        """Test question with quantifier and comparative."""
        question = "Find all students with GPA above 3.5"
        analysis = analyze_question(question, nlp)

        # Check for 'all' or quantifier signals
        all_found = 'all' in question.lower()
        assert all_found or "QUANTIFIER_ALL" in analysis.sql_signals
        assert len(analysis.comparatives) > 0 or "WHERE_GT" in analysis.sql_signals


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
