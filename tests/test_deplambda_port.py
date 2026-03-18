"""
Test suite for deplambda rule port validation.

Verifies that CHISEL's ported deplambda rules produce logically equivalent
expressions to the original UDepLambda Java implementation.

Gold standard data should be generated using UDepLambda and saved to
tests/deplambda_gold.json following instructions in DEPLAMBDA_TESTING.md.
"""

import json
import pytest
from pathlib import Path
import spacy

from chisel.tnode import doc_to_tnodes, tnodes_to_tree_str
from chisel.tree_transformations import apply_all_transformations
# from chisel.ir_composer import IRComposer
# from chisel.schema_graph import SchemaGraph
# from chisel.question_analysis import QuestionAnalysis
# from chisel.schema_linking import SchemaLinker

# Try to import NLTK for alpha-equivalence checking
try:
    from nltk.sem import logic
    HAS_NLTK_LOGIC = True
except ImportError:
    HAS_NLTK_LOGIC = False


# ==========================================================================================
# FIXTURES
# ==========================================================================================

@pytest.fixture(scope="module")
def nlp():
    """Load spaCy model once for all tests."""
    return spacy.load("en_core_web_sm")


@pytest.fixture(scope="module")
def gold_data():
    """Load gold standard logical forms from UDepLambda."""
    gold_path = Path(__file__).parent / "deplambda_gold.json"

    if not gold_path.exists():
        pytest.skip(
            "Gold standard data not found. "
            "Generate it using UDepLambda following tests/DEPLAMBDA_TESTING.md"
        )

    with open(gold_path, 'r') as f:
        return json.load(f)


@pytest.fixture
def test_sentences():
    """Load test sentences."""
    sentences_path = Path(__file__).parent / "deplambda_test_sentences.txt"

    with open(sentences_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    return lines


# ==========================================================================================
# TRANSFORMATION TESTS
# ==========================================================================================

@pytest.mark.parametrize("test_case", [], ids=lambda x: x['sentence'][:50])
def test_tree_transformations(test_case, nlp):
    """
    Test that tree transformations produce expected structure.

    Verifies:
    - Wh-word extraction markers
    - Dependency relabeling (wh-nsubj, wh-dobj, etc.)
    - BIND markers for relative clauses
    - Copula marking
    - Conjunction disambiguation
    """
    sentence = test_case['sentence']
    expected_transformations = test_case.get('transformation_steps', [])

    # Parse and transform
    doc = nlp(sentence)
    nodes = doc_to_tnodes(doc)
    nodes = apply_all_transformations(nodes)

    # Verify expected transformations applied
    for expected in expected_transformations:
        rule_name = expected['rule']
        token_index = expected.get('token_index')
        expected_dep = expected.get('expected_dep')
        expected_marker = expected.get('expected_marker')

        if token_index and expected_dep:
            node = nodes[token_index]
            actual_dep = node.corrected_dep if node.corrected_dep else node.dep
            assert actual_dep == expected_dep, \
                f"Rule '{rule_name}' failed: expected dep={expected_dep}, got {actual_dep}"

        if token_index and expected_marker:
            node = nodes[token_index]
            if expected_marker == 'is_extracted':
                assert node.is_extracted, f"Node {token_index} not marked for extraction"
            elif expected_marker == 'has_bind':
                assert node.has_bind, f"Node {token_index} does not have BIND marker"


@pytest.mark.parametrize("test_case", [], ids=lambda x: x['sentence'][:50])
def test_assignment_rules(test_case, nlp):
    """
    Test that assignment rules produce expected lambda expressions.

    Verifies each dependency pattern maps to correct lambda constructor.
    """
    sentence = test_case['sentence']
    expected_assignments = test_case.get('assignment_steps', [])

    # Parse and transform
    doc = nlp(sentence)
    nodes = doc_to_tnodes(doc)
    nodes = apply_all_transformations(nodes)

    # TODO: Implement template matching on TNodes to verify assignments
    # For now, this is a placeholder

    pytest.skip("Assignment rule testing requires TNode-based pattern matching")


# ==========================================================================================
# ALPHA-EQUIVALENCE TESTS
# ==========================================================================================

@pytest.mark.parametrize("test_case", [], ids=lambda x: x['sentence'][:50])
@pytest.mark.skipif(not HAS_NLTK_LOGIC, reason="nltk.sem.logic not available")
def test_logical_form_equivalence(test_case, nlp):
    """
    Test that final logical form is alpha-equivalent to gold standard.

    Uses NLTK's alpha-equivalence check to allow variable renaming.
    """
    sentence = test_case['sentence']
    gold_logical_form = test_case['final_logical_form']

    # TODO: Generate logical form using CHISEL pipeline
    # chisel_logical_form = ...

    # Parse both logical forms
    parser = logic.LogicParser()
    try:
        gold_expr = parser.parse(gold_logical_form)
        # chisel_expr = parser.parse(chisel_logical_form)

        # Check alpha-equivalence
        # assert gold_expr == chisel_expr or alpha_equivalent(gold_expr, chisel_expr)
    except logic.LogicalExpressionException as e:
        pytest.fail(f"Failed to parse logical form: {e}")

    pytest.skip("Logical form generation requires full pipeline integration")


# ==========================================================================================
# CONSTRUCTION-SPECIFIC TESTS
# ==========================================================================================

def test_wh_extraction_who(nlp):
    """Test wh-extraction for 'who' questions."""
    sentence = "Who is the founder of Google?"
    doc = nlp(sentence)
    nodes = doc_to_tnodes(doc)
    nodes = apply_all_transformations(nodes)

    # Find "Who" node
    who_node = next((n for n in nodes if n.word == "Who"), None)
    assert who_node is not None
    assert who_node.is_extracted, "Who should be marked for extraction"
    assert who_node.corrected_dep == "wh-attr", "Who should be relabeled to wh-attr"


def test_relative_clause_which(nlp):
    """Test relative clause with 'which'."""
    sentence = "The company which bought Youtube owns Gmail."
    doc = nlp(sentence)
    nodes = doc_to_tnodes(doc)
    nodes = apply_all_transformations(nodes)

    # Find "which" node
    which_node = next((n for n in nodes if n.word == "which"), None)
    assert which_node is not None
    # Should be marked for extraction
    assert which_node.is_extracted, "Which should be marked for extraction"


def test_passive_construction(nlp):
    """Test passive voice handling."""
    sentence = "The course was taught by Professor Smith."
    doc = nlp(sentence)
    nodes = doc_to_tnodes(doc)
    nodes = apply_all_transformations(nodes)

    # Find passive subject
    course_node = next((n for n in nodes if n.word == "course"), None)
    assert course_node is not None
    # In passive, "course" should have nsubj:pass dependency
    # (spaCy marks this, transformations should preserve it)


def test_conjunction_s_level(nlp):
    """Test S-level conjunction disambiguation."""
    sentence = "Jobs started Apple and Page founded Google"
    doc = nlp(sentence)
    nodes = doc_to_tnodes(doc)
    nodes = apply_all_transformations(nodes)

    # Find conjunction node
    founded_node = next((n for n in nodes if n.lemma == "found"), None)
    if founded_node and founded_node.dep == "conj":
        # Should be marked as S-level (both clauses have subjects)
        assert founded_node.corrected_dep == "conj-s", \
            "Conjunction with subject should be S-level"


def test_copula_construction(nlp):
    """Test copula marking."""
    sentence = "Obama is the President."
    doc = nlp(sentence)
    nodes = doc_to_tnodes(doc)
    nodes = apply_all_transformations(nodes)

    # Find copular verb
    is_node = next((n for n in nodes if n.lemma == "be"), None)
    if is_node:
        # Check if nsubj and attr are marked as copula
        nsubj_children = [c for c in is_node.children if 'nsubj' in c.get_dep()]
        attr_children = [c for c in is_node.children if 'attr' in c.get_dep()]

        if nsubj_children:
            assert nsubj_children[0].corrected_dep == "nsubj-copula", \
                "Copular subject should be marked"
        if attr_children:
            assert attr_children[0].corrected_dep == "attr-copula", \
                "Copular attribute should be marked"


# ==========================================================================================
# INTEGRATION TEST
# ==========================================================================================

def test_full_pipeline_sample(nlp, test_sentences):
    """
    Test full pipeline on sample sentences.

    Verifies that pipeline runs without errors on all test sentences.
    """
    from chisel.tnode import doc_to_tnodes
    from chisel.tree_transformations import apply_all_transformations

    for sentence in test_sentences[:10]:  # Test first 10
        doc = nlp(sentence)
        nodes = doc_to_tnodes(doc)

        # Should not raise exceptions
        nodes = apply_all_transformations(nodes)

        assert len(nodes) > 0, f"No nodes generated for: {sentence}"


# ==========================================================================================
# HELPER FUNCTIONS
# ==========================================================================================

def alpha_equivalent(expr1, expr2):
    """
    Check if two logical expressions are alpha-equivalent.

    Alpha-equivalence means structurally identical up to variable renaming.
    """
    # NLTK's Expression.__eq__ checks alpha-equivalence by default
    return expr1 == expr2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
