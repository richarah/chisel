"""
Tree Transformation Rules

Implements deplambda's transformation-rules.pb.txt on TNode structure.
These rules restructure the dependency tree before lambda assignment:
- Fix parser errors
- Mark wh-word extraction
- Add BIND nodes for relative clause extraction
- Distinguish copula constructions
- Disambiguate conjunction types (S vs VP level)

Based on Reddy et al. (2016, TACL): https://aclanthology.org/Q16-1010/
Original rules: https://github.com/sivareddyg/deplambda/blob/master/transformation-rules.pb.txt
"""

from typing import List, Set, Dict, Optional
from .tnode import TNode, find_node_by_index, find_nodes_by_dep, find_nodes_by_lemma


# ==========================================================================================
# PARSER BUG FIXES (Priority: 1)
# ==========================================================================================

def fix_preposition_labeled_dep(nodes: List[TNode]) -> List[TNode]:
    """
    Fix: Preposition incorrectly labeled as 'dep'.

    Original tregex: /^l-(?:dep|rel)$/=relation < t-IN $ /^t-(?:IN|N|V|W?RB[RS]).*$/
    Action: CHANGE_LABEL to 'l-prep' (UD: 'case')
    """
    for node in nodes:
        if node.dep in ['dep', 'rel'] and node.pos == 'ADP':
            # Get parent node
            parent = find_node_by_index(nodes, node.head_index)
            if parent:
                # Check if sibling is noun/verb/adverb
                siblings = [c for c in parent.children if c != node]
                if any(s.pos in ['NOUN', 'VERB', 'ADV', 'ADJ'] for s in siblings):
                    # Relabel as case
                    node.corrected_dep = 'case'
    return nodes


def fix_duplicate_nsubj(nodes: List[TNode]) -> List[TNode]:
    """
    Fix: Two nsubj children (parser error).

    Original tregex: l-nsubj=relation $ l-nsubj
    Action: CHANGE_LABEL first to 'l-attr'
    """
    for node in nodes:
        nsubj_children = node.get_children_by_dep('nsubj')
        if len(nsubj_children) > 1:
            # Change first nsubj to attr
            nsubj_children[0].corrected_dep = 'attr'
    return nodes


# ==========================================================================================
# WH-WORD EXTRACTION MARKING (Priority: 1)
# ==========================================================================================

def mark_wh_extraction(nodes: List[TNode]) -> List[TNode]:
    """
    Mark nodes that should be extracted (modified by wh-words).

    Original rules:
    - tregex: /^l-(?:nsubj|nsubjpass|attr|dobj|pobj|tmod|dep)$/=target
             [< /^t-W.*$/ | < (/l-.*$/ < /^t-W.*$/) | ...]
    - Action: ADD_CHILD l-wh-marker

    UD equivalent: Mark extracted arguments with special flag.
    """
    # Find all wh-words (PRON with tag starting with W)
    wh_words = [n for n in nodes if n.pos == 'PRON' and n.tag.startswith('W')]

    # Mark wh-words themselves if in extraction position
    for wh_word in wh_words:
        if wh_word.dep in ['nsubj', 'nsubj:pass', 'attr', 'obj', 'obl', 'obl:tmod', 'dep', 'pobj']:
            wh_word.is_extracted = True

    return nodes


def relabel_wh_dependencies(nodes: List[TNode]) -> List[TNode]:
    """
    Relabel dependencies for extracted wh-arguments.

    Original rules:
    - l-nsubj with wh-marker → l-wh-nsubj
    - l-dobj with wh-marker → l-wh-dobj
    - l-attr with wh-marker → l-wh-attr
    - etc.
    """
    for node in nodes:
        if node.is_extracted:
            # Relabel dependency
            if node.dep == 'nsubj':
                node.corrected_dep = 'wh-nsubj'
            elif node.dep in ['nsubj:pass', 'nsubjpass']:
                node.corrected_dep = 'wh-nsubjpass'
            elif node.dep == 'obj':
                node.corrected_dep = 'wh-dobj'
            elif node.dep == 'attr':
                node.corrected_dep = 'wh-attr'
            elif node.dep == 'advmod':
                node.corrected_dep = 'wh-advmod'
            elif node.dep in ['obl', 'obl:tmod']:
                node.corrected_dep = 'wh-tmod'
            elif node.dep == 'dep':
                node.corrected_dep = 'wh-dep'

    return nodes


# ==========================================================================================
# CONJUNCTIONS (Priority: 1)
# ==========================================================================================

def disambiguate_conjunction_type(nodes: List[TNode]) -> List[TNode]:
    """
    Distinguish S-level vs VP-level conjunction.

    Original rules:
    - S conjunction: l-conj $ /^t-V.*$/ [< (/^t-V.*$/ $ l-nsubj) | < (t-VBN $ l-nsubjpass)]
      Example: Jobs "started" Apple and Page "founded" Google
    - VP conjunction: l-conj $ /^t-V.*$/ [< (/^t-V.*$/ $ l-dobj) | ...]
      Example: Jobs "started" Apple and "founded" Google
    """
    for node in nodes:
        if node.dep == 'conj':
            parent = find_node_by_index(nodes, node.head_index)
            if parent and parent.pos == 'VERB':
                # Check if conjunct has its own subject (S-level)
                if node.has_dep('nsubj') or node.has_dep('nsubj:pass'):
                    node.corrected_dep = 'conj-s'  # Sentence-level conjunction
                # Check if conjunct has object (VP-level)
                elif node.has_dep('obj') or node.has_dep('obl'):
                    node.corrected_dep = 'conj-vp'  # VP-level conjunction

    return nodes


# ==========================================================================================
# CONTROL (Priority: 1)
# ==========================================================================================

def add_control_subjects(nodes: List[TNode]) -> List[TNode]:
    """
    Add implicit subjects to xcomp based on control rules.

    Original rules:
    - Subject control: l-xcomp !< nsubj > (/^l-.*$/=verb [!< l-dobj] < /^l-nsubj.*$/=nsubj)
      Example: "John wants to sleep" → "John" controls subject of "sleep"
    - Object control: l-xcomp !< nsubj > (/^l-.*$/=verb < l-dobj=dobj)
      Example: "John asked Mary to leave" → "Mary" controls subject of "leave"

    Action: ADD_CHILD l-nsubj, ADD_CHILD l-extract, ADD_CHILD l-TRACE
    """
    for node in nodes:
        if node.dep == 'xcomp' and not node.has_dep('nsubj'):
            verb = find_node_by_index(nodes, node.head_index)
            if not verb:
                continue

            # Subject control (default for most cases)
            if verb.has_dep('nsubj') and not verb.has_dep('obj'):
                nsubj_children = verb.get_children_by_dep('nsubj')
                if nsubj_children:
                    node.controller_index = nsubj_children[0].index

            # Object control (verbs like "ask", "tell", "force")
            elif verb.has_dep('obj'):
                obj_children = verb.get_children_by_dep('obj')
                if obj_children:
                    node.controller_index = obj_children[0].index

    return nodes


# ==========================================================================================
# EXTRACTION (Priority: 1)
# ==========================================================================================

def add_extraction_markers(nodes: List[TNode]) -> List[TNode]:
    """
    Add BIND nodes for relative clause extraction.

    Original rules:
    - Pobj extraction: /^l-.*$/=target < /^t-(?:N|V).*$/ !<< /^l-(?:rcmod|partmod|xcomp).*$/
                      [<< (l-prep=prep !< /^l-(?:pobj|pcomp)$/) | ...]
      Example: "The country Darwin belongs to" → bind "country" for extraction
    - Nsubj extraction: /^l-.*$/=target < /^t-V.*$/ < /^l-wh-nsubj$/
      Example: "The company which bought Youtube" → bind "company"
    - Dobj extraction: Similar pattern

    Action: ADD_CHILD l-BIND
    """
    for node in nodes:
        # Pobj extraction: Stranded prepositions
        if node.pos in ['NOUN', 'VERB']:
            # Check for case without obl
            for child in node.children:
                if child.get_dep() == 'case' and not node.has_dep('obl'):
                    # Mark for extraction
                    node.has_bind = True

        # Wh-nsubj extraction
        if node.pos == 'VERB' and node.has_dep('wh-nsubj'):
            node.has_bind = True

        # Wh-dobj extraction
        if node.pos == 'VERB' and node.has_dep('wh-dobj'):
            node.has_bind = True

    return nodes


# ==========================================================================================
# COPULA (Priority: 1)
# ==========================================================================================

def mark_copula_constructions(nodes: List[TNode]) -> List[TNode]:
    """
    Mark copular constructions.

    Original rule:
    - tregex: /^l-(?:nsubj|attr)$/=first $ /^l-attr$/=second !$ l-prep
    - Action: CHANGE_LABEL first to 'l-nsubj-copula', second to 'l-attr-copula'

    Example: "Obama is the President" → nsubj-copula(is, Obama), attr-copula(is, President)
    """
    for node in nodes:
        # Find copular verbs (be, become, seem, etc.)
        if node.lemma in ['be', 'become', 'seem', 'appear', 'remain'] and node.pos == 'VERB':
            nsubj_children = node.get_children_by_dep('nsubj')
            attr_children = node.get_children_by_dep('attr')

            if nsubj_children and attr_children:
                # Mark as copular construction
                nsubj_children[0].corrected_dep = 'nsubj-copula'
                attr_children[0].corrected_dep = 'attr-copula'

    return nodes


# ==========================================================================================
# MAIN TRANSFORMATION PIPELINE
# ==========================================================================================

def apply_all_transformations(nodes: List[TNode]) -> List[TNode]:
    """
    Apply all tree transformations in priority order.

    Args:
        nodes: List of TNodes

    Returns:
        Transformed TNodes with restructured dependency tree
    """
    # Priority 1: Parser bug fixes
    nodes = fix_preposition_labeled_dep(nodes)
    nodes = fix_duplicate_nsubj(nodes)

    # Priority 1: Conjunction disambiguation
    nodes = disambiguate_conjunction_type(nodes)

    # Priority 1: Wh-word extraction
    nodes = mark_wh_extraction(nodes)
    nodes = relabel_wh_dependencies(nodes)

    # Priority 1: Control
    nodes = add_control_subjects(nodes)

    # Priority 1: Extraction markers
    nodes = add_extraction_markers(nodes)

    # Priority 1: Copula
    nodes = mark_copula_constructions(nodes)

    return nodes


# ==========================================================================================
# TESTING
# ==========================================================================================

if __name__ == "__main__":
    import spacy
    from .tnode import doc_to_tnodes, tnodes_to_tree_str

    nlp = spacy.load("en_core_web_sm")

    # Test sentences
    test_sentences = [
        "Who is the founder of Google?",
        "The company which bought Youtube owns Gmail",
        "Jobs started Apple and Page founded Google",
        "What country does Darwin belong to?",
        "John wants to sleep",
    ]

    for sent in test_sentences:
        doc = nlp(sent)
        nodes = doc_to_tnodes(doc)

        print(f"\n{sent}")
        print("  Before:")
        print(tnodes_to_tree_str(nodes))

        nodes = apply_all_transformations(nodes)
        print("  After:")
        print(tnodes_to_tree_str(nodes))
