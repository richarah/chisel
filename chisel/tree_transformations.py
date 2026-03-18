"""
Tree Transformation Rules

Implements deplambda's transformation-rules.pb.txt in spaCy.
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
from spacy.tokens import Doc, Token
import spacy


# ==========================================================================================
# HELPER FUNCTIONS
# ==========================================================================================

def has_dep(token: Token, dep: str) -> bool:
    """Check if token has a child with given dependency."""
    return any(child.dep_ == dep for child in token.children)


def get_children_by_dep(token: Token, dep: str) -> List[Token]:
    """Get all children with given dependency."""
    return [child for child in token.children if child.dep_ == dep]


def has_pos(token: Token, pos_pattern: str) -> bool:
    """Check if token matches POS pattern."""
    if pos_pattern.startswith("^") and pos_pattern.endswith("$"):
        pos_pattern = pos_pattern[1:-1]
    return token.pos_ == pos_pattern or token.tag_.startswith(pos_pattern)


# ==========================================================================================
# PARSER BUG FIXES (Priority: 1)
# ==========================================================================================

def fix_preposition_labeled_dep(doc: Doc) -> Doc:
    """
    Fix: Preposition incorrectly labeled as 'dep'.

    Original tregex: /^l-(?:dep|rel)$/=relation < t-IN $ /^t-(?:IN|N|V|W?RB[RS]).*$/
    Action: CHANGE_LABEL to 'l-prep'

    Note: spaCy Doc dependencies are read-only. We store corrections in custom attributes.
    """
    if not Token.has_extension('corrected_dep'):
        Token.set_extension('corrected_dep', default=None)

    for token in doc:
        if token.dep_ in ['dep', 'rel'] and token.pos_ == 'ADP':
            # Check if sibling is noun/verb/adverb
            siblings = [t for t in token.head.children if t != token]
            if any(s.pos_ in ['NOUN', 'VERB', 'ADV', 'ADJ'] for s in siblings):
                # Store corrected label (can't modify doc directly)
                token._.corrected_dep = 'case'
    return doc


def fix_duplicate_nsubj(doc: Doc) -> Doc:
    """
    Fix: Two nsubj children (parser error).

    Original tregex: l-nsubj=relation $ l-nsubj
    Action: CHANGE_LABEL first to 'l-attr'
    """
    for token in doc:
        nsubj_children = get_children_by_dep(token, 'nsubj')
        if len(nsubj_children) > 1:
            # Change first nsubj to attr
            nsubj_children[0].dep_ = 'attr'
    return doc


# ==========================================================================================
# WH-WORD EXTRACTION MARKING (Priority: 1)
# ==========================================================================================

def mark_wh_extraction(doc: Doc) -> Doc:
    """
    Mark nodes that should be extracted (modified by wh-words).

    Original rules:
    - tregex: /^l-(?:nsubj|nsubjpass|attr|dobj|pobj|tmod|dep)$/=target
             [< /^t-W.*$/ | < (/l-.*$/ < /^t-W.*$/) | ...]
    - Action: ADD_CHILD l-wh-marker

    UD equivalent: Mark extracted arguments with special flag.
    """
    # Find all wh-words (PRON with PronType=Int or PronType=Rel)
    wh_words = []
    for token in doc:
        if token.pos_ == 'PRON' and token.tag_.startswith('W'):
            wh_words.append(token)

    # Mark their heads for extraction
    for wh_word in wh_words:
        head = wh_word.head
        if head.dep_ in ['nsubj', 'nsubj:pass', 'attr', 'obj', 'obl', 'obl:tmod', 'dep']:
            # Add extraction marker (custom attribute)
            if not Token.has_extension('is_extracted'):
                Token.set_extension('is_extracted', default=False)
            head._.is_extracted = True

    return doc


def relabel_wh_dependencies(doc: Doc) -> Doc:
    """
    Relabel dependencies for extracted wh-arguments.

    Original rules:
    - l-nsubj with wh-marker → l-wh-nsubj
    - l-dobj with wh-marker → l-wh-dobj
    - l-attr with wh-marker → l-wh-attr
    - etc.
    """
    if not Token.has_extension('is_extracted'):
        return doc

    for token in doc:
        if token._.is_extracted:
            # Relabel dependency
            if token.dep_ == 'nsubj':
                token.dep_ = 'wh-nsubj'
            elif token.dep_ == 'nsubj:pass':
                token.dep_ = 'wh-nsubjpass'
            elif token.dep_ == 'obj':
                token.dep_ = 'wh-dobj'
            elif token.dep_ == 'attr':
                token.dep_ = 'wh-attr'
            elif token.dep_ == 'advmod':
                token.dep_ = 'wh-advmod'
            elif token.dep_ in ['obl', 'obl:tmod']:
                token.dep_ = 'wh-tmod'
            elif token.dep_ == 'dep':
                token.dep_ = 'wh-dep'

    return doc


# ==========================================================================================
# CONJUNCTIONS (Priority: 1)
# ==========================================================================================

def disambiguate_conjunction_type(doc: Doc) -> Doc:
    """
    Distinguish S-level vs VP-level conjunction.

    Original rules:
    - S conjunction: l-conj $ /^t-V.*$/ [< (/^t-V.*$/ $ l-nsubj) | < (t-VBN $ l-nsubjpass)]
      Example: Jobs "started" Apple and Page "founded" Google
    - VP conjunction: l-conj $ /^t-V.*$/ [< (/^t-V.*$/ $ l-dobj) | ...]
      Example: Jobs "started" Apple and "founded" Google
    """
    for token in doc:
        if token.dep_ == 'conj' and token.head.pos_ == 'VERB':
            # Check if conjunct has its own subject (S-level)
            if has_dep(token, 'nsubj') or has_dep(token, 'nsubj:pass'):
                token.dep_ = 'conj-s'  # Sentence-level conjunction
            # Check if conjunct has object (VP-level)
            elif has_dep(token, 'obj') or has_dep(token, 'obl'):
                token.dep_ = 'conj-vp'  # VP-level conjunction

    return doc


# ==========================================================================================
# CONTROL (Priority: 1)
# ==========================================================================================

def add_control_subjects(doc: Doc) -> Doc:
    """
    Add implicit subjects to xcomp based on control rules.

    Original rules:
    - Subject control: l-xcomp !< nsubj > (/^l-.*$/=verb [!< l-dobj] < /^l-nsubj.*$/=nsubj)
      Example: "John wants to sleep" → "John" controls subject of "sleep"
    - Object control: l-xcomp !< nsubj > (/^l-.*$/=verb < l-dobj=dobj)
      Example: "John asked Mary to leave" → "Mary" controls subject of "leave"

    Action: ADD_CHILD l-nsubj, ADD_CHILD l-extract, ADD_CHILD l-TRACE

    Note: spaCy already has basic xcomp relations, but doesn't mark control explicitly.
    We add custom attributes to track the controller.
    """
    if not Token.has_extension('controller'):
        Token.set_extension('controller', default=None)

    for token in doc:
        if token.dep_ == 'xcomp' and not has_dep(token, 'nsubj'):
            verb = token.head

            # Subject control (default for most cases)
            if has_dep(verb, 'nsubj') and not has_dep(verb, 'obj'):
                nsubj = get_children_by_dep(verb, 'nsubj')[0]
                token._.controller = nsubj

            # Object control (verbs like "ask", "tell", "force")
            elif has_dep(verb, 'obj'):
                obj = get_children_by_dep(verb, 'obj')[0]
                token._.controller = obj

    return doc


# ==========================================================================================
# EXTRACTION (Priority: 1)
# ==========================================================================================

def add_extraction_markers(doc: Doc) -> Doc:
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
    if not Token.has_extension('has_bind'):
        Token.set_extension('has_bind', default=False)

    for token in doc:
        # Pobj extraction: Stranded prepositions
        if token.pos_ in ['NOUN', 'VERB']:
            # Check for prep without pobj
            for child in token.children:
                if child.dep_ == 'case' and not has_dep(child, 'obl'):
                    # Mark for extraction
                    token._.has_bind = True

        # Wh-nsubj extraction
        if token.pos_ == 'VERB' and has_dep(token, 'wh-nsubj'):
            token._.has_bind = True

        # Wh-dobj extraction
        if token.pos_ == 'VERB' and has_dep(token, 'wh-dobj'):
            token._.has_bind = True

    return doc


# ==========================================================================================
# COPULA (Priority: 1)
# ==========================================================================================

def mark_copula_constructions(doc: Doc) -> Doc:
    """
    Mark copular constructions.

    Original rule:
    - tregex: /^l-(?:nsubj|attr)$/=first $ /^l-attr$/=second !$ l-prep
    - Action: CHANGE_LABEL first to 'l-nsubj-copula', second to 'l-attr-copula'

    Example: "Obama is the President" → nsubj-copula(is, Obama), attr-copula(is, President)
    """
    for token in doc:
        # Find copular verbs (be, become, seem, etc.)
        if token.lemma_ in ['be', 'become', 'seem', 'appear', 'remain'] and token.pos_ == 'VERB':
            nsubj_children = get_children_by_dep(token, 'nsubj')
            attr_children = get_children_by_dep(token, 'attr')

            if nsubj_children and attr_children:
                # Mark as copular construction
                nsubj_children[0].dep_ = 'nsubj-copula'
                attr_children[0].dep_ = 'attr-copula'

    return doc


# ==========================================================================================
# MAIN TRANSFORMATION PIPELINE
# ==========================================================================================

def apply_all_transformations(doc: Doc) -> Doc:
    """
    Apply all tree transformations in priority order.

    Args:
        doc: spaCy Doc object

    Returns:
        Transformed Doc with restructured dependency tree
    """
    # Priority 1: Parser bug fixes
    doc = fix_preposition_labeled_dep(doc)
    doc = fix_duplicate_nsubj(doc)

    # Priority 1: Conjunction disambiguation
    doc = disambiguate_conjunction_type(doc)

    # Priority 1: Wh-word extraction
    doc = mark_wh_extraction(doc)
    doc = relabel_wh_dependencies(doc)

    # Priority 1: Control
    doc = add_control_subjects(doc)

    # Priority 1: Extraction markers
    doc = add_extraction_markers(doc)

    # Priority 1: Copula
    doc = mark_copula_constructions(doc)

    return doc


# ==========================================================================================
# TESTING
# ==========================================================================================

if __name__ == "__main__":
    import spacy

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
        print(f"\n{sent}")
        print("  Before:")
        for token in doc:
            print(f"    {token.text} --{token.dep_}-> {token.head.text}")

        doc = apply_all_transformations(doc)
        print("  After:")
        for token in doc:
            print(f"    {token.text} --{token.dep_}-> {token.head.text}")
