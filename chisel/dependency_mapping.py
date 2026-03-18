"""
Stanford Dependencies → Universal Dependencies Mapping

deplambda uses Stanford Dependencies (v3.5), spaCy uses Universal Dependencies (v2).
This module provides the mapping between the two schemes.

References:
- UD documentation: https://universaldependencies.org/
- Stanford→UD conversion: https://universaldependencies.org/docsv1/en/dep/index.html
- Reddy et al. (2016), deplambda paper: https://aclanthology.org/Q16-1010/
"""

# ==========================================================================================
# CORE ARGUMENT RELATIONS
# ==========================================================================================

CORE_ARGUMENTS = {
    # Subject relations
    'nsubj': 'nsubj',           # Nominal subject (unchanged)
    'nsubjpass': 'nsubj:pass',  # Passive nominal subject → nsubj with aux:pass

    # Object relations
    'dobj': 'obj',              # Direct object → obj
    'iobj': 'iobj',             # Indirect object (unchanged)

    # Clausal arguments
    'ccomp': 'ccomp',           # Clausal complement (unchanged)
    'xcomp': 'xcomp',           # Open clausal complement (unchanged)
}

# ==========================================================================================
# NOMINAL MODIFIERS
# ==========================================================================================

NOMINAL_MODIFIERS = {
    # Adjectival
    'amod': 'amod',             # Adjectival modifier (unchanged)
    'acomp': 'acomp',           # Adjectival complement (unchanged in UD v1, deprecated in v2)

    # Nominal
    'nn': 'compound',           # Noun compound modifier → compound
    'appos': 'appos',           # Appositional modifier (unchanged)

    # Determiners and quantifiers
    'det': 'det',               # Determiner (unchanged)
    'predet': 'det:predet',     # Predeterminer → det:predet

    # Numeric
    'num': 'nummod',            # Numeric modifier → nummod
    'number': 'nummod',         # Number compound → nummod

    # Possessive
    'poss': 'nmod:poss',        # Possessive modifier → nmod:poss
    'possessive': 'case',       # Possessive marker (e.g., 's) → case
}

# ==========================================================================================
# PREPOSITIONAL PHRASES
# ==========================================================================================

PREPOSITIONAL = {
    'prep': 'case',             # Preposition marker → case
    'pobj': 'obl',              # Object of preposition → oblique
    'pcomp': 'obl',             # Prepositional complement → oblique
}

# ==========================================================================================
# CLAUSAL MODIFIERS
# ==========================================================================================

CLAUSAL_MODIFIERS = {
    # Relative clauses
    'rcmod': 'acl:relcl',       # Relative clause modifier → acl:relcl
    'relcl': 'acl:relcl',       # Relative clause (unchanged in UD)

    # Participial/adjectival clauses
    'partmod': 'advcl',         # Participial modifier → advcl
    'infmod': 'advcl',          # Infinitival modifier → advcl
    'vmod': 'acl',              # Verbal modifier → acl (adjectival clause)

    # Adverbial clauses
    'advcl': 'advcl',           # Adverbial clause (unchanged)
    'purpcl': 'advcl',          # Purpose clause → advcl
}

# ==========================================================================================
# VERB MODIFIERS
# ==========================================================================================

VERB_MODIFIERS = {
    # Adverbial
    'advmod': 'advmod',         # Adverbial modifier (unchanged)
    'npadvmod': 'obl:npmod',    # Noun phrase adverbial modifier → obl:npmod
    'tmod': 'obl:tmod',         # Temporal modifier → obl:tmod

    # Negation
    'neg': 'advmod',            # Negation → advmod (with Polarity=Neg feature)

    # Auxiliary
    'aux': 'aux',               # Auxiliary (unchanged)
    'auxpass': 'aux:pass',      # Passive auxiliary → aux:pass

    # Modal
    'modal': 'aux',             # Modal → aux (UD doesn't distinguish)
}

# ==========================================================================================
# COORDINATION
# ==========================================================================================

COORDINATION = {
    'conj': 'conj',             # Conjunct (unchanged)
    'cc': 'cc',                 # Coordinating conjunction (unchanged)
    'preconj': 'cc:preconj',    # Preconjunct → cc:preconj
}

# ==========================================================================================
# OTHER RELATIONS
# ==========================================================================================

OTHER = {
    # Markers
    'mark': 'mark',             # Subordinating conjunction marker (unchanged)
    'complm': 'mark',           # Complementizer → mark

    # Particles
    'prt': 'compound:prt',      # Particle → compound:prt

    # Multi-word expressions
    'mwe': 'fixed',             # Multi-word expression → fixed

    # Expletive
    'expl': 'expl',             # Expletive (unchanged)

    # Discourse
    'discourse': 'discourse',   # Discourse element (unchanged)

    # Default
    'dep': 'dep',               # Unspecified dependency (unchanged)
    'rel': 'ref',               # Relative (unclear role) → ref (referent)
}

# ==========================================================================================
# SPECIAL DEPLAMBDA LABELS
# ==========================================================================================

DEPLAMBDA_SPECIAL = {
    # Wh-word extraction (deplambda-specific)
    'wh-nsubj': 'nsubj',        # Wh-subject → nsubj (extracted)
    'wh-nsubjpass': 'nsubj:pass',  # Wh-passive subject → nsubj:pass
    'wh-dobj': 'obj',           # Wh-object → obj (extracted)
    'wh-attr': 'attr',          # Wh-attribute (no direct UD equivalent, keep as-is)
    'wh-advmod': 'advmod',      # Wh-adverbial → advmod (extracted)
    'wh-tmod': 'obl:tmod',      # Wh-temporal → obl:tmod (extracted)
    'wh-pobj': 'obl',           # Wh-prepositional object → obl (extracted)
    'wh-dep': 'dep',            # Wh-unspecified → dep

    # Control and extraction markers (deplambda tree operations)
    'BIND': 'BIND',             # Bind marker (keep as-is, not a real dependency)
    'TRACE': 'TRACE',           # Trace marker (keep as-is, not a real dependency)
    'extract': 'extract',       # Extraction marker (keep as-is, not a real dependency)

    # Copula
    'nsubj-copula': 'nsubj',    # Copular subject → nsubj
    'attr-copula': 'attr',      # Copular attribute (no UD equivalent, keep as-is)

    # Conjunction types
    'conj-s': 'conj',           # Sentence-level conjunction → conj
    'conj-vp': 'conj',          # VP-level conjunction → conj
}

# ==========================================================================================
# COMPLETE MAPPING
# ==========================================================================================

# Combine all mappings
STANFORD_TO_UD = {}
STANFORD_TO_UD.update(CORE_ARGUMENTS)
STANFORD_TO_UD.update(NOMINAL_MODIFIERS)
STANFORD_TO_UD.update(PREPOSITIONAL)
STANFORD_TO_UD.update(CLAUSAL_MODIFIERS)
STANFORD_TO_UD.update(VERB_MODIFIERS)
STANFORD_TO_UD.update(COORDINATION)
STANFORD_TO_UD.update(OTHER)
STANFORD_TO_UD.update(DEPLAMBDA_SPECIAL)


def stanford_to_ud(stanford_label: str) -> str:
    """
    Convert Stanford dependency label to Universal Dependencies label.

    Args:
        stanford_label: Stanford Dependencies label (e.g., 'dobj', 'nsubjpass')

    Returns:
        UD label (e.g., 'obj', 'nsubj:pass')
    """
    # Remove 'l-' prefix if present (deplambda format)
    if stanford_label.startswith('l-'):
        stanford_label = stanford_label[2:]

    return STANFORD_TO_UD.get(stanford_label, stanford_label)


def is_subject_relation(label: str) -> bool:
    """Check if label is a subject relation."""
    return label in ['nsubj', 'nsubj:pass', 'nsubjpass', 'wh-nsubj', 'wh-nsubjpass']


def is_object_relation(label: str) -> bool:
    """Check if label is an object relation."""
    return label in ['obj', 'dobj', 'iobj', 'wh-dobj']


def is_verbal_argument(label: str) -> bool:
    """Check if label is a core verbal argument."""
    return is_subject_relation(label) or is_object_relation(label)


def is_prepositional(label: str) -> bool:
    """Check if label is part of prepositional phrase."""
    return label in ['case', 'prep', 'obl', 'pobj', 'pcomp']


def is_clausal_modifier(label: str) -> bool:
    """Check if label is a clausal modifier."""
    return label in ['acl', 'acl:relcl', 'advcl', 'rcmod', 'relcl', 'partmod', 'infmod', 'vmod']


def is_coordination(label: str) -> bool:
    """Check if label is part of coordination."""
    return label in ['conj', 'cc']
