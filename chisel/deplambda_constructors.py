"""
deplambda Constructor Functions

SQL-grounded implementations of linguistic rules from Reddy et al. (2016, TACL).
These constructors compose lambda expressions bottom-up from dependency patterns.

Unlike deplambda's full event semantics, these focus on SQL operations:
- Joins (FROM/JOIN clauses)
- Filters (WHERE clauses)
- Projections (SELECT clauses)
"""

from typing import Optional
from nltk.sem.logic import Expression, ApplicationExpression, LambdaExpression, Variable

from . import ir_vocabulary as vocab


# ==========================================================================================
# VERBAL ARGUMENTS (nsubj, obj, iobj)
# ==========================================================================================

def nsubj_relation(verb_expr: Expression, subject_expr: Expression) -> Expression:
    """
    Subject of verb.

    deplambda: (LAMBDA F G E : (EXISTS X : (F E) (G X) (p:EVENT.ENTITY:arg_1 E X)))
    SQL: Filters verb predicate by subject predicate

    Example:
        nsubj_relation(λx.works(x), λx.employee(x))
        = λx.employee(x) & works(x)  [implicit join on x]
    """
    # For SQL, nsubj typically means "filter by subject"
    # Return composition: subject & verb
    return vocab.filter_(subject_expr, verb_expr)


def obj_relation(verb_expr: Expression, object_expr: Expression) -> Expression:
    """
    Direct object of verb.

    deplambda: (LAMBDA F G E : (EXISTS X : (F E) (G X) (p:EVENT.ENTITY:arg_2 E X)))
    SQL: Filters verb predicate by object predicate

    Example:
        obj_relation(λx.manages(x), λx.department(x))
        = λx.department(x) & manages(x)
    """
    return vocab.filter_(verb_expr, object_expr)


def iobj_relation(verb_expr: Expression, iobject_expr: Expression) -> Expression:
    """
    Indirect object of verb.

    deplambda: (LAMBDA F G E : (EXISTS X : (F E) (G X) (p:EVENT.ENTITY:arg_3 E X)))
    SQL: Similar to obj_relation
    """
    return vocab.filter_(verb_expr, iobject_expr)


# ==========================================================================================
# PREPOSITIONAL PHRASES
# ==========================================================================================

def prep_relation(head_expr: Expression, prep_expr: Expression, pobj_expr: Expression) -> Expression:
    """
    Prepositional phrase attachment.

    deplambda: (LAMBDA F G E : (EXISTS P : (F E) (G P) (p:PPMOD E P)))
    SQL: Join head with prepositional object

    Example:
        prep_relation(λx.employee(x), "in", λx.department(x))
        = λx.employee(x) & department(x)  [join via FK]
    """
    # For SQL, prepositions indicate joins
    # Combine head with pobj
    return vocab.filter_(head_expr, pobj_expr)


# ==========================================================================================
# CONTROL AND COMPLEMENTS
# ==========================================================================================

def xcomp_relation(main_verb: Expression, complement_verb: Expression) -> Expression:
    """
    Open clausal complement (infinitive).

    deplambda: (LAMBDA F G E1 : (EXISTS E2 : (F E1) (G E2) (p:EVENT.EVENT:arg_ctrl E1 E2)))
    SQL: Compose predicates

    Example:
        xcomp_relation(λx.wants(x), λx.hire(x))
        = λx.wants(x) & hire(x)
    """
    return vocab.filter_(main_verb, complement_verb)


def ccomp_relation(main_verb: Expression, complement_clause: Expression) -> Expression:
    """
    Clausal complement (that-clause).

    deplambda: Similar to xcomp
    SQL: Compose predicates or EXISTS subquery
    """
    return vocab.filter_(main_verb, complement_clause)


# ==========================================================================================
# NOUN MODIFICATION
# ==========================================================================================

def amod_relation(noun_expr: Expression, adj_expr: Optional[Expression] = None) -> Expression:
    """
    Adjectival modifier.

    deplambda: (LAMBDA F G X : (F X) (G X))  [both predicates apply to same entity]
    SQL: Add filter condition

    Example:
        amod_relation(λx.employee(x), "senior")
        = λx.employee(x) & senior(x)
    """
    if adj_expr is None:
        return noun_expr
    # For SQL, adjectives become WHERE filters
    # For now, just return noun (adjectives need schema grounding)
    return noun_expr


def compound_relation(head_noun: Expression, modifier_noun: Expression) -> Expression:
    """
    Noun compound (e.g., "employee salary").

    deplambda: (LAMBDA F G X : (EXISTS Y : (F X) (G Y) (p:EVENT.ENTITY:q_arg_2 X Y)))
    SQL: Join two tables or filter

    Example:
        compound_relation(λx.salary(x), λx.employee(x))
        = λx.salary(x) & employee(x)  [join via FK]
    """
    return vocab.filter_(modifier_noun, head_noun)


def det_relation(noun_expr: Expression, det_str: Optional[str] = None) -> Expression:
    """
    Determiner (the, a, an).

    deplambda: (LAMBDA X : (p:UNIQUE X))  [for "the"]
    SQL: Determiners don't affect SQL directly (no uniqueness constraints)

    Example:
        det_relation(λx.employee(x), "the")
        = λx.employee(x)  [no change]
    """
    # Determiners don't affect SQL queries
    return noun_expr


def poss_relation(head_noun: Expression, possessor: Expression) -> Expression:
    """
    Possessive (e.g., "John's department").

    deplambda: (LAMBDA F G X : (EXISTS Y : (F X) (G Y) (p:EVENT.ENTITY:s_arg_2 X Y)))
    SQL: Join via FK

    Example:
        poss_relation(λx.department(x), λx.john(x))
        = λx.department(x) & john(x)  [join via manager FK]
    """
    return vocab.filter_(head_noun, possessor)


def nummod_relation(noun_expr: Expression, number: Optional[str] = None) -> Expression:
    """
    Numeric modifier.

    deplambda: (LAMBDA F G X : (EXISTS Y : (F X) (G Y) (p:EVENT.ENTITY:q_arg_2 X Y)))
    SQL: Numeric filter (e.g., COUNT = number)

    Example:
        nummod_relation(λx.employee(x), "5")
        = λx.employee(x) & COUNT(x) = 5  [GROUP BY with HAVING]
    """
    # For now, just return noun (numbers need context)
    return noun_expr


# ==========================================================================================
# VERB MODIFICATION
# ==========================================================================================

def advmod_relation(verb_expr: Expression, adverb: Optional[str] = None) -> Expression:
    """
    Adverbial modifier.

    deplambda: (LAMBDA E : (p:EVENTMOD:$adverb$ E))
    SQL: Adverbs typically don't affect SQL structure

    Example:
        advmod_relation(λx.works(x), "quickly")
        = λx.works(x)  [no change]
    """
    # Adverbs don't typically affect SQL
    return verb_expr


def negation(verb_expr: Expression) -> Expression:
    """
    Negation (not, n't).

    deplambda: (LAMBDA E : (p:NEGATION E))
    SQL: NOT EXISTS or negative filter

    Example:
        negation(λx.works(x))
        = NOT EXISTS (λx.works(x))
    """
    return vocab.not_exists(verb_expr)


def aux_relation(main_verb: Expression, auxiliary: Optional[str] = None) -> Expression:
    """
    Auxiliary verb (will, can, do, be, have).

    deplambda: (LAMBDA E : (p:EVENTMOD:$modal$ E))
    SQL: Auxiliaries don't affect SQL (tense/aspect/modality)

    Example:
        aux_relation(λx.works(x), "will")
        = λx.works(x)  [no change]
    """
    # Auxiliaries don't affect SQL structure
    return main_verb


# ==========================================================================================
# RELATIVE CLAUSES
# ==========================================================================================

def relcl_relation(noun_expr: Expression, rel_clause: Expression) -> Expression:
    """
    Relative clause (e.g., "employees who work").

    deplambda: (LAMBDA F G X : (EXISTS E : (F X) (G E) (p:EVENT.ENTITY:arg_1 E X)))
    SQL: Filter noun by relative clause predicate

    Example:
        relcl_relation(λx.employee(x), λx.works_remotely(x))
        = λx.employee(x) & works_remotely(x)
    """
    return vocab.filter_(noun_expr, rel_clause)


def acl_relation(noun_expr: Expression, acl_clause: Expression) -> Expression:
    """
    Adjectival clause (e.g., "person hired yesterday").

    deplambda: Similar to relcl
    SQL: Filter noun by clause predicate
    """
    return vocab.filter_(noun_expr, acl_clause)


# ==========================================================================================
# COORDINATION
# ==========================================================================================

def conj_relation(first_expr: Expression, second_expr: Expression, cc: str = "and") -> Expression:
    """
    Conjunction (and, or).

    deplambda: (LAMBDA F G X : (EXISTS X1 X2 : (F X1) (G X2) (p:PAIR X X1 X2)))
    SQL: UNION (for "or") or intersection (for "and")

    Example:
        conj_relation(λx.engineer(x), λx.manager(x), "or")
        = λx.engineer(x) UNION λx.manager(x)
    """
    if cc.lower() == "or":
        return vocab.union_(first_expr, second_expr)
    elif cc.lower() == "and":
        return vocab.intersect_(first_expr, second_expr)
    else:
        # Default to intersection
        return vocab.intersect_(first_expr, second_expr)


# ==========================================================================================
# DEFAULT FALLBACK
# ==========================================================================================

def identity(expr: Expression) -> Expression:
    """
    Identity function (no transformation).

    deplambda: (LAMBDA F G X : (F X) (G X))
    SQL: Return expression as-is
    """
    return expr


def compose_predicates(expr1: Expression, expr2: Expression) -> Expression:
    """
    Generic composition of two predicates.

    deplambda: (LAMBDA F G X : (F X) (G X))
    SQL: Filter by both predicates (conjunction)
    """
    return vocab.filter_(expr1, expr2)
