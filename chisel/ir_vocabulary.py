"""
SQL DRT Vocabulary

SQL-specific Discourse Representation Theory primitives built on nltk.sem.drt.
Upgraded from bare lambda calculus to DRT for proper negation/quantifier scope.

Key Changes from ir_vocabulary.py:
- Uses DRS (Discourse Representation Structures) instead of Expression
- Discourse referents explicitly introduced in DRS boxes
- Negation creates nested DRS boxes with explicit scope
- Quantifiers use DRT duplex conditions (implication)
- .fol() method converts DRS to FOL for SQL compilation

The SQL semantics are identical - we just wrap in DRT structure for better
handling of negation scope, quantifiers, and discourse referents.
"""

from typing import Dict, List, Optional
from nltk.sem.drt import (
    DRS, DrtExpression, DrtParser,
    DrtLambdaExpression, DrtApplicationExpression,
    DrtIndividualVariableExpression,
    DrtNegatedExpression, DrtConcatenation,
    DrtConstantExpression, DrtTokens
)
from nltk.sem.logic import Variable
from nltk.sem import logic
from dataclasses import dataclass
from enum import Enum


# ==========================
# TYPE SYSTEM (unchanged)
# ==========================

class SQLType(Enum):
    """SQL-specific types for lambda calculus."""
    ENTITY = "e"          # Database row
    NUMBER = "n"          # Numeric value
    STRING = "s"          # Text value
    BOOL = "t"            # Truth value
    SET_E = "e->t"        # Set of entities (predicate)
    PROP_N = "e->n"       # Property returning number
    PROP_S = "e->s"       # Property returning string
    RELATION = "e->e->t"  # Binary relation


@dataclass
class TypedExpression:
    """DRT expression with SQL type annotation."""
    expr: DrtExpression
    sql_type: SQLType


# ==========================
# DRT PARSER
# ==========================

_parser = DrtParser()


def parse_drs(drs_string: str) -> DRS:
    """Parse DRS from string."""
    return _parser.parse(drs_string)


# ==========================
# AGGREGATION OPERATORS
# ==========================

def count(set_pred: DrtExpression) -> DrtExpression:
    """
    COUNT aggregation: count(λx.P(x)) → number

    Example: count(λx.([],[student(x)]))
            = SELECT COUNT(*) FROM student

    The DRS introduces discourse referent x for the counted entities.
    """
    return DrtApplicationExpression(
        DrtConstantExpression(Variable('COUNT')),
        set_pred
    )


def argmax(set_pred: DrtExpression, measure_fn: DrtExpression) -> DrtExpression:
    """
    ARGMAX (superlative): argmax(λx.P(x), λx.f(x)) → entity

    Example: argmax(λx.([],[student(x)]), λx.([],[age(x)]))
            = SELECT * FROM student ORDER BY age DESC LIMIT 1
    """
    return DrtApplicationExpression(
        DrtApplicationExpression(
            DrtConstantExpression(Variable('ARGMAX')),
            set_pred
        ),
        measure_fn
    )


def order_by_desc(set_pred: DrtExpression, measure_fn: DrtExpression) -> DrtExpression:
    """ORDER BY DESC (all rows): SELECT * FROM ... ORDER BY ... DESC"""
    return DrtApplicationExpression(
        DrtApplicationExpression(
            DrtConstantExpression(Variable('ORDER_DESC')),
            set_pred
        ),
        measure_fn
    )


def order_by_asc(set_pred: DrtExpression, measure_fn: DrtExpression) -> DrtExpression:
    """ORDER BY ASC (all rows): SELECT * FROM ... ORDER BY ... ASC"""
    return DrtApplicationExpression(
        DrtApplicationExpression(
            DrtConstantExpression(Variable('ORDER_ASC')),
            set_pred
        ),
        measure_fn
    )


def project(set_expr: DrtExpression, columns: List[str]) -> DrtExpression:
    """
    PROJECT (column selection): SELECT col1, col2 FROM ...

    Example: project(λx.([],[student(x)]), ['name', 'age'])
            = SELECT name, age FROM student
    """
    columns_str = ','.join(columns)
    return DrtApplicationExpression(
        DrtApplicationExpression(
            DrtConstantExpression(Variable('PROJECT')),
            set_expr
        ),
        DrtConstantExpression(Variable(f'COLUMNS_{columns_str}'))
    )


def argmin(set_pred: DrtExpression, measure_fn: DrtExpression) -> DrtExpression:
    """ARGMIN (superlative): SELECT * FROM ... ORDER BY ... ASC LIMIT 1"""
    return DrtApplicationExpression(
        DrtApplicationExpression(
            DrtConstantExpression(Variable('ARGMIN')),
            set_pred
        ),
        measure_fn
    )


def avg(set_pred: DrtExpression, measure_fn: DrtExpression) -> DrtExpression:
    """AVG aggregation: SELECT AVG(...) FROM ..."""
    return DrtApplicationExpression(
        DrtApplicationExpression(
            DrtConstantExpression(Variable('AVG')),
            set_pred
        ),
        measure_fn
    )


def sum_(set_pred: DrtExpression, measure_fn: DrtExpression) -> DrtExpression:
    """SUM aggregation: SELECT SUM(...) FROM ..."""
    return DrtApplicationExpression(
        DrtApplicationExpression(
            DrtConstantExpression(Variable('SUM')),
            set_pred
        ),
        measure_fn
    )


def min_(set_pred: DrtExpression, measure_fn: DrtExpression) -> DrtExpression:
    """MIN aggregation: SELECT MIN(...) FROM ..."""
    return DrtApplicationExpression(
        DrtApplicationExpression(
            DrtConstantExpression(Variable('MIN')),
            set_pred
        ),
        measure_fn
    )


def max_(set_pred: DrtExpression, measure_fn: DrtExpression) -> DrtExpression:
    """MAX aggregation: SELECT MAX(...) FROM ..."""
    return DrtApplicationExpression(
        DrtApplicationExpression(
            DrtConstantExpression(Variable('MAX')),
            set_pred
        ),
        measure_fn
    )


# ==========================
# SET OPERATIONS
# ==========================

def filter_(set_pred1: DrtExpression, set_pred2: DrtExpression) -> DrtExpression:
    """
    SET INTERSECTION: filter(P, Q) = λx.DRS([],[P(x), Q(x)])

    Example: filter(λx.([],[student(x)]), λx.([],[gt(age(x), 20)]))
            = SELECT * FROM student WHERE age > 20

    The DRS concatenates both predicates in the same box - conjunction.
    """
    x = DrtIndividualVariableExpression(Variable('x'))
    # Apply both predicates to x and concatenate their DRS boxes
    # This creates: λx.DRS([],[student(x) & age(x) > 20])
    return DrtLambdaExpression(
        x.variable,
        DrtConcatenation(
            DrtApplicationExpression(set_pred1, x),
            DrtApplicationExpression(set_pred2, x)
        )
    )


def union_(set_pred1: DrtExpression, set_pred2: DrtExpression) -> DrtExpression:
    """
    SET UNION: λx.(P(x) ∨ Q(x))

    In DRT: λx.DRS([],[P(x) | Q(x)])
    The disjunction is within the DRS box.
    """
    x = DrtIndividualVariableExpression(Variable('x'))
    from nltk.sem.drt import DrtOrExpression
    return DrtLambdaExpression(
        x.variable,
        DRS([], [DrtOrExpression(
            DrtApplicationExpression(set_pred1, x),
            DrtApplicationExpression(set_pred2, x)
        )])
    )


def intersect_(set_pred1: DrtExpression, set_pred2: DrtExpression) -> DrtExpression:
    """SET INTERSECTION (same as filter_)."""
    return filter_(set_pred1, set_pred2)


def except_(set_pred1: DrtExpression, set_pred2: DrtExpression) -> DrtExpression:
    """
    SET DIFFERENCE: λx.DRS([],[P(x), NOT(Q(x))])

    Negation creates a nested DRS box with explicit scope.
    This is clearer than bare lambda negation.
    """
    x = DrtIndividualVariableExpression(Variable('x'))
    return DrtLambdaExpression(
        x.variable,
        DRS([], [
            DrtApplicationExpression(set_pred1, x),
            DrtNegatedExpression(DrtApplicationExpression(set_pred2, x))
        ])
    )


# ==========================
# COMPARISON OPERATORS
# ==========================

def gt(prop_fn: DrtExpression, value: DrtExpression) -> DrtExpression:
    """GREATER THAN: λx.DRS([],[f(x) > v])"""
    x = DrtIndividualVariableExpression(Variable('x'))
    return DrtLambdaExpression(
        x.variable,
        DRS([], [DrtApplicationExpression(
            DrtApplicationExpression(
                DrtConstantExpression(Variable('GT')),
                DrtApplicationExpression(prop_fn, x)
            ),
            value
        )])
    )


def lt(prop_fn: DrtExpression, value: DrtExpression) -> DrtExpression:
    """LESS THAN: λx.DRS([],[f(x) < v])"""
    x = DrtIndividualVariableExpression(Variable('x'))
    return DrtLambdaExpression(
        x.variable,
        DRS([], [DrtApplicationExpression(
            DrtApplicationExpression(
                DrtConstantExpression(Variable('LT')),
                DrtApplicationExpression(prop_fn, x)
            ),
            value
        )])
    )


def gte(prop_fn: DrtExpression, value: DrtExpression) -> DrtExpression:
    """GREATER THAN OR EQUAL: λx.DRS([],[f(x) >= v])"""
    x = DrtIndividualVariableExpression(Variable('x'))
    return DrtLambdaExpression(
        x.variable,
        DRS([], [DrtApplicationExpression(
            DrtApplicationExpression(
                DrtConstantExpression(Variable('GTE')),
                DrtApplicationExpression(prop_fn, x)
            ),
            value
        )])
    )


def lte(prop_fn: DrtExpression, value: DrtExpression) -> DrtExpression:
    """LESS THAN OR EQUAL: λx.DRS([],[f(x) <= v])"""
    x = DrtIndividualVariableExpression(Variable('x'))
    return DrtLambdaExpression(
        x.variable,
        DRS([], [DrtApplicationExpression(
            DrtApplicationExpression(
                DrtConstantExpression(Variable('LTE')),
                DrtApplicationExpression(prop_fn, x)
            ),
            value
        )])
    )


def eq(prop_fn: DrtExpression, value: DrtExpression) -> DrtExpression:
    """EQUALITY: λx.DRS([],[f(x) = v])"""
    x = DrtIndividualVariableExpression(Variable('x'))
    from nltk.sem.drt import DrtEqualityExpression
    return DrtLambdaExpression(
        x.variable,
        DRS([], [DrtEqualityExpression(
            DrtApplicationExpression(prop_fn, x),
            value
        )])
    )


def neq(prop_fn: DrtExpression, value: DrtExpression) -> DrtExpression:
    """NOT EQUAL: λx.DRS([],[NOT(f(x) = v)])"""
    x = DrtIndividualVariableExpression(Variable('x'))
    from nltk.sem.drt import DrtEqualityExpression
    return DrtLambdaExpression(
        x.variable,
        DRS([], [DrtNegatedExpression(DrtEqualityExpression(
            DrtApplicationExpression(prop_fn, x),
            value
        ))])
    )


# ==========================
# QUANTIFIERS & EXISTS
# ==========================

def exists(set_pred: DrtExpression) -> DrtExpression:
    """
    EXISTS quantifier: EXISTS(λx.P(x))

    In DRT, this is already handled by discourse referents in DRS boxes.
    We wrap in EXISTS constant for SQL compilation.
    """
    return DrtApplicationExpression(
        DrtConstantExpression(Variable('EXISTS')),
        set_pred
    )


def not_exists(set_pred: DrtExpression) -> DrtExpression:
    """
    NOT EXISTS quantifier.

    In DRT: NOT(EXISTS(P)) = NOT(DRS([x],[P(x)]))
    The negation creates a nested DRS box with explicit scope.
    """
    return DrtApplicationExpression(
        DrtConstantExpression(Variable('NOT_EXISTS')),
        set_pred
    )


# ==========================
# GROUP BY
# ==========================

def group_by(set_pred: DrtExpression, grouping_fn: DrtExpression, agg_fn: DrtExpression) -> DrtExpression:
    """
    GROUP BY with aggregation:
    group_by(λx.P(x), λx.f(x), λs.count(s))

    Example: SELECT dept, COUNT(*) FROM student GROUP BY dept
    """
    return DrtApplicationExpression(
        DrtApplicationExpression(
            DrtApplicationExpression(
                DrtConstantExpression(Variable('GROUP_BY')),
                set_pred
            ),
            grouping_fn
        ),
        agg_fn
    )


# ==========================
# SCHEMA ATOMS
# ==========================

def table_atom(table_name: str) -> DrtExpression:
    """
    Table as set predicate: λx.DRS([],[table_name(x)])

    Example: table_atom('student') = λx.DRS([],[student(x)])

    The DRS introduces discourse referent x for table rows.
    """
    return parse_drs(f'\\x.([],[{table_name}(x)])')


def column_atom(table_name: str, column_name: str) -> DrtExpression:
    """
    Column as property function: λx.DRS([],[table_column(x)])

    Example: column_atom('student', 'age') = λx.DRS([],[student_age(x)])
    """
    atom_name = f"{table_name}_{column_name}"
    return parse_drs(f'\\x.([],[{atom_name}(x)])')


def constant_atom(value: str, sql_type: SQLType = SQLType.STRING) -> DrtExpression:
    """
    Constant value as DRT expression.

    Example: constant_atom('John') = 'John'
    """
    if sql_type == SQLType.NUMBER:
        return DrtConstantExpression(Variable(str(value)))
    else:
        return DrtConstantExpression(Variable(f"'{value}'"))


# ==========================
# UTILITIES
# ==========================

def is_aggregation(expr: DrtExpression) -> bool:
    """Check if expression contains aggregation operators."""
    expr_str = str(expr)
    agg_ops = ['COUNT', 'AVG', 'SUM', 'MIN', 'MAX', 'ARGMAX', 'ARGMIN']
    return any(op in expr_str for op in agg_ops)


def is_superlative(expr: DrtExpression) -> bool:
    """Check if expression is a superlative (ARGMAX/ARGMIN)."""
    expr_str = str(expr)
    return 'ARGMAX' in expr_str or 'ARGMIN' in expr_str


def extract_table_refs(expr: DrtExpression) -> List[str]:
    """
    Extract table names referenced in DRT expression.

    Used for join inference. Works by converting to FOL first.
    """
    # Convert DRS to FOL for easier pattern matching
    if hasattr(expr, 'fol'):
        fol_expr = expr.fol()
        expr_str = str(fol_expr)
    else:
        expr_str = str(expr)

    import re
    pattern = r'\b([a-z_]+)\('
    matches = re.findall(pattern, expr_str)

    logical_ops = {'and', 'or', 'not', 'exists', 'all', 'count', 'avg', 'sum', 'min', 'max',
                   'argmax', 'argmin', 'gt', 'lt', 'gte', 'lte', 'eq', 'group_by'}

    table_names = set()
    for m in matches:
        if m not in logical_ops:
            if '_' in m:
                table_names.add(m.split('_')[0])
            else:
                table_names.add(m)

    return list(table_names)


def get_expression_type(expr: DrtExpression) -> Optional[SQLType]:
    """Infer SQL type of DRT expression."""
    expr_str = str(expr)

    if any(op in expr_str for op in ['COUNT', 'AVG', 'SUM', 'MIN', 'MAX']):
        return SQLType.NUMBER

    if any(op in expr_str for op in ['ARGMAX', 'ARGMIN']):
        return SQLType.ENTITY

    if isinstance(expr, DrtLambdaExpression):
        return SQLType.SET_E

    return None
