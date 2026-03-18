"""
SQL Lambda Vocabulary

SQL-specific typed lambda calculus primitives built on nltk.sem.logic.
These are the atomic building blocks for compositional SQL semantics.

Design: Each SQL operation is a typed constant or combinator that can be
composed using lambda abstraction and application.
"""

from typing import Dict, List, Optional
from nltk.sem.logic import (
    Expression, ApplicationExpression, LambdaExpression,
    Variable, ConstantExpression, AndExpression, OrExpression,
    NegatedExpression, EqualityExpression
)
from nltk.sem import logic
from dataclasses import dataclass
from enum import Enum


# ==========================
# TYPE SYSTEM
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
    """Lambda expression with SQL type annotation."""
    expr: Expression
    sql_type: SQLType


# ==========================
# AGGREGATION OPERATORS
# ==========================

def count(set_pred: Expression) -> Expression:
    """
    COUNT aggregation: count(λx.P(x)) → number

    Example: count(λx.student(x)) = SELECT COUNT(*) FROM student
    """
    return ApplicationExpression(
        ConstantExpression(Variable('COUNT')),
        set_pred
    )


def argmax(set_pred: Expression, measure_fn: Expression) -> Expression:
    """
    ARGMAX (superlative): argmax(λx.P(x), λx.f(x)) → entity

    Example: argmax(λx.student(x), λx.age(x))
            = SELECT * FROM student ORDER BY age DESC LIMIT 1
    """
    return ApplicationExpression(
        ApplicationExpression(
            ConstantExpression(Variable('ARGMAX')),
            set_pred
        ),
        measure_fn
    )


def order_by_desc(set_pred: Expression, measure_fn: Expression) -> Expression:
    """
    ORDER BY DESC (all rows): order_by_desc(λx.P(x), λx.f(x)) → set

    Example: order_by_desc(λx.student(x), λx.age(x))
            = SELECT * FROM student ORDER BY age DESC
    """
    return ApplicationExpression(
        ApplicationExpression(
            ConstantExpression(Variable('ORDER_DESC')),
            set_pred
        ),
        measure_fn
    )


def order_by_asc(set_pred: Expression, measure_fn: Expression) -> Expression:
    """
    ORDER BY ASC (all rows): order_by_asc(λx.P(x), λx.f(x)) → set

    Example: order_by_asc(λx.student(x), λx.age(x))
            = SELECT * FROM student ORDER BY age ASC
    """
    return ApplicationExpression(
        ApplicationExpression(
            ConstantExpression(Variable('ORDER_ASC')),
            set_pred
        ),
        measure_fn
    )


def project(set_expr: Expression, columns: List[str]) -> Expression:
    """
    PROJECT (column selection): project(expr, [col1, col2]) → expr with projected columns

    Example: project(λx.student(x), ['name', 'age'])
            = SELECT name, age FROM student

    The columns list is encoded as a comma-separated COLUMNS constant for the compiler.
    """
    columns_str = ','.join(columns)
    return ApplicationExpression(
        ApplicationExpression(
            ConstantExpression(Variable('PROJECT')),
            set_expr
        ),
        ConstantExpression(Variable(f'COLUMNS_{columns_str}'))
    )


def argmin(set_pred: Expression, measure_fn: Expression) -> Expression:
    """
    ARGMIN (superlative): argmin(λx.P(x), λx.f(x)) → entity

    Example: argmin(λx.student(x), λx.age(x))
            = SELECT * FROM student ORDER BY age ASC LIMIT 1
    """
    return ApplicationExpression(
        ApplicationExpression(
            ConstantExpression(Variable('ARGMIN')),
            set_pred
        ),
        measure_fn
    )


def avg(set_pred: Expression, measure_fn: Expression) -> Expression:
    """
    AVG aggregation: avg(λx.P(x), λx.f(x)) → number

    Example: avg(λx.student(x), λx.gpa(x))
            = SELECT AVG(gpa) FROM student
    """
    return ApplicationExpression(
        ApplicationExpression(
            ConstantExpression(Variable('AVG')),
            set_pred
        ),
        measure_fn
    )


def sum_(set_pred: Expression, measure_fn: Expression) -> Expression:
    """SUM aggregation."""
    return ApplicationExpression(
        ApplicationExpression(
            ConstantExpression(Variable('SUM')),
            set_pred
        ),
        measure_fn
    )


def min_(set_pred: Expression, measure_fn: Expression) -> Expression:
    """MIN aggregation."""
    return ApplicationExpression(
        ApplicationExpression(
            ConstantExpression(Variable('MIN')),
            set_pred
        ),
        measure_fn
    )


def max_(set_pred: Expression, measure_fn: Expression) -> Expression:
    """MAX aggregation."""
    return ApplicationExpression(
        ApplicationExpression(
            ConstantExpression(Variable('MAX')),
            set_pred
        ),
        measure_fn
    )


# ==========================
# SET OPERATIONS
# ==========================

def filter_(set_pred1: Expression, set_pred2: Expression) -> Expression:
    """
    SET INTERSECTION: filter(P, Q) = λx.(P(x) ∧ Q(x))

    Example: filter(λx.student(x), λx.gt(age(x), 20))
            = SELECT * FROM student WHERE age > 20
    """
    x = Variable('x')
    return LambdaExpression(x, AndExpression(
        ApplicationExpression(set_pred1, x),
        ApplicationExpression(set_pred2, x)
    ))


def union_(set_pred1: Expression, set_pred2: Expression) -> Expression:
    """SET UNION: λx.(P(x) ∨ Q(x))"""
    x = Variable('x')
    return LambdaExpression(x, OrExpression(
        ApplicationExpression(set_pred1, x),
        ApplicationExpression(set_pred2, x)
    ))


def intersect_(set_pred1: Expression, set_pred2: Expression) -> Expression:
    """SET INTERSECTION (same as filter_)."""
    return filter_(set_pred1, set_pred2)


def except_(set_pred1: Expression, set_pred2: Expression) -> Expression:
    """SET DIFFERENCE: λx.(P(x) ∧ ¬Q(x))"""
    x = Variable('x')
    return LambdaExpression(x, AndExpression(
        ApplicationExpression(set_pred1, x),
        NegatedExpression(ApplicationExpression(set_pred2, x))
    ))


# ==========================
# COMPARISON OPERATORS
# ==========================

def gt(prop_fn: Expression, value: Expression) -> Expression:
    """
    GREATER THAN: λx.(f(x) > v)

    Example: gt(age, 20) = λx.(age(x) > 20)
    """
    x = Variable('x')
    # Represent as application for pattern matching
    return LambdaExpression(x, ApplicationExpression(
        ApplicationExpression(
            ConstantExpression(Variable('GT')),
            ApplicationExpression(prop_fn, x)
        ),
        value
    ))


def lt(prop_fn: Expression, value: Expression) -> Expression:
    """LESS THAN: λx.(f(x) < v)"""
    x = Variable('x')
    return LambdaExpression(x, ApplicationExpression(
        ApplicationExpression(
            ConstantExpression(Variable('LT')),
            ApplicationExpression(prop_fn, x)
        ),
        value
    ))


def gte(prop_fn: Expression, value: Expression) -> Expression:
    """GREATER THAN OR EQUAL: λx.(f(x) >= v)"""
    x = Variable('x')
    return LambdaExpression(x, ApplicationExpression(
        ApplicationExpression(
            ConstantExpression(Variable('GTE')),
            ApplicationExpression(prop_fn, x)
        ),
        value
    ))


def lte(prop_fn: Expression, value: Expression) -> Expression:
    """LESS THAN OR EQUAL: λx.(f(x) <= v)"""
    x = Variable('x')
    return LambdaExpression(x, ApplicationExpression(
        ApplicationExpression(
            ConstantExpression(Variable('LTE')),
            ApplicationExpression(prop_fn, x)
        ),
        value
    ))


def eq(prop_fn: Expression, value: Expression) -> Expression:
    """
    EQUALITY: λx.(f(x) = v)

    Example: eq(name, 'John') = λx.(name(x) = 'John')
    """
    x = Variable('x')
    return LambdaExpression(x, EqualityExpression(
        ApplicationExpression(prop_fn, x),
        value
    ))


def neq(prop_fn: Expression, value: Expression) -> Expression:
    """NOT EQUAL: λx.(f(x) ≠ v)"""
    x = Variable('x')
    return LambdaExpression(x, NegatedExpression(
        EqualityExpression(
            ApplicationExpression(prop_fn, x),
            value
        )
    ))


# ==========================
# QUANTIFIERS & EXISTS
# ==========================

def exists(set_pred: Expression) -> Expression:
    """
    EXISTS quantifier: EXISTS(λx.P(x))

    Compiles to: WHERE EXISTS (SELECT * FROM ... WHERE P)
    """
    return ApplicationExpression(
        ConstantExpression(Variable('EXISTS')),
        set_pred
    )


def not_exists(set_pred: Expression) -> Expression:
    """NOT EXISTS quantifier."""
    return ApplicationExpression(
        ConstantExpression(Variable('NOT_EXISTS')),
        set_pred
    )


# ==========================
# GROUP BY
# ==========================

def group_by(set_pred: Expression, grouping_fn: Expression, agg_fn: Expression) -> Expression:
    """
    GROUP BY with aggregation:
    group_by(λx.P(x), λx.f(x), λs.count(s))

    Example: group_by(λx.student(x), λx.dept(x), λs.count(s))
            = SELECT dept, COUNT(*) FROM student GROUP BY dept
    """
    return ApplicationExpression(
        ApplicationExpression(
            ApplicationExpression(
                ConstantExpression(Variable('GROUP_BY')),
                set_pred
            ),
            grouping_fn
        ),
        agg_fn
    )


# ==========================
# SCHEMA ATOMS
# ==========================

def table_atom(table_name: str) -> Expression:
    """
    Table as set predicate: λx.table_name(x)

    Example: table_atom('student') = λx.student(x)
    """
    return Expression.fromstring(f'\\x.{table_name}(x)')


def column_atom(table_name: str, column_name: str) -> Expression:
    """
    Column as property function: λx.table_name_column_name(x)

    Example: column_atom('student', 'age') = λx.student_age(x)
    """
    # Use table.column format for uniqueness
    atom_name = f"{table_name}_{column_name}"
    return Expression.fromstring(f'\\x.{atom_name}(x)')


def constant_atom(value: str, sql_type: SQLType = SQLType.STRING) -> Expression:
    """
    Constant value as expression.

    Example: constant_atom('John') = 'John'
    """
    if sql_type == SQLType.NUMBER:
        return ConstantExpression(Variable(str(value)))
    else:
        # String constants need proper escaping
        return ConstantExpression(Variable(f"'{value}'"))


# ==========================
# UTILITIES
# ==========================

def is_aggregation(expr: Expression) -> bool:
    """Check if expression contains aggregation operators."""
    expr_str = str(expr)
    agg_ops = ['COUNT', 'AVG', 'SUM', 'MIN', 'MAX', 'ARGMAX', 'ARGMIN']
    return any(op in expr_str for op in agg_ops)


def is_superlative(expr: Expression) -> bool:
    """Check if expression is a superlative (ARGMAX/ARGMIN)."""
    expr_str = str(expr)
    return 'ARGMAX' in expr_str or 'ARGMIN' in expr_str


def extract_table_refs(expr: Expression) -> List[str]:
    """
    Extract table names referenced in expression.

    Used for join inference.
    """
    expr_str = str(expr)
    # Simple heuristic: look for predicates like student(x), student_age(x)
    import re
    # Match lowercase identifiers followed by (
    pattern = r'\b([a-z_]+)\('
    matches = re.findall(pattern, expr_str)
    # Filter out logical operators
    logical_ops = {'and', 'or', 'not', 'exists', 'all', 'count', 'avg', 'sum', 'min', 'max',
                   'argmax', 'argmin', 'gt', 'lt', 'gte', 'lte', 'eq', 'group_by'}

    table_names = set()
    for m in matches:
        if m not in logical_ops:
            # If has underscore (table_column), extract table part
            if '_' in m:
                table_names.add(m.split('_')[0])
            else:
                # Otherwise it's a table name directly
                table_names.add(m)

    return list(table_names)


def get_expression_type(expr: Expression) -> Optional[SQLType]:
    """
    Infer SQL type of expression.

    Used for type checking during composition.
    """
    expr_str = str(expr)

    # Aggregations return numbers
    if any(op in expr_str for op in ['COUNT', 'AVG', 'SUM', 'MIN', 'MAX']):
        return SQLType.NUMBER

    # Superlatives return entities
    if any(op in expr_str for op in ['ARGMAX', 'ARGMIN']):
        return SQLType.ENTITY

    # Lambda expressions with single variable are predicates (sets)
    if isinstance(expr, LambdaExpression):
        # Check body for boolean operators
        body_str = str(expr.term)
        if any(op in body_str for op in ['&', '|', '=', 'GT', 'LT']):
            return SQLType.SET_E
        return SQLType.SET_E

    # Default
    return None
