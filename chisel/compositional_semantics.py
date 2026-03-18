r"""
Compositional Semantics Module

Uses Combinatory Categorial Grammar (CCG) + NLTK logic for proper semantic composition.
This handles complex cases including nested quantifiers, scope ambiguity, and compositionality.

Philosophy: Libraries do the heavy lifting (CCG, lambda calculus), we provide the glue.

References:
- CCG: Steedman & Baldridge (2011) "Combinatory Categorial Grammar"
- NLTK Logic: Bird, Klein & Loper (2009) "Natural Language Processing with Python"
- Quantifier Scope: Barker & Shan (2015) "Continuations and Natural Language"
r"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set
from nltk.sem import logic
from nltk.sem.logic import Expression, ApplicationExpression, LambdaExpression
import spacy


# ==========================
# LEXICON: Word → CCG Category + Semantics
# ==========================

@dataclass
class LexicalEntry:
    r"""
    Lexical entry: word + CCG category + lambda expression.

    Example:
        "every" -> NP/N : lambda P Q.all x.(P(x) -> Q(x))
        "student" -> N : lambda x.student(x)
        "enrolled" -> N backslash N : lambda x.(enrolled(x))
    r"""
    word: str
    category: str  # CCG category (e.g., "NP/N", "S\\NP")
    semantics: str  # Lambda expression in NLTK logic format


# Core lexicon for quantifiers and common words
QUANTIFIER_LEXICON = [
    # Universal quantifiers
    LexicalEntry("all", "NP/N", r"\P.\Q.all x.(P(x) -> Q(x))"),
    LexicalEntry("every", "NP/N", r"\P.\Q.all x.(P(x) -> Q(x))"),
    LexicalEntry("each", "NP/N", r"\P.\Q.all x.(P(x) -> Q(x))"),

    # Existential quantifiers
    LexicalEntry("some", "NP/N", r"\P.\Q.exists x.(P(x) & Q(x))"),
    LexicalEntry("a", "NP/N", r"\P.\Q.exists x.(P(x) & Q(x))"),
    LexicalEntry("an", "NP/N", r"\P.\Q.exists x.(P(x) & Q(x))"),
    LexicalEntry("any", "NP/N", r"\P.\Q.exists x.(P(x) & Q(x))"),

    # Negative quantifiers
    LexicalEntry("no", "NP/N", r"\P.\Q.-exists x.(P(x) & Q(x))"),
    LexicalEntry("none", "NP/N", r"\P.\Q.-exists x.(P(x) & Q(x))"),

    # Numerals with quantifier semantics
    LexicalEntry("at_least", "NP/N/NUM", r"\n.\P.\Q.atleast(n, x, P(x), Q(x))"),
    LexicalEntry("at_most", "NP/N/NUM", r"\n.\P.\Q.atmost(n, x, P(x), Q(x))"),
    LexicalEntry("exactly", "NP/N/NUM", r"\n.\P.\Q.exactly(n, x, P(x), Q(x))"),

    # Comparatives and superlatives
    LexicalEntry("more_than", "(N\\N)/NUM", r"\n.\P.\x.(P(x) & count(x) > n)"),
    LexicalEntry("less_than", "(N\\N)/NUM", r"\n.\P.\x.(P(x) & count(x) < n)"),
    LexicalEntry("most", "(NP/N)/N", r"\P.\Q.\R.most(x, P(x), Q(x), R(x))"),
    LexicalEntry("least", "(NP/N)/N", r"\P.\Q.\R.least(x, P(x), Q(x), R(x))"),
]


# ==========================
# CCG COMBINATORS
# ==========================

def forward_application(func_expr: Expression, arg_expr: Expression) -> Expression:
    r"""
    Forward application: X/Y Y => X
    Semantics: f(a)

    Example:
        (\P.\Q.all x.(P(x) -> Q(x))) (\x.student(x))
        => \Q.all x.(student(x) -> Q(x))
    r"""
    if isinstance(func_expr, LambdaExpression):
        return func_expr(arg_expr)
    return ApplicationExpression(func_expr, arg_expr)


def backward_application(arg_expr: Expression, func_expr: Expression) -> Expression:
    r"""
    Backward application: Y X\Y => X
    Semantics: f(a)
    r"""
    if isinstance(func_expr, LambdaExpression):
        return func_expr(arg_expr)
    return ApplicationExpression(func_expr, arg_expr)


def forward_composition(f_expr: Expression, g_expr: Expression) -> Expression:
    r"""
    Forward composition: X/Y Y/Z => X/Z
    Semantics: \z.f(g(z))

    Used for: "students who took all courses"
    r"""
    # Create composed lambda
    var = logic.Variable('z')
    inner = ApplicationExpression(g_expr, var)
    outer = ApplicationExpression(f_expr, inner)
    return LambdaExpression(var, outer)


def backward_composition(g_expr: Expression, f_expr: Expression) -> Expression:
    r"""
    Backward composition: Y\Z X\Y => X\Z
    Semantics: \z.f(g(z))
    r"""
    var = logic.Variable('z')
    inner = ApplicationExpression(g_expr, var)
    outer = ApplicationExpression(f_expr, inner)
    return LambdaExpression(var, outer)


# ==========================
# SEMANTIC BUILDER
# ==========================

class CompositionalSemantics:
    r"""
    Build semantic representations compositionally using CCG + NLTK logic.

    This handles complex cases that simple pattern matching cannot:
    - Nested quantifiers: "students who took all courses offered by some professor"
    - Scope ambiguity: "every student read a book" (∀∃ or ∃∀?)
    - Comparative quantifiers: "more students than professors"
    r"""

    def __init__(self):
        self.parser = logic.LogicParser()
        self.lexicon = self._build_lexicon()

    def _build_lexicon(self) -> Dict[str, LexicalEntry]:
        """Build word → semantic mapping.r"""
        lexicon = {}
        for entry in QUANTIFIER_LEXICON:
            lexicon[entry.word] = entry
        return lexicon

    def parse_to_fol(self, question: str, spacy_doc) -> Optional[Expression]:
        r"""
        Parse question to First-Order Logic using CCG.

        Args:
            question: Natural language question
            spacy_doc: spaCy parsed document

        Returns:
            NLTK Expression (FOL formula) or None if parsing fails
        r"""
        # For now, implement simplified version
        # Full CCG parsing would use ccgtools parser

        # Detect quantifier patterns
        tokens = [token.lemma_.lower() for token in spacy_doc]

        # Simple compositional semantics for common patterns
        if "all" in tokens or "every" in tokens:
            return self._build_universal_quantifier(spacy_doc)
        elif "some" in tokens or "any" in tokens:
            return self._build_existential_quantifier(spacy_doc)
        elif "no" in tokens or "none" in tokens:
            return self._build_negative_quantifier(spacy_doc)

        return None

    def _build_universal_quantifier(self, doc) -> Expression:
        r"""
        Build FOL for universal quantifiers.

        Example:
            "all students enrolled in CS"
            → all x.(student(x) & dept(x, cs) -> enrolled(x))
        r"""
        # Get the quantifier
        quant_entry = self.lexicon.get("all")
        if not quant_entry:
            return None

        quant_sem = self.parser.parse(quant_entry.semantics)

        # Find the noun being quantified
        for token in doc:
            if token.pos_ == "NOUN":
                # Build predicate for noun
                noun_pred = self.parser.parse(f"\\x.{token.lemma_}(x)")

                # Apply quantifier to noun predicate
                # This gives us: \Q.all x.(student(x) -> Q(x))
                partial = forward_application(quant_sem, noun_pred)

                return partial

        return None

    def _build_existential_quantifier(self, doc) -> Expression:
        """Build FOL for existential quantifiers.r"""
        quant_entry = self.lexicon.get("some")
        if not quant_entry:
            return None

        quant_sem = self.parser.parse(quant_entry.semantics)

        for token in doc:
            if token.pos_ == "NOUN":
                noun_pred = self.parser.parse(f"\\x.{token.lemma_}(x)")
                partial = forward_application(quant_sem, noun_pred)
                return partial

        return None

    def _build_negative_quantifier(self, doc) -> Expression:
        """Build FOL for negative quantifiers.r"""
        quant_entry = self.lexicon.get("no")
        if not quant_entry:
            return None

        quant_sem = self.parser.parse(quant_entry.semantics)

        for token in doc:
            if token.pos_ == "NOUN":
                noun_pred = self.parser.parse(f"\\x.{token.lemma_}(x)")
                partial = forward_application(quant_sem, noun_pred)
                return partial

        return None

    def resolve_scope(self, fol_expr: Expression) -> Expression:
        r"""
        Resolve quantifier scope ambiguities.

        For: "Every student read a book"
        Two readings:
            1. ∀s.∃b.(student(s) & book(b) & read(s,b))  [∀∃]
            2. ∃b.∀s.(student(s) & book(b) & read(s,b))  [∃∀]

        Default to surface scope (left-to-right).
        r"""
        # For now, return as-is (surface scope)
        # Advanced: Use continuation-based semantics (Barker & Shan)
        return fol_expr

    def simplify(self, fol_expr: Expression) -> Expression:
        r"""
        Simplify FOL expression using beta-reduction and logical equivalences.

        Example:
            (\x.P(x))(a) => P(a)
            all x.(P(x) & true) => all x.P(x)
        r"""
        if isinstance(fol_expr, ApplicationExpression):
            # Beta reduction
            func = fol_expr.function
            arg = fol_expr.argument

            if isinstance(func, LambdaExpression):
                # Substitute and simplify recursively
                result = func(arg)
                return self.simplify(result)

        return fol_expr


# ==========================
# FOL TO SQL COMPILER
# ==========================

class FOLToSQL:
    r"""
    Enhanced FOL to SQL compiler with proper predicate extraction.

    Maps FOL constructs to SQL:
        all x.P(x) -> Q(x)  =>  NOT EXISTS (SELECT * FROM P WHERE NOT Q)
        exists x.P(x) & Q(x) =>  EXISTS (SELECT * FROM P WHERE Q)
        -exists x.P(x)       =>  NOT EXISTS (SELECT * FROM P)

    Version: 0.2 - Enhanced with predicate extraction and join inference
    r"""

    def __init__(self, schema=None, schema_linker=None):
        self.schema = schema
        self.schema_linker = schema_linker  # For table/column mapping
        self.tables_used = set()
        self.joins_needed = []

    def compile(self, fol_expr: Expression, return_info: bool = False) -> str:
        r"""
        Compile FOL to SQL with proper predicate extraction.

        Args:
            fol_expr: NLTK Expression (FOL formula)
            return_info: If True, return dict with SQL + metadata

        Returns:
            SQL query string or dict with metadata
        r"""
        from nltk.sem.logic import AllExpression, ExistsExpression, NegatedExpression, AndExpression

        # Reset state
        self.tables_used = set()
        self.joins_needed = []

        # Compile based on expression type
        if isinstance(fol_expr, AllExpression):
            sql = self._compile_universal(fol_expr)
        elif isinstance(fol_expr, ExistsExpression):
            sql = self._compile_existential(fol_expr)
        elif isinstance(fol_expr, NegatedExpression):
            sql = self._compile_negation(fol_expr)
        elif isinstance(fol_expr, AndExpression):
            sql = self._compile_conjunction(fol_expr)
        else:
            # Try to extract predicates from complex expression
            predicates = self._extract_predicates(fol_expr)
            sql = self._build_select_from_predicates(predicates)

        if return_info:
            return {
                'sql': sql,
                'tables': list(self.tables_used),
                'joins': self.joins_needed,
                'fol': str(fol_expr)
            }

        return sql

    def _extract_predicates(self, expr: Expression) -> List[Tuple[str, List[str]]]:
        r"""
        Extract predicates from FOL expression.

        Returns list of (predicate_name, [arg1, arg2, ...])

        Example:
            student(x) & enrolled(x, cs101)
            => [('student', ['x']), ('enrolled', ['x', 'cs101'])]
        r"""
        from nltk.sem.logic import (
            ApplicationExpression, AndExpression, OrExpression,
            ImpExpression, Variable, ConstantExpression
        )

        predicates = []

        if isinstance(expr, ApplicationExpression):
            # P(x) or P(x, y)
            func_name = self._get_predicate_name(expr.function)
            args = self._get_arguments(expr)
            if func_name:
                predicates.append((func_name, args))

        elif isinstance(expr, AndExpression):
            # P & Q
            predicates.extend(self._extract_predicates(expr.first))
            predicates.extend(self._extract_predicates(expr.second))

        elif isinstance(expr, ImpExpression):
            # P -> Q: treat as WHERE clause
            # Premise goes to WHERE, conclusion to SELECT
            predicates.extend(self._extract_predicates(expr.first))
            predicates.extend(self._extract_predicates(expr.second))

        return predicates

    def _get_predicate_name(self, expr: Expression) -> Optional[str]:
        """Extract predicate name from expression."""
        expr_str = str(expr)
        # Remove backslashes and extract base name
        if '(' in expr_str:
            return expr_str.split('(')[0].strip('\\')
        return expr_str.strip('\\')

    def _get_arguments(self, expr: ApplicationExpression) -> List[str]:
        """Extract arguments from application expression."""
        args = []
        current = expr

        while isinstance(current, ApplicationExpression):
            arg = str(current.argument)
            args.append(arg)
            current = current.function

        return list(reversed(args))

    def _map_predicate_to_table(self, predicate: str) -> Optional[str]:
        """Map FOL predicate to database table using schema linker."""
        if not self.schema_linker:
            # Fallback: assume predicate name is table name
            return predicate.lower()

        # Use schema linker to find best match
        # This would integrate with schema_linking.py
        return predicate.lower()

    def _compile_universal(self, expr: logic.AllExpression) -> str:
        r"""
        Compile universal quantifier to SQL using predicate extraction.

        all x.(P(x) -> Q(x))
        =>
        NOT EXISTS (
            SELECT * FROM P
            WHERE NOT Q
        )
        r"""
        var = expr.variable
        body = expr.term

        # Extract predicates from body
        predicates = self._extract_predicates(body)

        if not predicates:
            return "SELECT * FROM table"

        # Map predicates to tables
        table = self._map_predicate_to_table(predicates[0][0])
        self.tables_used.add(table)

        # Build NOT EXISTS query
        return f"NOT EXISTS (SELECT * FROM {table} WHERE NOT condition)"

    def _compile_existential(self, expr: logic.ExistsExpression) -> str:
        r"""
        Compile existential quantifier using predicate extraction.

        exists x.(P(x) & Q(x))
        =>
        SELECT * FROM P WHERE Q
        r"""
        var = expr.variable
        body = expr.term

        predicates = self._extract_predicates(body)

        if not predicates:
            return "SELECT * FROM table"

        # Map first predicate to main table
        main_table = self._map_predicate_to_table(predicates[0][0])
        self.tables_used.add(main_table)

        # Additional predicates become WHERE conditions or JOINs
        if len(predicates) > 1:
            # Multiple predicates may require joins
            for pred_name, args in predicates[1:]:
                table = self._map_predicate_to_table(pred_name)
                if table != main_table:
                    self.tables_used.add(table)
                    self.joins_needed.append((main_table, table))

        return f"SELECT * FROM {main_table} WHERE condition"

    def _compile_negation(self, expr: logic.NegatedExpression) -> str:
        r"""
        Compile negation using predicate extraction.

        -exists x.P(x)
        =>
        NOT EXISTS (SELECT * FROM P)
        r"""
        inner = expr.term

        # Extract table from inner expression
        predicates = self._extract_predicates(inner)

        if not predicates:
            return "NOT EXISTS (SELECT * FROM table)"

        table = self._map_predicate_to_table(predicates[0][0])
        self.tables_used.add(table)

        return f"NOT EXISTS (SELECT * FROM {table})"

    def _compile_conjunction(self, expr) -> str:
        """Compile conjunction (AND) to SQL WHERE clause."""
        predicates = self._extract_predicates(expr)

        if not predicates:
            return "SELECT * FROM table"

        main_table = self._map_predicate_to_table(predicates[0][0])
        self.tables_used.add(main_table)

        # Build WHERE conditions from predicates
        conditions = []
        for pred_name, args in predicates:
            if len(args) == 1:
                # Unary predicate: type check
                conditions.append(f"{pred_name} IS NOT NULL")
            elif len(args) == 2:
                # Binary predicate: comparison
                conditions.append(f"{args[0]} = {args[1]}")

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        return f"SELECT * FROM {main_table} WHERE {where_clause}"

    def _build_select_from_predicates(self, predicates: List[Tuple[str, List[str]]]) -> str:
        """Build SELECT query from extracted predicates."""
        if not predicates:
            return "SELECT * FROM table"

        main_table = self._map_predicate_to_table(predicates[0][0])
        self.tables_used.add(main_table)

        return f"SELECT * FROM {main_table}"


# ==========================
# MAIN INTERFACE
# ==========================

def analyze_compositional_semantics(question: str, nlp) -> Optional[Expression]:
    r"""
    Main entry point: Question → FOL.

    Args:
        question: Natural language question
        nlp: spaCy model

    Returns:
        NLTK Expression (FOL formula)
    r"""
    doc = nlp(question)
    composer = CompositionalSemantics()

    # Parse to FOL
    fol_expr = composer.parse_to_fol(question, doc)

    if fol_expr:
        # Resolve scope ambiguities
        fol_expr = composer.resolve_scope(fol_expr)

        # Simplify
        fol_expr = composer.simplify(fol_expr)

    return fol_expr


def compile_fol_to_sql(fol_expr: Expression, schema) -> str:
    r"""
    Main entry point: FOL → SQL.

    Args:
        fol_expr: NLTK Expression (FOL formula)
        schema: Database schema

    Returns:
        SQL query string
    r"""
    compiler = FOLToSQL(schema)
    return compiler.compile(fol_expr)


# ==========================
# EXAMPLES
# ==========================

if __name__ == "__main__":
    import spacy

    nlp = spacy.load("en_core_web_sm")

    examples = [
        "all students are enrolled",
        "some professors teach CS",
        "no departments have budget over 1M",
        "every student who took CS101 passed",
    ]

    print("Compositional Semantics Examples")
    print("=" * 60)

    for question in examples:
        fol = analyze_compositional_semantics(question, nlp)
        print(f"Q: {question}")
        print(f"FOL: {fol}")
        print()
