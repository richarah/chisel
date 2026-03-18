"""
IR Composer

Bottom-up compositional semantic construction using template matching.
Inspired by ccg2lambda's template-based approach.

Architecture:
1. Load templates from ir_templates.yaml
2. Match dependency patterns against spaCy parse tree
3. Build lambda expressions compositionally (leaves → root)
4. Type-check expressions during construction
5. Return final lambda expression for SQL compilation
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import yaml

import spacy
from spacy.tokens import Token, Doc
from spacy.matcher import DependencyMatcher
from nltk.sem.logic import Expression

from . import ir_vocabulary as vocab
from . import deplambda_constructors as dlambda
from . import tree_transformations as transforms
from .question_analysis import QuestionAnalysis
from .schema_linking import SchemaLink
from .schema_graph import SchemaGraph


# ==========================
# TEMPLATE STRUCTURE
# ==========================

@dataclass
class CompositionTemplate:
    """A template for mapping dependency patterns to lambda constructors."""
    name: str
    priority: int
    pattern: List[Dict[str, Any]]  # spaCy DependencyMatcher format
    constructor: str  # Function name from ir_vocabulary
    description: str


# ==========================
# COMPOSITION ENGINE
# ==========================

class IRComposer:
    """
    Compositional semantics engine using template matching.

    Builds lambda expressions from spaCy dependency trees using
    pattern-based templates (ccg2lambda style).
    """

    def __init__(self, schema: SchemaGraph, nlp: spacy.Language):
        """
        Initialize composer.

        Args:
            schema: Database schema for grounding
            nlp: spaCy language model for parsing
        """
        self.schema = schema
        self.nlp = nlp
        self.templates: List[CompositionTemplate] = []
        self.matcher = DependencyMatcher(nlp.vocab)

        # Load templates
        self._load_templates()

    def _load_templates(self):
        """Load composition templates from YAML files."""
        # Load both SQL-specific and deplambda linguistic templates
        template_files = [
            "ir_templates.yaml",
            "deplambda_templates.yaml"
        ]

        total_loaded = 0
        for filename in template_files:
            template_path = Path(__file__).parent / filename

            if not template_path.exists():
                print(f"[WARN] Templates not found: {template_path}")
                continue

            with open(template_path, 'r') as f:
                template_data = yaml.safe_load(f)

            if not template_data:
                print(f"[WARN] No templates in {filename}")
                continue

            # Parse templates
            for tmpl in template_data:
                template = CompositionTemplate(
                    name=tmpl['name'],
                    priority=tmpl.get('priority', 50),
                    pattern=tmpl['pattern'],
                    constructor=tmpl['constructor'],
                    description=tmpl.get('description', '')
                )
                self.templates.append(template)

                # Register pattern with DependencyMatcher
                try:
                    self.matcher.add(template.name, [template.pattern])
                    total_loaded += 1
                except Exception as e:
                    print(f"[WARN] Failed to load template '{template.name}': {e}")

        # Sort templates by priority (higher first)
        self.templates.sort(key=lambda t: t.priority, reverse=True)

        print(f"  [OK] Loaded {total_loaded} composition templates")

    def compose(
        self,
        question: str,
        analysis: QuestionAnalysis,
        links: List[SchemaLink]
    ) -> Optional[Expression]:
        """
        Build lambda expression from question analysis.

        Args:
            question: Original English question
            analysis: Linguistic analysis from spaCy
            links: Schema links from schema linking stage

        Returns:
            Lambda expression or None if composition fails
        """
        # Parse question with spaCy if not already done
        if not hasattr(analysis, 'doc') or analysis.doc is None:
            doc = self.nlp(question)
        else:
            doc = analysis.doc

        # Apply deplambda tree transformations
        # TODO: Re-enable after fixing spaCy Doc immutability issues
        # doc = transforms.apply_all_transformations(doc)

        # Annotate tokens with schema links
        self._annotate_schema_links(doc, links)

        # Match templates bottom-up
        matches = self.matcher(doc)

        if not matches:
            # No patterns matched - try simple fallback
            return self._simple_fallback(analysis, links)

        # Build expression from highest-priority match
        for match_id, token_ids in matches:
            template_name = self.nlp.vocab.strings[match_id]
            template = self._get_template(template_name)

            if template:
                # Get constructor function from vocab or dlambda
                constructor_fn = getattr(vocab, template.constructor, None)
                if not constructor_fn:
                    constructor_fn = getattr(dlambda, template.constructor, None)
                if not constructor_fn:
                    continue

                # Extract matched tokens
                matched_tokens = [doc[tid] for tid in token_ids]

                # Build expression
                try:
                    expr = self._build_expression(
                        template,
                        constructor_fn,
                        matched_tokens,
                        doc,
                        links,
                        analysis
                    )
                    if expr:
                        # Add column projection if columns are explicitly mentioned
                        return self._add_projection_if_needed(expr, links, analysis)
                except Exception as e:
                    # Continue to next match
                    continue

        # No successful composition
        fallback_expr = self._simple_fallback(analysis, links)
        if fallback_expr:
            return self._add_projection_if_needed(fallback_expr, links, analysis)
        return None

    def _annotate_schema_links(self, doc: Doc, links: List[SchemaLink]):
        """
        Annotate spaCy tokens with schema linking information.

        Sets custom attributes:
        - token._.schema_type: 'table', 'column', or 'value'
        - token._.schema_ref: Table/column name or value
        """
        # Register custom attributes
        if not Token.has_extension('schema_type'):
            Token.set_extension('schema_type', default=None)
        if not Token.has_extension('schema_ref'):
            Token.set_extension('schema_ref', default=None)

        # Annotate tokens
        for link in links:
            # Find tokens matching the linked phrase
            phrase_lower = link.question_phrase.lower()
            for token in doc:
                if phrase_lower in token.text.lower() or token.text.lower() in phrase_lower:
                    # Set schema type
                    if link.link_type.value == 'table':
                        token._.schema_type = 'table'
                        token._.schema_ref = link.table_name
                    elif link.link_type.value == 'column':
                        token._.schema_type = 'column'
                        token._.schema_ref = (link.table_name, link.column_name)
                    elif link.link_type.value == 'value':
                        token._.schema_type = 'value'
                        token._.schema_ref = link.value

    def _get_template(self, name: str) -> Optional[CompositionTemplate]:
        """Get template by name."""
        for tmpl in self.templates:
            if tmpl.name == name:
                return tmpl
        return None

    def _build_expression(
        self,
        template: CompositionTemplate,
        constructor_fn: Callable,
        matched_tokens: List[Token],
        doc: Doc,
        links: List[SchemaLink],
        analysis: QuestionAnalysis
    ) -> Optional[Expression]:
        """
        Build lambda expression from matched template and tokens.

        Args:
            template: Matched template
            constructor_fn: Constructor function from ir_vocabulary
            matched_tokens: Tokens matched by pattern
            doc: Full spaCy doc
            links: Schema links
            analysis: Question analysis with question_type

        Returns:
            Lambda expression or None
        """
        # Extract arguments based on template type
        constructor_name = template.constructor

        # COUNT patterns
        if constructor_name == 'count':
            # Extract base set from matched tokens
            base_expr = self._extract_base_set(matched_tokens, links)
            if base_expr:
                return vocab.count(base_expr)

        # ARGMAX/ARGMIN patterns
        elif constructor_name in ('argmax', 'argmin'):
            # Check if this is actually an ORDER BY (all rows) vs superlative (LIMIT 1)
            # If question_type is 'order_by', use order_by combinators instead
            set_expr, measure_expr = self._extract_set_and_measure(matched_tokens, links)
            if set_expr and measure_expr:
                if analysis.question_type == 'order_by':
                    # ORDER BY all rows (no LIMIT)
                    is_desc = any(w in analysis.original_text.lower()
                                 for w in ['most', 'highest', 'largest', 'maximum', 'oldest', 'best', 'descending', 'decreasing'])
                    if is_desc:
                        return vocab.order_by_desc(set_expr, measure_expr)
                    else:
                        return vocab.order_by_asc(set_expr, measure_expr)
                else:
                    # True superlative (LIMIT 1)
                    if constructor_name == 'argmax':
                        return vocab.argmax(set_expr, measure_expr)
                    else:
                        return vocab.argmin(set_expr, measure_expr)

        # AGGREGATION patterns (avg, sum, min, max)
        elif constructor_name in ('avg', 'sum_', 'min_', 'max_'):
            set_expr, measure_expr = self._extract_set_and_measure(matched_tokens, links)
            if set_expr and measure_expr:
                return constructor_fn(set_expr, measure_expr)

        # FILTER patterns
        elif constructor_name == 'filter_':
            # Extract two set predicates
            pred1, pred2 = self._extract_two_predicates(matched_tokens, links)
            if pred1 and pred2:
                return vocab.filter_(pred1, pred2)

        # COMPARISON patterns
        elif constructor_name in ('gt', 'lt', 'gte', 'lte', 'eq', 'neq'):
            prop_expr, value_expr = self._extract_property_and_value(matched_tokens, links)
            if prop_expr and value_expr:
                return constructor_fn(prop_expr, value_expr)

        # TABLE/COLUMN atoms
        elif constructor_name == 'table_atom':
            for token in matched_tokens:
                if token._.schema_type == 'table':
                    return vocab.table_atom(token._.schema_ref)

        elif constructor_name == 'column_atom':
            for token in matched_tokens:
                if token._.schema_type == 'column':
                    table, column = token._.schema_ref
                    return vocab.column_atom(table, column)

        return None

    def _extract_base_set(self, tokens: List[Token], links: List[SchemaLink]) -> Optional[Expression]:
        """Extract base set predicate from tokens."""
        # Look for schema-linked table
        for token in tokens:
            if token._.schema_type == 'table':
                return vocab.table_atom(token._.schema_ref)

        # Fall back to first table link
        from .schema_linking import get_table_links
        table_links = get_table_links(links)
        if table_links:
            return vocab.table_atom(table_links[0].table_name)

        return None

    def _extract_set_and_measure(
        self,
        tokens: List[Token],
        links: List[SchemaLink]
    ) -> Tuple[Optional[Expression], Optional[Expression]]:
        """Extract set predicate and measurement function from tokens."""
        from .schema_linking import get_table_links, get_column_links

        table_links = get_table_links(links)
        col_links = get_column_links(links)

        set_expr = None
        measure_expr = None

        if table_links:
            set_expr = vocab.table_atom(table_links[0].table_name)

        # Try to find column linked to matched tokens first
        matched_token_texts = {t.text.lower() for t in tokens}
        for link in col_links:
            # Check if this column's phrase appears in matched tokens
            link_words = set(link.question_phrase.lower().split())
            if link_words & matched_token_texts:  # Intersection
                table = link.table_name
                column = link.column_name
                measure_expr = vocab.column_atom(table, column)
                break

        # Fallback to first column if no match
        if not measure_expr and col_links:
            table = col_links[0].table_name
            column = col_links[0].column_name
            measure_expr = vocab.column_atom(table, column)

        return set_expr, measure_expr

    def _extract_two_predicates(
        self,
        tokens: List[Token],
        links: List[SchemaLink]
    ) -> Tuple[Optional[Expression], Optional[Expression]]:
        """Extract two set predicates for filter operations."""
        # For now, return base set and True
        # TODO: Extract actual filter condition
        base_set = self._extract_base_set(tokens, links)
        # Second predicate needs more sophisticated extraction
        return base_set, base_set  # Placeholder

    def _extract_property_and_value(
        self,
        tokens: List[Token],
        links: List[SchemaLink]
    ) -> Tuple[Optional[Expression], Optional[Expression]]:
        """Extract property function and comparison value."""
        from .schema_linking import get_column_links

        col_links = get_column_links(links)
        prop_expr = None
        value_expr = None

        if col_links:
            table = col_links[0].table_name
            column = col_links[0].column_name
            prop_expr = vocab.column_atom(table, column)

        # Extract value from tokens
        for token in tokens:
            if token._.schema_type == 'value':
                value_expr = vocab.constant_atom(str(token._.schema_ref))
            elif token.like_num:
                value_expr = vocab.constant_atom(token.text, vocab.SQLType.NUMBER)

        return prop_expr, value_expr

    def _add_projection_if_needed(
        self,
        expr: Expression,
        links: List[SchemaLink],
        analysis: QuestionAnalysis
    ) -> Expression:
        """
        Wrap expression with PROJECT if columns are explicitly mentioned in question.

        Detects patterns like "Show name, country, age" and extracts column list.
        """
        from .schema_linking import get_column_links, get_table_links

        # Don't add projection to aggregation queries (COUNT, AVG, etc.)
        if vocab.is_aggregation(expr):
            return expr

        # Check if question starts with projection verbs
        question_lower = analysis.original_text.lower()
        has_projection_verb = any(
            question_lower.startswith(verb)
            for verb in ['show', 'list', 'display', 'give', 'find']  # Removed 'what is', 'what are'
        )

        if not has_projection_verb:
            return expr  # No projection needed

        # Get main table (first table link)
        table_links = get_table_links(links)
        if not table_links:
            return expr
        main_table = table_links[0].table_name

        # Extract high-scoring column links from main table only
        col_links = get_column_links(links)
        relevant_cols = [
            link for link in col_links
            if link.score >= 90.0  # Include near-exact matches
            and link.table_name == main_table  # Filter to main table only
        ]

        if not relevant_cols:
            return expr  # No columns to project

        # Sort by position in question (to preserve "Show name, country, age" order)
        # Use position of first character of phrase in question
        relevant_cols_with_pos = []
        for link in relevant_cols:
            pos = question_lower.find(link.question_phrase.lower())
            if pos >= 0:
                relevant_cols_with_pos.append((pos, link))

        relevant_cols_with_pos.sort(key=lambda x: x[0])  # Sort by position

        # Get unique column names (deduplicate by column name, keeping first occurrence)
        seen_cols = set()
        columns = []
        for pos, link in relevant_cols_with_pos:
            col_name = link.column_name
            if col_name not in seen_cols:
                seen_cols.add(col_name)
                columns.append(col_name)

        # Limit to first 10 columns (avoid projecting too many)
        columns = columns[:10]

        if columns:
            return vocab.project(expr, columns)

        return expr

    def _simple_fallback(
        self,
        analysis: QuestionAnalysis,
        links: List[SchemaLink]
    ) -> Optional[Expression]:
        """
        Simple fallback when no templates match.

        Uses SQL signals from question analysis.
        """
        from .schema_linking import get_table_links, get_column_links

        table_links = get_table_links(links)
        col_links = get_column_links(links)

        if not table_links:
            return None

        base_set = vocab.table_atom(table_links[0].table_name)

        # COUNT queries
        if 'COUNT' in analysis.sql_signals:
            return vocab.count(base_set)

        # ORDER BY (all rows, no LIMIT)
        if analysis.question_type == 'order_by' and col_links:
            measure = vocab.column_atom(col_links[0].table_name, col_links[0].column_name)
            # Determine direction
            is_desc = any(w in analysis.original_text.lower()
                         for w in ['most', 'highest', 'largest', 'maximum', 'oldest', 'best', 'descending', 'decreasing'])
            if is_desc:
                return vocab.order_by_desc(base_set, measure)
            else:
                return vocab.order_by_asc(base_set, measure)

        # SUPERLATIVE queries (with LIMIT 1)
        if analysis.question_type == 'superlative' and col_links:
            measure = vocab.column_atom(col_links[0].table_name, col_links[0].column_name)
            # Determine direction
            is_max = any(w in analysis.original_text.lower()
                        for w in ['most', 'highest', 'largest', 'maximum', 'oldest', 'best'])
            if is_max:
                return vocab.argmax(base_set, measure)
            else:
                return vocab.argmin(base_set, measure)

        # Default: simple select
        return base_set
