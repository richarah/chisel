"""
IR Compiler

Deterministic compilation from lambda expressions to SQL via sqlglot AST.

Architecture:
1. Pattern match on lambda expression structure
2. Emit sqlglot AST nodes for each pattern
3. Infer joins from multi-table references using schema FK graph
4. Compile nested expressions to subqueries
5. Return SQL string via sqlglot.generate()
"""

from typing import Optional, List, Set, Tuple
from nltk.sem.drt import DrtExpression, DrtApplicationExpression, DrtLambdaExpression
from nltk.sem.logic import Expression, ApplicationExpression, LambdaExpression

import sqlglot
from sqlglot import exp, parse_one, select

from .schema_graph import SchemaGraph
from . import ir_vocabulary as vocab


class IRCompiler:
    """
    Deterministic compiler from lambda expressions to SQL.

    Uses sqlglot for SQL AST construction and generation.
    """

    def __init__(self, schema: SchemaGraph):
        """
        Initialize compiler.

        Args:
            schema: Database schema with FK graph for join inference
        """
        self.schema = schema

    def compile(self, lambda_expr: DrtExpression, verbose: bool = False) -> Optional[str]:
        """
        Compile DRT expression to SQL.

        Args:
            lambda_expr: NLTK DRT expression (DRS)
            verbose: Print compilation steps

        Returns:
            SQL string or None if compilation fails
        """
        try:
            # Convert DRS to FOL for compilation
            if hasattr(lambda_expr, 'fol'):
                fol_expr = lambda_expr.fol()
                if verbose:
                    print(f"[IR Compiler] DRS: {lambda_expr}")
                    print(f"[IR Compiler] FOL: {fol_expr}")
            else:
                # Already FOL (fallback)
                fol_expr = lambda_expr

            # Convert to string for pattern matching
            expr_str = str(fol_expr)

            if verbose:
                print(f"[IR Compiler] Compiling: {expr_str}")

            # Pattern match on expression structure (on FOL, not DRS)
            sql_ast = self._compile_expression(fol_expr)

            if sql_ast is None:
                if verbose:
                    print(f"[IR Compiler] Failed to compile expression")
                return None

            # Generate SQL from AST
            sql = sql_ast.sql(dialect='sqlite')

            # Lowercase SQL keywords to match Spider gold standard
            sql = self._normalize_sql_case(sql)

            if verbose:
                print(f"[IR Compiler] Generated SQL: {sql}")

            return sql

        except Exception as e:
            if verbose:
                print(f"[IR Compiler] Error: {e}")
            return None

    def _compile_expression(self, expr: Expression) -> Optional[exp.Expression]:
        """
        Compile expression to sqlglot AST.

        Pattern matches on expression structure.
        """
        expr_str = str(expr)

        # PROJECT (column projection) - wrap base query
        if 'PROJECT' in expr_str:
            return self._compile_project(expr)

        # COUNT aggregation
        if 'COUNT' in expr_str:
            return self._compile_count(expr)

        # ORDER BY without LIMIT
        if 'ORDER_DESC' in expr_str:
            return self._compile_order_by(expr, desc=True)
        if 'ORDER_ASC' in expr_str:
            return self._compile_order_by(expr, desc=False)

        # ARGMAX/ARGMIN (superlatives with LIMIT)
        if 'ARGMAX' in expr_str:
            return self._compile_argmax(expr)
        if 'ARGMIN' in expr_str:
            return self._compile_argmin(expr)

        # AVG/SUM/MIN/MAX aggregations
        if 'AVG' in expr_str:
            return self._compile_agg(expr, 'AVG')
        if 'SUM' in expr_str:
            return self._compile_agg(expr, 'SUM')
        if 'MIN' in expr_str and 'ARGMIN' not in expr_str:
            return self._compile_agg(expr, 'MIN')
        if 'MAX' in expr_str and 'ARGMAX' not in expr_str:
            return self._compile_agg(expr, 'MAX')

        # GROUP BY
        if 'GROUP_BY' in expr_str:
            return self._compile_group_by(expr)

        # EXISTS/NOT EXISTS
        if 'EXISTS' in expr_str:
            return self._compile_exists(expr, negated='NOT_EXISTS' in expr_str)

        # Simple set query (SELECT * FROM table)
        if isinstance(expr, LambdaExpression):
            return self._compile_simple_select(expr)

        # Fallback
        return None

    def _compile_count(self, expr: Expression) -> Optional[exp.Select]:
        """
        Compile COUNT expression.

        COUNT(λx.table(x)) → SELECT COUNT(*) FROM table
        COUNT(λx.table(x) & filter(x)) → SELECT COUNT(*) FROM table WHERE filter
        """
        # Extract table from expression
        tables = self._extract_tables(expr)
        if not tables:
            return None

        # Build SELECT COUNT(*)
        query = select(exp.Count(this=exp.Star()))

        # FROM clause
        query = query.from_(tables[0])

        # WHERE clause (if filter present)
        where_cond = self._extract_where_condition(expr)
        if where_cond:
            query = query.where(where_cond)

        # JOIN clause (if multiple tables)
        if len(tables) > 1:
            query = self._add_joins(query, tables)

        return query

    def _compile_argmax(self, expr: Expression) -> Optional[exp.Select]:
        """
        Compile ARGMAX expression.

        ARGMAX(λx.table(x), λx.column(x)) → SELECT * FROM table ORDER BY column DESC LIMIT 1
        """
        tables = self._extract_tables(expr)
        column = self._extract_order_column(expr)

        if not tables:
            return None

        # Build SELECT *
        query = select(exp.Star())
        query = query.from_(tables[0])

        # ORDER BY DESC LIMIT 1
        if column:
            query = query.order_by(exp.Ordered(this=exp.Column(this=column), desc=True))
            query = query.limit(1)

        # JOIN if needed
        if len(tables) > 1:
            query = self._add_joins(query, tables)

        return query

    def _compile_argmin(self, expr: Expression) -> Optional[exp.Select]:
        """
        Compile ARGMIN expression.

        ARGMIN(λx.table(x), λx.column(x)) → SELECT * FROM table ORDER BY column ASC LIMIT 1
        """
        tables = self._extract_tables(expr)
        column = self._extract_order_column(expr)

        if not tables:
            return None

        query = select(exp.Star())
        query = query.from_(tables[0])

        if column:
            query = query.order_by(exp.Ordered(this=exp.Column(this=column), desc=False))
            query = query.limit(1)

        if len(tables) > 1:
            query = self._add_joins(query, tables)

        return query

    def _compile_order_by(self, expr: Expression, desc: bool = True) -> Optional[exp.Select]:
        """
        Compile ORDER BY expression (without LIMIT).

        ORDER_DESC(λx.table(x), λx.column(x)) → SELECT * FROM table ORDER BY column DESC
        ORDER_ASC(λx.table(x), λx.column(x)) → SELECT * FROM table ORDER BY column ASC
        """
        tables = self._extract_tables(expr)
        column = self._extract_order_column(expr)

        if not tables:
            return None

        query = select(exp.Star())
        query = query.from_(tables[0])

        if column:
            query = query.order_by(exp.Ordered(this=exp.Column(this=column), desc=desc))
            # No LIMIT - this is ORDER BY all rows

        if len(tables) > 1:
            query = self._add_joins(query, tables)

        return query

    def _compile_agg(self, expr: Expression, agg_func: str) -> Optional[exp.Select]:
        """
        Compile aggregation expression.

        AVG(λx.table(x), λx.column(x)) → SELECT AVG(column) FROM table
        """
        tables = self._extract_tables(expr)
        column = self._extract_agg_column(expr)

        if not tables or not column:
            return None

        # Build SELECT AGG(column)
        agg_exp = getattr(exp, agg_func)(this=exp.Column(this=column))
        query = select(agg_exp)
        query = query.from_(tables[0])

        # WHERE clause
        where_cond = self._extract_where_condition(expr)
        if where_cond:
            query = query.where(where_cond)

        # JOIN if needed
        if len(tables) > 1:
            query = self._add_joins(query, tables)

        return query

    def _compile_group_by(self, expr: Expression) -> Optional[exp.Select]:
        """
        Compile GROUP BY expression.

        GROUP_BY(λx.table(x), λx.group_col(x), COUNT) →
            SELECT group_col, COUNT(*) FROM table GROUP BY group_col
        """
        tables = self._extract_tables(expr)
        group_col = self._extract_group_column(expr)

        if not tables or not group_col:
            return None

        # Build SELECT group_col, COUNT(*)
        query = select(
            exp.Column(this=group_col),
            exp.Count(this=exp.Star())
        )
        query = query.from_(tables[0])
        query = query.group_by(exp.Column(this=group_col))

        return query

    def _compile_exists(self, expr: Expression, negated: bool = False) -> Optional[exp.Select]:
        """
        Compile EXISTS/NOT EXISTS expression.

        EXISTS(λx.table(x)) → WHERE EXISTS (SELECT * FROM table)
        """
        # This is a WHERE clause predicate, not a full query
        # Return None for now - needs context from outer query
        return None

    def _compile_simple_select(self, expr: LambdaExpression) -> Optional[exp.Select]:
        """
        Compile simple SELECT expression.

        λx.table(x) → SELECT * FROM table
        """
        tables = self._extract_tables(expr)
        if not tables:
            return None

        query = select(exp.Star())
        query = query.from_(tables[0])

        # Check for filters in lambda body
        where_cond = self._extract_where_condition(expr)
        if where_cond:
            query = query.where(where_cond)

        return query

    def _compile_project(self, expr: Expression) -> Optional[exp.Select]:
        """
        Compile PROJECT expression (column projection).

        PROJECT(inner_expr, COLUMNS_col1,col2,col3) → SELECT col1, col2, col3 FROM ...
        """
        expr_str = str(expr)

        # Extract column list from COLUMNS_xxx constant
        import re
        columns_match = re.search(r'COLUMNS_([a-z_,]+)', expr_str)
        if not columns_match:
            return None

        columns_str = columns_match.group(1)
        columns = columns_str.split(',')

        # Extract inner expression (remove PROJECT wrapper)
        # The inner expression is the first argument to PROJECT
        # Format: PROJECT(inner)(COLUMNS_...)
        # We need to compile the inner expression first

        # Remove PROJECT and COLUMNS parts to get inner expression
        inner_str = re.sub(r'PROJECT\((.*)\)\(COLUMNS_[^)]+\)', r'\1', expr_str)

        # Recursively compile inner expression
        try:
            # Parse inner expression back to Expression object
            from nltk.sem import logic
            inner_expr = logic.Expression.fromstring(inner_str)
            base_query = self._compile_expression(inner_expr)
        except:
            # Fallback: compile without PROJECT
            # Extract what's inside PROJECT(...)(COLUMNS_...)
            base_query = None
            if 'ORDER_DESC' in expr_str:
                base_query = self._compile_order_by(expr, desc=True)
            elif 'ORDER_ASC' in expr_str:
                base_query = self._compile_order_by(expr, desc=False)
            elif 'ARGMAX' in expr_str:
                base_query = self._compile_argmax(expr)
            elif 'ARGMIN' in expr_str:
                base_query = self._compile_argmin(expr)
            else:
                # Simple select
                base_query = self._compile_simple_select(expr)

        if not base_query:
            return None

        # Replace SELECT * with SELECT col1, col2, col3
        column_exps = [exp.Column(this=col) for col in columns]
        base_query.args['expressions'] = column_exps

        return base_query

    # ==========================
    # SQL NORMALIZATION
    # ==========================

    def _normalize_sql_case(self, sql: str) -> str:
        """
        Normalize SQL to match Spider gold standard case convention.

        Spider uses:
        - UPPERCASE: SELECT, FROM, WHERE, ORDER BY, JOIN, etc. (structural keywords)
        - lowercase: count, sum, avg, min, max (aggregate functions)
        """
        import re

        # Structural keywords -> UPPERCASE
        structural_keywords = [
            'select', 'from', 'where', 'order by', 'group by', 'having',
            'limit', 'offset', 'join', 'inner join', 'left join', 'right join',
            'on', 'as', 'and', 'or', 'not', 'in', 'like', 'between',
            'asc', 'desc', 'is', 'null', 'exists',
            'union', 'intersect', 'except', 'case', 'when', 'then', 'else', 'end',
            'distinct'
        ]

        # Aggregate functions -> lowercase
        functions = ['count', 'sum', 'avg', 'min', 'max']

        # Uppercase structural keywords
        for keyword in structural_keywords:
            pattern = r'\b' + keyword + r'\b'
            sql = re.sub(pattern, keyword.upper(), sql, flags=re.IGNORECASE)

        # Lowercase aggregate functions
        for func in functions:
            pattern = r'\b' + func + r'\b'
            sql = re.sub(pattern, func.lower(), sql, flags=re.IGNORECASE)

        return sql

    # ==========================
    # EXTRACTION UTILITIES
    # ==========================

    def _extract_tables(self, expr: Expression) -> List[str]:
        """
        Extract table names from expression.

        Uses vocabulary.py's extract_table_refs().
        """
        return vocab.extract_table_refs(expr)

    def _extract_where_condition(self, expr: Expression) -> Optional[exp.Expression]:
        """
        Extract WHERE clause condition from expression.

        Looks for comparison operators (GT, LT, EQ, etc.)
        """
        expr_str = str(expr)

        # Check for comparison operators
        if 'GT(' in expr_str:
            return self._build_comparison(expr, '>')
        if 'LT(' in expr_str:
            return self._build_comparison(expr, '<')
        if 'GTE(' in expr_str:
            return self._build_comparison(expr, '>=')
        if 'LTE(' in expr_str:
            return self._build_comparison(expr, '<=')
        if '=' in expr_str and 'ARGMAX' not in expr_str:
            return self._build_comparison(expr, '=')

        return None

    def _build_comparison(self, expr: Expression, op: str) -> Optional[exp.Expression]:
        """Build comparison expression for WHERE clause."""
        # Extract column and value
        # Simplified - needs proper AST traversal
        return None

    def _extract_order_column(self, expr: Expression) -> Optional[str]:
        """Extract ORDER BY column from ARGMAX/ARGMIN expression."""
        expr_str = str(expr)

        # Look for column references (format: table_column)
        import re
        pattern = r'\\x\.([a-z_]+)\(x\)'
        matches = re.findall(pattern, expr_str)

        # Return last match (typically the measure function)
        if matches:
            # Extract column name (after last underscore)
            col_ref = matches[-1]
            if '_' in col_ref:
                parts = col_ref.split('_')
                return parts[-1]  # Return column name
        return None

    def _extract_agg_column(self, expr: Expression) -> Optional[str]:
        """Extract aggregation column."""
        return self._extract_order_column(expr)

    def _extract_group_column(self, expr: Expression) -> Optional[str]:
        """Extract GROUP BY column."""
        return self._extract_order_column(expr)

    def _add_joins(self, query: exp.Select, tables: List[str]) -> exp.Select:
        """
        Add JOIN clauses using schema FK graph.

        Uses networkx shortest path to find join sequence.
        """
        if len(tables) < 2:
            return query

        # Find join path using schema's FK graph
        join_conditions = self.schema.find_join_path(tables)

        for join_cond in join_conditions:
            right_table = join_cond['right_table']
            left_col = join_cond['left_col']
            right_col = join_cond['right_col']

            # Add JOIN clause
            on_cond = exp.EQ(
                this=exp.Column(this=left_col),
                expression=exp.Column(this=right_col)
            )

            query = query.join(
                exp.Table(this=right_table),
                on=on_cond,
                kind='INNER'
            )

        return query
