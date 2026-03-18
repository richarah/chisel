"""
IR Compiler (Refactored with sqlglot Builder API)

Deterministic compilation from DRT expressions to SQL using sqlglot's builder API.

Architecture:
1. Accept DRT expression, call .fol() to get FOL
2. Pattern match on FOL expression structure
3. Use sqlglot builder API (select().from_().where()...) to construct queries
4. Infer joins from multi-table references using schema FK graph
5. Return SQL string with configurable dialect

Benefits of builder API:
- Less verbose than manual AST construction
- sqlglot parses and validates string fragments
- Cleaner, more readable code
- Dialect-independent by design
"""

from typing import Optional, List
from nltk.sem.drt import DrtExpression
from nltk.sem.logic import Expression

from sqlglot import select

from .schema_graph import SchemaGraph
from . import ir_vocabulary as vocab


class IRCompiler:
    """
    Deterministic compiler from DRT expressions to SQL.

    Uses sqlglot builder API for clean, readable SQL construction.
    """

    def __init__(self, schema: SchemaGraph):
        """
        Initialize compiler.

        Args:
            schema: Database schema with FK graph for join inference
        """
        self.schema = schema

    def compile(
        self,
        lambda_expr: DrtExpression,
        verbose: bool = False,
        dialect: str = 'sqlite'
    ) -> Optional[str]:
        """
        Compile DRT expression to SQL.

        Args:
            lambda_expr: NLTK DRT expression (DRS)
            verbose: Print compilation steps
            dialect: SQL dialect (sqlite, duckdb, postgres, mysql, etc.)

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

            # Build query using sqlglot builder API
            query = self._build_query(fol_expr, expr_str)

            if query is None:
                if verbose:
                    print(f"[IR Compiler] Failed to compile expression")
                return None

            # Generate SQL for target dialect
            sql = query.sql(dialect=dialect)

            # Normalize case to match Spider conventions
            sql = self._normalize_sql_case(sql)

            if verbose:
                print(f"[IR Compiler] Generated SQL: {sql}")

            return sql

        except Exception as e:
            if verbose:
                print(f"[IR Compiler] Error: {e}")
            return None

    def _build_query(self, expr: Expression, expr_str: str):
        """
        Build sqlglot query using builder API.

        Walks FOL expression and chains builder calls.
        """
        # PROJECT (column projection)
        if 'PROJECT' in expr_str:
            return self._build_project(expr, expr_str)

        # COUNT aggregation
        if 'COUNT' in expr_str:
            return self._build_count(expr, expr_str)

        # ORDER BY (all rows, no LIMIT)
        if 'ORDER_DESC' in expr_str:
            return self._build_order_by(expr, expr_str, desc=True)
        if 'ORDER_ASC' in expr_str:
            return self._build_order_by(expr, expr_str, desc=False)

        # ARGMAX/ARGMIN (superlatives with LIMIT 1)
        if 'ARGMAX' in expr_str:
            return self._build_superlative(expr, expr_str, desc=True)
        if 'ARGMIN' in expr_str:
            return self._build_superlative(expr, expr_str, desc=False)

        # AVG/SUM/MIN/MAX aggregations
        for agg_func in ['AVG', 'SUM', 'MIN', 'MAX']:
            if agg_func in expr_str:
                # Check it's not ARGMIN/ARGMAX
                if agg_func == 'MIN' and 'ARGMIN' in expr_str:
                    continue
                if agg_func == 'MAX' and 'ARGMAX' in expr_str:
                    continue
                return self._build_agg(expr, expr_str, agg_func)

        # GROUP BY
        if 'GROUP_BY' in expr_str:
            return self._build_group_by(expr, expr_str)

        # EXISTS/NOT EXISTS (subquery in WHERE)
        if 'EXISTS' in expr_str:
            # This needs context from outer query
            return None

        # Simple SELECT
        return self._build_simple_select(expr, expr_str)

    def _build_count(self, expr: Expression, expr_str: str):
        """
        Build COUNT query.

        COUNT(λx.table(x)) → SELECT COUNT(*) FROM table
        """
        tables = vocab.extract_table_refs(expr)
        if not tables:
            return None

        # Start with SELECT COUNT(*)
        query = select("COUNT(*)")

        # FROM clause
        query = query.from_(tables[0])

        # WHERE clause
        where_cond = self._extract_where_str(expr_str)
        if where_cond:
            query = query.where(where_cond)

        # JOIN if multiple tables
        if len(tables) > 1:
            query = self._add_joins(query, tables)

        return query

    def _build_superlative(self, expr: Expression, expr_str: str, desc: bool):
        """
        Build ARGMAX/ARGMIN query.

        ARGMAX(λx.table(x), λx.col(x)) → SELECT * FROM table ORDER BY col DESC LIMIT 1
        """
        tables = vocab.extract_table_refs(expr)
        column = self._extract_column(expr_str)

        if not tables:
            return None

        # SELECT *
        query = select("*").from_(tables[0])

        # ORDER BY col DESC/ASC LIMIT 1
        if column:
            direction = "DESC" if desc else "ASC"
            query = query.order_by(f"{column} {direction}").limit(1)

        # JOIN if needed
        if len(tables) > 1:
            query = self._add_joins(query, tables)

        return query

    def _build_order_by(self, expr: Expression, expr_str: str, desc: bool):
        """
        Build ORDER BY query (all rows, no LIMIT).

        ORDER_DESC(λx.table(x), λx.col(x)) → SELECT * FROM table ORDER BY col DESC
        """
        tables = vocab.extract_table_refs(expr)
        column = self._extract_column(expr_str)

        if not tables:
            return None

        query = select("*").from_(tables[0])

        if column:
            direction = "DESC" if desc else "ASC"
            query = query.order_by(f"{column} {direction}")

        if len(tables) > 1:
            query = self._add_joins(query, tables)

        return query

    def _build_agg(self, expr: Expression, expr_str: str, agg_func: str):
        """
        Build aggregation query.

        AVG(λx.table(x), λx.col(x)) → SELECT AVG(col) FROM table
        """
        tables = vocab.extract_table_refs(expr)
        column = self._extract_column(expr_str)

        if not tables or not column:
            return None

        # SELECT AGG(column)
        query = select(f"{agg_func}({column})")
        query = query.from_(tables[0])

        # WHERE clause
        where_cond = self._extract_where_str(expr_str)
        if where_cond:
            query = query.where(where_cond)

        # JOIN if needed
        if len(tables) > 1:
            query = self._add_joins(query, tables)

        return query

    def _build_group_by(self, expr: Expression, expr_str: str):
        """
        Build GROUP BY query.

        GROUP_BY(λx.table(x), λx.group_col(x), COUNT) →
            SELECT group_col, COUNT(*) FROM table GROUP BY group_col
        """
        tables = vocab.extract_table_refs(expr)
        group_col = self._extract_column(expr_str)

        if not tables or not group_col:
            return None

        # SELECT group_col, COUNT(*)
        query = select(group_col, "COUNT(*)")
        query = query.from_(tables[0])
        query = query.group_by(group_col)

        return query

    def _build_simple_select(self, expr: Expression, expr_str: str):
        """
        Build simple SELECT query.

        λx.table(x) → SELECT * FROM table
        """
        tables = vocab.extract_table_refs(expr)
        if not tables:
            return None

        query = select("*").from_(tables[0])

        # WHERE clause
        where_cond = self._extract_where_str(expr_str)
        if where_cond:
            query = query.where(where_cond)

        # JOIN if multiple tables
        if len(tables) > 1:
            query = self._add_joins(query, tables)

        return query

    def _build_project(self, expr: Expression, expr_str: str):
        """
        Build PROJECT query (column projection).

        PROJECT(inner_expr, COLUMNS_col1,col2) → SELECT col1, col2 FROM ...
        """
        import re

        # Extract column list
        columns_match = re.search(r'COLUMNS_([a-z_,]+)', expr_str)
        if not columns_match:
            return None

        columns_str = columns_match.group(1)
        columns = columns_str.split(',')

        # Build base query (remove PROJECT wrapper)
        # Recursively compile inner expression
        if 'ORDER_DESC' in expr_str:
            base_query = self._build_order_by(expr, expr_str, desc=True)
        elif 'ORDER_ASC' in expr_str:
            base_query = self._build_order_by(expr, expr_str, desc=False)
        elif 'ARGMAX' in expr_str:
            base_query = self._build_superlative(expr, expr_str, desc=True)
        elif 'ARGMIN' in expr_str:
            base_query = self._build_superlative(expr, expr_str, desc=False)
        else:
            base_query = self._build_simple_select(expr, expr_str)

        if not base_query:
            return None

        # Replace SELECT * with SELECT col1, col2, ...
        # sqlglot builder: rebuild with specific columns
        tables = vocab.extract_table_refs(expr)
        if not tables:
            return None

        query = select(*columns).from_(tables[0])

        # Copy WHERE, ORDER BY, LIMIT from base query
        if hasattr(base_query, 'args'):
            if 'where' in base_query.args:
                query.args['where'] = base_query.args['where']
            if 'order' in base_query.args:
                query.args['order'] = base_query.args['order']
            if 'limit' in base_query.args:
                query.args['limit'] = base_query.args['limit']
            if 'joins' in base_query.args:
                query.args['joins'] = base_query.args['joins']

        return query

    # ==========================
    # SQL NORMALIZATION
    # ==========================

    def _normalize_sql_case(self, sql: str) -> str:
        """
        Normalize SQL to match Spider gold standard case convention.

        Spider uses:
        - UPPERCASE: SELECT, FROM, WHERE, ORDER BY, JOIN, etc.
        - lowercase: count, sum, avg, min, max
        """
        import re

        # Structural keywords -> UPPERCASE
        keywords = [
            'select', 'from', 'where', 'order by', 'group by', 'having',
            'limit', 'offset', 'join', 'inner join', 'left join', 'right join',
            'on', 'as', 'and', 'or', 'not', 'in', 'like', 'between',
            'asc', 'desc', 'is', 'null', 'exists', 'nulls last', 'nulls first',
            'union', 'intersect', 'except', 'distinct'
        ]

        for keyword in keywords:
            pattern = r'\b' + keyword + r'\b'
            sql = re.sub(pattern, keyword.upper(), sql, flags=re.IGNORECASE)

        # Aggregate functions -> lowercase
        for func in ['count', 'sum', 'avg', 'min', 'max']:
            pattern = r'\b' + func + r'\b'
            sql = re.sub(pattern, func.lower(), sql, flags=re.IGNORECASE)

        return sql

    # ==========================
    # EXTRACTION UTILITIES
    # ==========================

    def _extract_column(self, expr_str: str) -> Optional[str]:
        """Extract column name from expression string."""
        import re

        # Look for table_column pattern
        pattern = r'\\x\.([a-z_]+)\(x\)'
        matches = re.findall(pattern, expr_str)

        if matches:
            # Return column name (after last underscore)
            col_ref = matches[-1]
            if '_' in col_ref:
                parts = col_ref.split('_')
                return parts[-1]
        return None

    def _extract_where_str(self, expr_str: str) -> Optional[str]:
        """Extract WHERE condition as SQL string."""
        # Check for comparison operators
        if 'GT(' in expr_str:
            return self._build_comparison_str(expr_str, '>')
        if 'LT(' in expr_str:
            return self._build_comparison_str(expr_str, '<')
        if 'GTE(' in expr_str:
            return self._build_comparison_str(expr_str, '>=')
        if 'LTE(' in expr_str:
            return self._build_comparison_str(expr_str, '<=')
        if '=' in expr_str and 'ARGMAX' not in expr_str:
            return self._build_comparison_str(expr_str, '=')

        return None

    def _build_comparison_str(self, expr_str: str, op: str) -> Optional[str]:
        """Build WHERE comparison as SQL string."""
        import re

        # Extract column and value
        # Simplified pattern matching
        col_pattern = r'([a-z_]+)\('
        matches = re.findall(col_pattern, expr_str)

        # Find column (not a keyword)
        keywords = {'and', 'or', 'not', 'count', 'avg', 'sum', 'min', 'max', 'gt', 'lt', 'gte', 'lte'}
        column = None
        for match in matches:
            if '_' in match and match not in keywords:
                parts = match.split('_')
                if len(parts) >= 2:
                    column = parts[-1]
                    break

        # Extract value (look for number or quoted string)
        value_match = re.search(r"['\"]?(\d+|[A-Za-z]+)['\"]?(?=\))", expr_str)
        value = value_match.group(1) if value_match else None

        if column and value:
            # Check if value is numeric
            if value.isdigit():
                return f"{column} {op} {value}"
            else:
                return f"{column} {op} '{value}'"

        return None

    def _add_joins(self, query, tables: List[str]):
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

            # Use builder API to add JOIN
            on_condition = f"{left_col} = {right_col}"
            query = query.join(right_table, on=on_condition, kind='INNER')

        return query
