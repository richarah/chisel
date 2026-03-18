"""
Task 5: Slot Filling

Fill the SQL skeleton with actual tables, columns, and values.
Build the final SQL query using sqlglot's AST builder.

Libraries doing the heavy lifting:
- sqlglot: SQL AST construction, validation, SQL generation
- networkx: Already computed join paths in schema_graph

We write: Rules to map linked schema elements -> SQL AST nodes
"""

from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Tuple
import sqlglot
from sqlglot import exp, parse_one

from .schema_graph import SchemaGraph
from .question_analysis import QuestionAnalysis
from .schema_linking import SchemaLink, get_column_links, get_table_links, get_value_links, get_tables_from_links
from .skeleton_prediction import SQLSkeleton


@dataclass
class FilledSQL:
    """
    Filled SQL query with all components.
    """
    select_columns: List[exp.Expression]
    from_table: Optional[str]
    join_clauses: List[Dict]  # List of join info
    where_conditions: List[exp.Expression]
    group_by_columns: List[exp.Expression]
    having_conditions: List[exp.Expression]
    order_by_columns: List[Tuple[exp.Expression, bool]]  # [(col, is_desc), ...]
    limit_value: Optional[int]
    distinct: bool = False

    def to_sql(self) -> str:
        """
        Build SQL string using sqlglot AST.
        sqlglot does all the hard work of generating valid SQL.
        """
        # Build SELECT expressions list
        if self.select_columns:
            expressions = self.select_columns
        else:
            # Default: SELECT *
            expressions = [exp.Star()]

        # Build FROM clause
        from_clause = None
        if self.from_table:
            from_clause = exp.From(this=exp.Table(this=self.from_table))

        # Build WHERE clause
        where_clause = None
        if self.where_conditions:
            combined_where = self.where_conditions[0]
            for cond in self.where_conditions[1:]:
                combined_where = exp.And(this=combined_where, expression=cond)
            where_clause = exp.Where(this=combined_where)

        # Build GROUP BY clause
        group_clause = None
        if self.group_by_columns:
            group_clause = exp.Group(expressions=self.group_by_columns)

        # Build HAVING clause
        having_clause = None
        if self.having_conditions:
            combined_having = self.having_conditions[0]
            for cond in self.having_conditions[1:]:
                combined_having = exp.And(this=combined_having, expression=cond)
            having_clause = exp.Having(this=combined_having)

        # Build ORDER BY clause
        order_clause = None
        if self.order_by_columns:
            ordered_exprs = [
                exp.Ordered(this=col_exp, desc=is_desc)
                for col_exp, is_desc in self.order_by_columns
            ]
            order_clause = exp.Order(expressions=ordered_exprs)

        # Build LIMIT clause
        limit_clause = None
        if self.limit_value:
            limit_clause = exp.Limit(expression=exp.Literal.number(self.limit_value))

        # Build DISTINCT
        distinct_clause = exp.Distinct() if self.distinct else None

        # Create SELECT using constructor with named parameters
        select = exp.Select(
            expressions=expressions,
            from_=from_clause,
            where=where_clause,
            group=group_clause,
            having=having_clause,
            order=order_clause,
            limit=limit_clause,
            distinct=distinct_clause
        )

        # Add JOINs (must use append after construction)
        if self.join_clauses and from_clause:
            for join_info in self.join_clauses:
                right_table = exp.Table(this=join_info["right_table"])

                # Build ON condition: left.col = right.col
                left_col = exp.Column(
                    this=join_info["left_col"],
                    table=join_info["left_table"]
                )
                right_col = exp.Column(
                    this=join_info["right_col"],
                    table=join_info["right_table"]
                )
                on_condition = exp.EQ(this=left_col, expression=right_col)

                join = exp.Join(
                    this=right_table,
                    on=on_condition,
                    kind="INNER"
                )
                select.append("joins", join)

        # Generate SQL string (sqlglot does this)
        return select.sql()


def fill_sql_skeleton(
    skeleton: SQLSkeleton,
    analysis: QuestionAnalysis,
    links: List[SchemaLink],
    schema: SchemaGraph
) -> Optional[FilledSQL]:
    """
    Fill SQL skeleton with actual schema elements.

    This is where we map:
    - Linked columns -> SELECT clause
    - Linked tables -> FROM clause
    - Schema graph -> JOIN clauses
    - Value links -> WHERE clause
    - Aggregation signals -> COUNT/MAX/MIN/etc

    Returns: FilledSQL or None if can't construct valid query
    """
    filled = FilledSQL(
        select_columns=[],
        from_table=None,
        join_clauses=[],
        where_conditions=[],
        group_by_columns=[],
        having_conditions=[],
        order_by_columns=[],
        limit_value=skeleton.limit_value,
        distinct=skeleton.select_distinct
    )

    # Get linked elements by type
    column_links = get_column_links(links)
    table_links = get_table_links(links)
    value_links = get_value_links(links)

    # Determine which tables are involved
    # Strategy: Use minimal set of tables that covers all required columns
    involved_tables = set()

    # First priority: explicit high-confidence table links (e.g., "from singer")
    high_conf_tables = [link for link in table_links if link.score >= 85]
    if high_conf_tables:
        # Use single table if high confidence
        involved_tables.add(high_conf_tables[0].table_name)

    # Second priority: tables from high-confidence column links
    if not involved_tables:
        high_conf_columns = [link for link in column_links if link.score >= 85]
        if high_conf_columns:
            # Use primary table from highest scoring column
            primary_table = high_conf_columns[0].table_name
            involved_tables.add(primary_table)

            # Add other tables only if columns from them are explicitly needed
            for link in high_conf_columns[1:]:
                if link.table_name != primary_table:
                    # Only add if this is a different high-confidence match
                    if link.score >= 90:
                        involved_tables.add(link.table_name)

    # Fallback: use any linked tables
    if not involved_tables:
        if table_links:
            involved_tables.add(table_links[0].table_name)
        elif column_links:
            involved_tables.add(column_links[0].table_name)

    if not involved_tables:
        # No tables identified - can't build query
        return None

    # Sort tables for deterministic ordering
    involved_tables = sorted(list(involved_tables))

    # ==========================
    # FROM CLAUSE
    # ==========================
    filled.from_table = involved_tables[0]

    # ==========================
    # JOIN CLAUSES
    # ==========================
    if len(involved_tables) > 1:
        # Use schema graph to find join path (networkx does this)
        join_path = schema.find_join_path(involved_tables)
        filled.join_clauses = join_path

    # ==========================
    # SELECT CLAUSE
    # ==========================
    if skeleton.use_count:
        # COUNT(*)
        count_exp = exp.Count(this=exp.Star())
        filled.select_columns.append(count_exp)

    elif skeleton.use_max or skeleton.use_min or skeleton.use_avg or skeleton.use_sum:
        # Aggregation on a column
        # Find the most relevant column link
        if column_links:
            col_link = column_links[0]  # Highest scored
            col_exp = exp.Column(
                this=exp.Identifier(this=col_link.column_name),
                table=exp.Identifier(this=col_link.table_name)
            )

            if skeleton.use_max:
                agg_exp = exp.Max(this=col_exp)
            elif skeleton.use_min:
                agg_exp = exp.Min(this=col_exp)
            elif skeleton.use_avg:
                agg_exp = exp.Avg(this=col_exp)
            elif skeleton.use_sum:
                agg_exp = exp.Sum(this=col_exp)

            filled.select_columns.append(agg_exp)
        else:
            # No column found - fallback to COUNT(*)
            filled.select_columns.append(exp.Count(this=exp.Star()))

    else:
        # Regular SELECT - add relevant columns
        if column_links:
            # Filter to high-confidence column links from involved tables
            relevant_columns = [
                link for link in column_links
                if link.table_name in involved_tables and link.score >= 80
            ]

            if relevant_columns:
                # Use top 5 high-confidence columns
                for col_link in relevant_columns[:5]:
                    col_exp = exp.Column(
                        this=exp.Identifier(this=col_link.column_name),
                        table=exp.Identifier(this=col_link.table_name)
                    )
                    filled.select_columns.append(col_exp)
            else:
                # No high-confidence columns - SELECT *
                filled.select_columns = []
        else:
            # No specific columns - SELECT *
            filled.select_columns = []

    # ==========================
    # WHERE CLAUSE
    # ==========================
    # Build WHERE conditions from value links and comparison signals
    for value_link in value_links:
        # Try to match this value to a column
        # Simple heuristic: use first column link from same table
        target_column = None

        if column_links:
            # Find column link that makes sense for this value
            for col_link in column_links:
                # Check if column type matches value type
                col_info = schema.get_column(col_link.table_name, col_link.column_name)
                if col_info:
                    # String value -> text column
                    if isinstance(value_link.value, str):
                        if col_info.data_type in ["text", "varchar", "char"]:
                            target_column = col_link
                            break
                    # Numeric value -> number column
                    else:
                        if col_info.data_type in ["number", "int", "integer", "float", "real"]:
                            target_column = col_link
                            break

            # If no type match, just use first column
            if not target_column:
                target_column = column_links[0]

        if target_column:
            col_exp = exp.Column(
                this=exp.Identifier(this=target_column.column_name),
                table=exp.Identifier(this=target_column.table_name)
            )

            # Build value expression
            if isinstance(value_link.value, str):
                val_exp = exp.Literal.string(value_link.value)
            else:
                val_exp = exp.Literal.number(value_link.value)

            # Determine comparison operator from signals
            if "WHERE_GT" in analysis.sql_signals:
                condition = exp.GT(this=col_exp, expression=val_exp)
            elif "WHERE_LT" in analysis.sql_signals:
                condition = exp.LT(this=col_exp, expression=val_exp)
            elif "WHERE_LIKE" in analysis.sql_signals:
                condition = exp.Like(this=col_exp, expression=val_exp)
            else:
                # Default: equality
                condition = exp.EQ(this=col_exp, expression=val_exp)

            filled.where_conditions.append(condition)

    # ==========================
    # GROUP BY CLAUSE
    # ==========================
    if skeleton.needs_group_by and filled.select_columns:
        # GROUP BY non-aggregated columns in SELECT
        for col_exp in filled.select_columns:
            # Check if it's an aggregation
            if not isinstance(col_exp, (exp.Count, exp.Max, exp.Min, exp.Avg, exp.Sum)):
                filled.group_by_columns.append(col_exp)

    # ==========================
    # ORDER BY CLAUSE
    # ==========================
    if skeleton.needs_order_by:
        # Order by first column in SELECT
        if filled.select_columns:
            order_col = filled.select_columns[0]
            filled.order_by_columns.append((order_col, skeleton.order_desc))
        elif column_links:
            # Use first column link
            col_link = column_links[0]
            col_exp = exp.Column(
                this=exp.Identifier(this=col_link.column_name),
                table=exp.Identifier(this=col_link.table_name)
            )
            filled.order_by_columns.append((col_exp, skeleton.order_desc))

    return filled


if __name__ == "__main__":
    # Test SQL generation
    from .schema_graph import SchemaGraph
    from .question_analysis import analyze_question
    from .schema_linking import link_question_to_schema
    from .skeleton_prediction import predict_skeleton

    # Sample schema
    sample_db = {
        "db_id": "test_db",
        "table_names_original": ["Student", "Course", "Enrollment"],
        "column_names_original": [
            [-1, "*"],
            [0, "student_id"],
            [0, "name"],
            [0, "age"],
            [1, "course_id"],
            [1, "title"],
            [2, "enrollment_id"],
            [2, "student_id"],
            [2, "course_id"]
        ],
        "column_types": ["text", "number", "text", "number", "number", "text", "number", "number", "number"],
        "primary_keys": [1, 4, 6],
        "foreign_keys": [[7, 1], [8, 4]]
    }

    schema = SchemaGraph.from_spider_json(sample_db)

    # Test questions
    test_questions = [
        "How many students are there?",
        "What are the names of all students?",
    ]

    for q in test_questions:
        print(f"\nQuestion: {q}")
        analysis = analyze_question(q)
        links = link_question_to_schema(analysis, schema)
        skeleton = predict_skeleton(analysis)

        filled = fill_sql_skeleton(skeleton, analysis, links, schema)

        if filled:
            sql = filled.to_sql()
            print(f"SQL: {sql}")
        else:
            print("Could not generate SQL")
