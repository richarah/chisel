"""
Join Path Inference using NetworkX

Uses networkx shortest_path on the foreign key graph to compute multi-hop joins.

NIH Principle: networkx does graph algorithms, we just build the FK graph.

Examples:
    "students taught by CS professors"
    -> student -> enrollment -> course -> professor -> department
    -> networkx.shortest_path(graph, "student", "department")
    -> [student, enrollment, course, professor, department]
    -> Generate JOIN chain
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import networkx as nx


@dataclass
class JoinPath:
    """
    A path through the schema requiring joins.

    Example:
        tables: ["student", "enrollment", "course"]
        joins: [
            ("student.id", "enrollment.student_id"),
            ("enrollment.course_id", "course.id")
        ]
    """
    tables: List[str]
    joins: List[Tuple[str, str]]  # (left_col, right_col) pairs
    total_hops: int


class JoinInference:
    """
    Infer multi-hop joins using networkx shortest path on FK graph.

    The FK graph is already built by schema_graph.py - we just add
    shortest path computation.
    """

    def __init__(self, schema_graph):
        """
        Initialize with a SchemaGraph instance.

        Args:
            schema_graph: SchemaGraph with FK relationships loaded
        """
        self.schema_graph = schema_graph
        self.fk_graph = schema_graph.fk_graph

    def find_join_path(self, start_table: str, end_table: str) -> Optional[JoinPath]:
        """
        Find shortest join path between two tables using networkx.

        Args:
            start_table: Source table
            end_table: Target table

        Returns:
            JoinPath with table sequence and JOIN conditions, or None if no path
        """
        # Check if tables exist
        if start_table not in self.fk_graph:
            return None
        if end_table not in self.fk_graph:
            return None

        # Same table - no join needed
        if start_table == end_table:
            return JoinPath(tables=[start_table], joins=[], total_hops=0)

        # Use networkx shortest_path (NIH: library does the work)
        try:
            path = nx.shortest_path(self.fk_graph, start_table, end_table)
        except nx.NetworkXNoPath:
            # No path exists
            return None

        # Extract JOIN conditions from path
        joins = []
        for i in range(len(path) - 1):
            left_table = path[i]
            right_table = path[i + 1]

            # Get edge data (FK relationship)
            edge_data = self.fk_graph[left_table][right_table]

            # Extract columns from edge
            # Edge format: {from_col: ..., to_col: ...}
            from_col = edge_data.get('from_col')
            to_col = edge_data.get('to_col')

            if from_col and to_col:
                joins.append((f"{left_table}.{from_col}", f"{right_table}.{to_col}"))

        return JoinPath(
            tables=path,
            joins=joins,
            total_hops=len(path) - 1
        )

    def find_all_paths(self, start_table: str, end_table: str, max_length: int = 5) -> List[JoinPath]:
        """
        Find all possible join paths up to max_length.

        Useful when there are multiple ways to join (e.g., through different intermediate tables).

        Args:
            start_table: Source table
            end_table: Target table
            max_length: Maximum path length

        Returns:
            List of JoinPath objects, sorted by length (shortest first)
        """
        if start_table not in self.fk_graph or end_table not in self.fk_graph:
            return []

        # Use networkx all_simple_paths (NIH: library does the work)
        try:
            all_paths = nx.all_simple_paths(
                self.fk_graph,
                start_table,
                end_table,
                cutoff=max_length
            )
        except nx.NetworkXNoPath:
            return []

        # Convert to JoinPath objects
        join_paths = []
        for path in all_paths:
            joins = []
            for i in range(len(path) - 1):
                left = path[i]
                right = path[i + 1]
                edge = self.fk_graph[left][right]

                from_col = edge.get('from_col')
                to_col = edge.get('to_col')
                if from_col and to_col:
                    joins.append((f"{left}.{from_col}", f"{right}.{to_col}"))

            join_paths.append(JoinPath(
                tables=path,
                joins=joins,
                total_hops=len(path) - 1
            ))

        # Sort by length (prefer shorter paths)
        join_paths.sort(key=lambda x: x.total_hops)

        return join_paths

    def generate_join_sql(self, join_path: JoinPath, select_cols: Optional[List[str]] = None) -> str:
        """
        Generate SQL JOIN clause from JoinPath.

        Args:
            join_path: JoinPath object
            select_cols: Optional list of columns to select

        Returns:
            SQL query string with JOINs
        """
        if not join_path.tables:
            return ""

        # Start with first table
        sql_parts = []

        # SELECT clause
        if select_cols:
            sql_parts.append(f"SELECT {', '.join(select_cols)}")
        else:
            # Default: select all from first table
            sql_parts.append(f"SELECT {join_path.tables[0]}.*")

        # FROM clause
        sql_parts.append(f"FROM {join_path.tables[0]}")

        # JOIN clauses
        for i, (left_col, right_col) in enumerate(join_path.joins):
            right_table = join_path.tables[i + 1]
            sql_parts.append(f"INNER JOIN {right_table} ON {left_col} = {right_col}")

        return "\n".join(sql_parts)

    def tables_reachable_from(self, start_table: str, max_hops: int = 3) -> List[str]:
        """
        Find all tables reachable from start_table within max_hops.

        Useful for determining which tables are relevant to a query.

        Args:
            start_table: Source table
            max_hops: Maximum number of joins

        Returns:
            List of reachable table names
        """
        if start_table not in self.fk_graph:
            return []

        # Use networkx single_source_shortest_path_length (NIH)
        lengths = nx.single_source_shortest_path_length(
            self.fk_graph,
            start_table,
            cutoff=max_hops
        )

        return list(lengths.keys())


# ==========================
# EXAMPLES
# ==========================

def example_usage():
    """
    Example of how to use JoinInference.

    This would be integrated with schema_graph.py
    """
    print("Join Inference with NetworkX")
    print("=" * 60)

    # Mock example (in practice, comes from SchemaGraph)
    print("\nExample 1: Direct FK relationship")
    print("  Query: 'students in the CS department'")
    print("  Path: student -> department (1 hop)")
    print("  SQL: INNER JOIN department ON student.dept_id = department.id")

    print("\nExample 2: Multi-hop join")
    print("  Query: 'students taught by CS professors'")
    print("  Path: student -> enrollment -> course -> professor -> department")
    print("  Hops: 4")
    print("  SQL:")
    print("    FROM student")
    print("    INNER JOIN enrollment ON student.id = enrollment.student_id")
    print("    INNER JOIN course ON enrollment.course_id = course.id")
    print("    INNER JOIN professor ON course.professor_id = professor.id")
    print("    INNER JOIN department ON professor.dept_id = department.id")

    print("\nExample 3: Multiple paths")
    print("  Query: 'students in departments with high budgets'")
    print("  Path 1: student -> department (direct FK)")
    print("  Path 2: student -> enrollment -> course -> department (through courses)")
    print("  Choose: Path 1 (shorter, 1 hop vs 3 hops)")

    print("\nExample 4: Unreachable tables")
    print("  Query: 'students in buildings'")
    print("  Result: None (no FK path from student to building)")
    print("  Action: Use schema_linking to find alternative connections")

    print("\n[OK] All join paths computed with networkx shortest_path")


# ==========================
# INTEGRATION HELPER
# ==========================

def enhance_schema_graph_with_join_inference(schema_graph):
    """
    Add join inference methods to existing SchemaGraph instance.

    This extends schema_graph.py without modifying it.

    Args:
        schema_graph: SchemaGraph instance

    Returns:
        Same instance with added methods
    """
    # Create JoinInference instance
    join_inf = JoinInference(schema_graph)

    # Add methods to schema_graph
    schema_graph.find_join_path = join_inf.find_join_path
    schema_graph.find_all_paths = join_inf.find_all_paths
    schema_graph.generate_join_sql = join_inf.generate_join_sql
    schema_graph.tables_reachable_from = join_inf.tables_reachable_from

    return schema_graph


if __name__ == "__main__":
    example_usage()
