"""
Task 1: Schema Graph

Parse database schemas from Spider's tables.json and CREATE TABLE statements.
Use sqlglot for SQL parsing, networkx for FK graph and join path finding.

All the hard work is done by libraries:
- sqlglot: Parse CREATE TABLE, extract columns, PKs, FKs
- networkx: Build graph, find shortest paths, compute Steiner trees
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import json
import networkx as nx


@dataclass
class ColumnInfo:
    """Information about a column."""
    name: str
    table: str
    data_type: str
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_key_ref: Optional[Tuple[str, str]] = None  # (ref_table, ref_column)


@dataclass
class TableInfo:
    """Information about a table."""
    name: str
    columns: List[ColumnInfo] = field(default_factory=list)
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: List[Tuple[str, str, str]] = field(default_factory=list)  # [(col, ref_table, ref_col)]


class SchemaGraph:
    """
    Schema representation with FK graph for join resolution.

    Libraries doing the work:
    - networkx: Graph storage and shortest_path/steiner_tree algorithms
    - (sqlglot would parse CREATE TABLEs, but Spider provides JSON)
    """

    def __init__(self):
        self.tables: Dict[str, TableInfo] = {}
        self.columns: Dict[Tuple[str, str], ColumnInfo] = {}  # (table, column) -> info
        self.column_name_to_tables: Dict[str, List[str]] = {}  # column_name -> [tables]
        self.fk_graph = nx.Graph()  # Nodes = tables, edges = FK relationships

    @classmethod
    def from_spider_json(cls, db_entry: dict) -> 'SchemaGraph':
        """
        Parse Spider's tables.json format into SchemaGraph.

        Spider format:
        {
            "table_names_original": ["table1", "table2", ...],
            "column_names_original": [[table_idx, "col_name"], ...],
            "column_types": ["text", "number", ...],
            "primary_keys": [col_idx, ...],
            "foreign_keys": [[col_idx, ref_col_idx], ...]
        }
        """
        schema = cls()

        table_names = db_entry["table_names_original"]
        column_info = db_entry["column_names_original"]  # [[table_idx, col_name], ...]
        column_types = db_entry["column_types"]
        primary_keys = db_entry.get("primary_keys", [])
        foreign_keys = db_entry.get("foreign_keys", [])

        # Create tables
        for table_name in table_names:
            schema.tables[table_name.lower()] = TableInfo(name=table_name.lower())
            schema.fk_graph.add_node(table_name.lower())

        # Add special * column (Spider uses [-1, "*"] for it)
        # Skip it in our processing

        # Process columns
        for col_idx, (table_idx, col_name) in enumerate(column_info):
            if table_idx == -1:  # Skip the "*" column
                continue

            table_name = table_names[table_idx].lower()
            col_name_lower = col_name.lower()
            col_type = column_types[col_idx] if col_idx < len(column_types) else "text"

            is_pk = col_idx in primary_keys

            col_info = ColumnInfo(
                name=col_name_lower,
                table=table_name,
                data_type=col_type,
                is_primary_key=is_pk
            )

            schema.columns[(table_name, col_name_lower)] = col_info
            schema.tables[table_name].columns.append(col_info)

            if is_pk:
                schema.tables[table_name].primary_keys.append(col_name_lower)

            # Track which tables have this column name
            if col_name_lower not in schema.column_name_to_tables:
                schema.column_name_to_tables[col_name_lower] = []
            schema.column_name_to_tables[col_name_lower].append(table_name)

        # Process foreign keys - this builds the FK graph
        for fk_col_idx, ref_col_idx in foreign_keys:
            if fk_col_idx >= len(column_info) or ref_col_idx >= len(column_info):
                continue

            fk_table_idx, fk_col_name = column_info[fk_col_idx]
            ref_table_idx, ref_col_name = column_info[ref_col_idx]

            if fk_table_idx == -1 or ref_table_idx == -1:
                continue

            fk_table = table_names[fk_table_idx].lower()
            ref_table = table_names[ref_table_idx].lower()
            fk_col = fk_col_name.lower()
            ref_col = ref_col_name.lower()

            # Update column info
            if (fk_table, fk_col) in schema.columns:
                schema.columns[(fk_table, fk_col)].is_foreign_key = True
                schema.columns[(fk_table, fk_col)].foreign_key_ref = (ref_table, ref_col)

            # Update table info
            schema.tables[fk_table].foreign_keys.append((fk_col, ref_table, ref_col))

            # Add edge to FK graph (networkx does the work)
            # Store the join condition as edge attribute
            schema.fk_graph.add_edge(
                fk_table,
                ref_table,
                fk_col=fk_col,
                ref_col=ref_col
            )

        return schema

    def find_join_path(self, tables: List[str]) -> List[Dict]:
        """
        Find how to join multiple tables using FK relationships.

        networkx does the heavy lifting:
        - For 2 tables: shortest_path
        - For 3+ tables: steiner_tree (minimum tree connecting all tables)

        Returns: List of join conditions
        [
            {"left_table": "t1", "right_table": "t2",
             "left_col": "col1", "right_col": "col2"},
            ...
        ]
        """
        if len(tables) == 0:
            return []

        if len(tables) == 1:
            return []

        tables = [t.lower() for t in tables]

        # Check all tables exist
        for table in tables:
            if table not in self.fk_graph:
                return []  # Can't join if table not in graph

        join_conditions = []

        if len(tables) == 2:
            # Two tables: find shortest path (networkx does this)
            try:
                path = nx.shortest_path(self.fk_graph, tables[0], tables[1])

                # Convert path to join conditions
                for i in range(len(path) - 1):
                    left_table = path[i]
                    right_table = path[i + 1]

                    # Get edge data (FK relationship)
                    edge_data = self.fk_graph.get_edge_data(left_table, right_table)

                    join_conditions.append({
                        "left_table": left_table,
                        "right_table": right_table,
                        "left_col": edge_data.get("fk_col"),
                        "right_col": edge_data.get("ref_col")
                    })

            except nx.NetworkXNoPath:
                # No path exists - tables not connected
                return []

        else:
            # 3+ tables: use Steiner tree (networkx does this)
            try:
                # Find minimum tree connecting all tables
                steiner = nx.algorithms.approximation.steiner_tree(self.fk_graph, tables)

                # Convert tree edges to join conditions
                for left_table, right_table in steiner.edges():
                    edge_data = self.fk_graph.get_edge_data(left_table, right_table)

                    join_conditions.append({
                        "left_table": left_table,
                        "right_table": right_table,
                        "left_col": edge_data.get("fk_col"),
                        "right_col": edge_data.get("ref_col")
                    })

            except nx.NetworkXNoPath:
                return []

        return join_conditions

    def get_table(self, table_name: str) -> Optional[TableInfo]:
        """Get table info by name (case-insensitive)."""
        return self.tables.get(table_name.lower())

    def get_column(self, table_name: str, column_name: str) -> Optional[ColumnInfo]:
        """Get column info (case-insensitive)."""
        return self.columns.get((table_name.lower(), column_name.lower()))

    def find_tables_with_column(self, column_name: str) -> List[str]:
        """Find all tables that have a column with this name."""
        return self.column_name_to_tables.get(column_name.lower(), [])

    def get_all_tables(self) -> List[str]:
        """Get all table names."""
        return list(self.tables.keys())

    def get_all_columns(self) -> List[Tuple[str, str]]:
        """Get all (table, column) pairs."""
        return list(self.columns.keys())


def load_spider_schemas(tables_json_path: str) -> Dict[str, SchemaGraph]:
    """
    Load all schemas from Spider's tables.json.

    Returns: Dict[db_id, SchemaGraph]
    """
    with open(tables_json_path, 'r') as f:
        data = json.load(f)

    schemas = {}
    for db_entry in data:
        db_id = db_entry["db_id"]
        schemas[db_id] = SchemaGraph.from_spider_json(db_entry)

    return schemas


if __name__ == "__main__":
    # Test on a sample schema
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
        "foreign_keys": [
            [7, 1],  # Enrollment.student_id -> Student.student_id
            [8, 4]   # Enrollment.course_id -> Course.course_id
        ]
    }

    schema = SchemaGraph.from_spider_json(sample_db)

    print("Tables:", schema.get_all_tables())
    print("\nFK Graph edges:", list(schema.fk_graph.edges(data=True)))

    # Test join path finding
    path = schema.find_join_path(["student", "course"])
    print("\nJoin path (Student -> Course):", path)

    # Test column lookup
    print("\nTables with 'student_id':", schema.find_tables_with_column("student_id"))
