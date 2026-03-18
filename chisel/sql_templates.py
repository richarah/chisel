"""
SQL Template Library (v0.1)

Pattern-based SQL generation for common and rare SQL constructs.

Approach:
1. Extract patterns from Spider training data (deterministic)
2. Match question patterns to SQL templates using spaCy
3. Handle rare constructs (self-joins, nested aggregations, INTERSECT/EXCEPT)

Philosophy: Templates encode SQL expertise without ML
"""

from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Tuple, Any
from enum import Enum
import re

# Local imports
from .schema_graph import SchemaGraph


# ==========================
# TEMPLATE DEFINITIONS
# ==========================

@dataclass
class SQLTemplate:
    """
    A reusable SQL pattern with placeholders.

    Example:
        pattern: "How many {entity} are there"
        template: "SELECT COUNT(*) FROM {table}"
        slots: ["entity", "table"]
    """
    template_id: str
    description: str
    pattern: str  # Natural language pattern
    sql_template: str  # SQL template with {placeholders}
    slots: List[str]  # Required slot names
    complexity: str  # 'simple', 'medium', 'complex'
    examples: List[Tuple[str, str]]  # (question, sql) pairs


class SQLComplexity(Enum):
    """SQL query complexity levels."""
    SIMPLE = "simple"  # Single table, no joins
    MEDIUM = "medium"  # Joins, basic aggregation
    COMPLEX = "complex"  # Nested queries, multiple aggregations
    EXPERT = "expert"  # Rare constructs (self-joins, INTERSECT, etc.)


# ==========================
# PREDEFINED TEMPLATE LIBRARY
# ==========================

# Simple Templates (70-80% accuracy expected)
SIMPLE_TEMPLATES = [
    SQLTemplate(
        template_id="count_all",
        description="Count all rows in a table",
        pattern="how many {entity}",
        sql_template="SELECT COUNT(*) FROM {table}",
        slots=["entity", "table"],
        complexity="simple",
        examples=[
            ("How many students are there?", "SELECT COUNT(*) FROM student"),
            ("How many courses?", "SELECT COUNT(*) FROM course")
        ]
    ),
    SQLTemplate(
        template_id="select_all",
        description="Select all rows from table",
        pattern="list all {entity}|show all {entity}|what are the {entity}",
        sql_template="SELECT * FROM {table}",
        slots=["entity", "table"],
        complexity="simple",
        examples=[
            ("List all students", "SELECT * FROM student"),
            ("Show all courses", "SELECT * FROM course")
        ]
    ),
    SQLTemplate(
        template_id="select_column",
        description="Select specific column",
        pattern="what {column} {entity}|list {column} of {entity}",
        sql_template="SELECT {column} FROM {table}",
        slots=["column", "entity", "table"],
        complexity="simple",
        examples=[
            ("What are the names of students?", "SELECT name FROM student"),
            ("List titles of courses", "SELECT title FROM course")
        ]
    ),
    SQLTemplate(
        template_id="where_equals",
        description="Filter by equality",
        pattern="{column} is {value}|where {column} equals {value}",
        sql_template="SELECT * FROM {table} WHERE {column} = {value}",
        slots=["column", "table", "value"],
        complexity="simple",
        examples=[
            ("Students whose age is 20", "SELECT * FROM student WHERE age = 20"),
            ("Courses where department is 'CS'", "SELECT * FROM course WHERE dept = 'CS'")
        ]
    ),
]

# Aggregation Templates (60-70% accuracy)
AGGREGATION_TEMPLATES = [
    SQLTemplate(
        template_id="max_column",
        description="Maximum value in column",
        pattern="maximum {column}|highest {column}|largest {column}",
        sql_template="SELECT MAX({column}) FROM {table}",
        slots=["column", "table"],
        complexity="simple",
        examples=[
            ("What is the maximum age?", "SELECT MAX(age) FROM student"),
            ("Highest salary", "SELECT MAX(salary) FROM employee")
        ]
    ),
    SQLTemplate(
        template_id="min_column",
        description="Minimum value in column",
        pattern="minimum {column}|lowest {column}|smallest {column}",
        sql_template="SELECT MIN({column}) FROM {table}",
        slots=["column", "table"],
        complexity="simple",
        examples=[
            ("What is the minimum age?", "SELECT MIN(age) FROM student"),
            ("Lowest price", "SELECT MIN(price) FROM product")
        ]
    ),
    SQLTemplate(
        template_id="avg_column",
        description="Average value",
        pattern="average {column}|mean {column}",
        sql_template="SELECT AVG({column}) FROM {table}",
        slots=["column", "table"],
        complexity="simple",
        examples=[
            ("Average age of students", "SELECT AVG(age) FROM student"),
            ("Mean salary", "SELECT AVG(salary) FROM employee")
        ]
    ),
    SQLTemplate(
        template_id="group_by_count",
        description="Count by group",
        pattern="how many {entity} in each {group}|count {entity} by {group}",
        sql_template="SELECT {group_col}, COUNT(*) FROM {table} GROUP BY {group_col}",
        slots=["entity", "group", "group_col", "table"],
        complexity="medium",
        examples=[
            ("How many students in each department?",
             "SELECT dept_id, COUNT(*) FROM student GROUP BY dept_id"),
        ]
    ),
]

# Superlative Templates (50-60% accuracy)
SUPERLATIVE_TEMPLATES = [
    SQLTemplate(
        template_id="argmax_simple",
        description="Entity with maximum value",
        pattern="{entity} with (the )?(highest|largest|maximum) {attribute}",
        sql_template="SELECT * FROM {table} ORDER BY {column} DESC LIMIT 1",
        slots=["entity", "attribute", "table", "column"],
        complexity="medium",
        examples=[
            ("Student with the highest age",
             "SELECT * FROM student ORDER BY age DESC LIMIT 1"),
            ("Course with largest enrollment",
             "SELECT * FROM course ORDER BY enrollment DESC LIMIT 1")
        ]
    ),
    SQLTemplate(
        template_id="argmin_simple",
        description="Entity with minimum value",
        pattern="{entity} with (the )?(lowest|smallest|minimum) {attribute}",
        sql_template="SELECT * FROM {table} ORDER BY {column} ASC LIMIT 1",
        slots=["entity", "attribute", "table", "column"],
        complexity="medium",
        examples=[
            ("Student with the lowest age",
             "SELECT * FROM student ORDER BY age ASC LIMIT 1"),
        ]
    ),
    SQLTemplate(
        template_id="top_k",
        description="Top K entities",
        pattern="top {k} {entity}|{k} {entity} with (highest|most) {attribute}",
        sql_template="SELECT * FROM {table} ORDER BY {column} DESC LIMIT {k}",
        slots=["k", "entity", "table", "column", "attribute"],
        complexity="medium",
        examples=[
            ("Top 5 students with highest GPA",
             "SELECT * FROM student ORDER BY gpa DESC LIMIT 5"),
        ]
    ),
]

# Join Templates (50-60% accuracy)
JOIN_TEMPLATES = [
    SQLTemplate(
        template_id="simple_join",
        description="Two-table join",
        pattern="{entity1} and their {entity2}|{entity2} of {entity1}",
        sql_template="SELECT * FROM {table1} JOIN {table2} ON {table1}.{fk} = {table2}.{pk}",
        slots=["entity1", "entity2", "table1", "table2", "fk", "pk"],
        complexity="medium",
        examples=[
            ("Students and their departments",
             "SELECT * FROM student JOIN department ON student.dept_id = department.id"),
        ]
    ),
    SQLTemplate(
        template_id="join_with_filter",
        description="Join with WHERE clause",
        pattern="{entity1} in {entity2} where {condition}",
        sql_template="SELECT t1.* FROM {table1} t1 JOIN {table2} t2 ON t1.{fk} = t2.{pk} WHERE {condition}",
        slots=["entity1", "entity2", "table1", "table2", "fk", "pk", "condition"],
        complexity="medium",
        examples=[
            ("Students in CS department",
             "SELECT s.* FROM student s JOIN department d ON s.dept_id = d.id WHERE d.name = 'CS'"),
        ]
    ),
]

# Complex Templates (30-40% accuracy)
COMPLEX_TEMPLATES = [
    SQLTemplate(
        template_id="nested_aggregation",
        description="Aggregation in subquery",
        pattern="{entity} with {attribute} (greater|less) than average",
        sql_template="SELECT * FROM {table} WHERE {column} > (SELECT AVG({column}) FROM {table})",
        slots=["entity", "attribute", "table", "column"],
        complexity="complex",
        examples=[
            ("Students with age greater than average",
             "SELECT * FROM student WHERE age > (SELECT AVG(age) FROM student)"),
        ]
    ),
    SQLTemplate(
        template_id="not_exists",
        description="Negation with NOT EXISTS",
        pattern="{entity1} (who|that) (do not|don't|never) {verb} {entity2}",
        sql_template="SELECT * FROM {table1} WHERE NOT EXISTS (SELECT 1 FROM {table2} WHERE {condition})",
        slots=["entity1", "entity2", "table1", "table2", "verb", "condition"],
        complexity="complex",
        examples=[
            ("Students who do not take any courses",
             "SELECT * FROM student s WHERE NOT EXISTS (SELECT 1 FROM enrollment e WHERE e.student_id = s.id)"),
        ]
    ),
    SQLTemplate(
        template_id="self_join",
        description="Self-join for comparisons",
        pattern="{entity} with same {attribute} as {other_entity}",
        sql_template="SELECT t1.* FROM {table} t1 JOIN {table} t2 ON t1.{column} = t2.{column} WHERE t2.{condition}",
        slots=["entity", "attribute", "table", "column", "other_entity", "condition"],
        complexity="expert",
        examples=[
            ("Students with same age as John",
             "SELECT s1.* FROM student s1 JOIN student s2 ON s1.age = s2.age WHERE s2.name = 'John'"),
        ]
    ),
]

# Set Operation Templates (30-40% accuracy)
SET_TEMPLATES = [
    SQLTemplate(
        template_id="intersect",
        description="Common elements",
        pattern="{entity} that are both {condition1} and {condition2}",
        sql_template="SELECT * FROM {table} WHERE {condition1} INTERSECT SELECT * FROM {table} WHERE {condition2}",
        slots=["entity", "table", "condition1", "condition2"],
        complexity="complex",
        examples=[
            ("Students who take both CS101 and CS102",
             "SELECT student_id FROM enrollment WHERE course_id = 'CS101' INTERSECT SELECT student_id FROM enrollment WHERE course_id = 'CS102'"),
        ]
    ),
    SQLTemplate(
        template_id="except",
        description="Set difference",
        pattern="{entity} that {condition1} but not {condition2}",
        sql_template="SELECT * FROM {table} WHERE {condition1} EXCEPT SELECT * FROM {table} WHERE {condition2}",
        slots=["entity", "table", "condition1", "condition2"],
        complexity="complex",
        examples=[
            ("Students who take CS101 but not CS102",
             "SELECT student_id FROM enrollment WHERE course_id = 'CS101' EXCEPT SELECT student_id FROM enrollment WHERE course_id = 'CS102'"),
        ]
    ),
]

# Additional Comparison Templates (40-50% accuracy)
COMPARISON_TEMPLATES = [
    SQLTemplate(
        template_id="greater_than",
        description="Greater than comparison",
        pattern="{entity} (with|where) {attribute} (greater than|more than|above|over) {value}",
        sql_template="SELECT * FROM {table} WHERE {column} > {value}",
        slots=["entity", "attribute", "table", "column", "value"],
        complexity="simple",
        examples=[
            ("Students with age greater than 20", "SELECT * FROM student WHERE age > 20"),
        ]
    ),
    SQLTemplate(
        template_id="less_than",
        description="Less than comparison",
        pattern="{entity} (with|where) {attribute} (less than|fewer than|below|under) {value}",
        sql_template="SELECT * FROM {table} WHERE {column} < {value}",
        slots=["entity", "attribute", "table", "column", "value"],
        complexity="simple",
        examples=[
            ("Students with age less than 25", "SELECT * FROM student WHERE age < 25"),
        ]
    ),
    SQLTemplate(
        template_id="between",
        description="Between two values",
        pattern="{entity} (with|where) {attribute} between {value1} and {value2}",
        sql_template="SELECT * FROM {table} WHERE {column} BETWEEN {value1} AND {value2}",
        slots=["entity", "attribute", "table", "column", "value1", "value2"],
        complexity="simple",
        examples=[
            ("Students with age between 18 and 25", "SELECT * FROM student WHERE age BETWEEN 18 AND 25"),
        ]
    ),
    SQLTemplate(
        template_id="like_pattern",
        description="String pattern matching",
        pattern="{entity} (with|where) {attribute} (contains|includes|like) {value}",
        sql_template="SELECT * FROM {table} WHERE {column} LIKE '%{value}%'",
        slots=["entity", "attribute", "table", "column", "value"],
        complexity="simple",
        examples=[
            ("Students with name containing John", "SELECT * FROM student WHERE name LIKE '%John%'"),
        ]
    ),
    SQLTemplate(
        template_id="starts_with",
        description="String starts with",
        pattern="{entity} (with|where) {attribute} (starts with|begins with) {value}",
        sql_template="SELECT * FROM {table} WHERE {column} LIKE '{value}%'",
        slots=["entity", "attribute", "table", "column", "value"],
        complexity="simple",
        examples=[
            ("Students whose name starts with A", "SELECT * FROM student WHERE name LIKE 'A%'"),
        ]
    ),
    SQLTemplate(
        template_id="in_list",
        description="Value in list",
        pattern="{entity} (with|where) {attribute} (in|among) {values}",
        sql_template="SELECT * FROM {table} WHERE {column} IN ({values})",
        slots=["entity", "attribute", "table", "column", "values"],
        complexity="simple",
        examples=[
            ("Students in CS or EE department", "SELECT * FROM student WHERE dept IN ('CS', 'EE')"),
        ]
    ),
]

# Additional Aggregation Templates
EXTENDED_AGGREGATION_TEMPLATES = [
    SQLTemplate(
        template_id="sum_column",
        description="Sum of column values",
        pattern="(total|sum of) {attribute}",
        sql_template="SELECT SUM({column}) FROM {table}",
        slots=["attribute", "table", "column"],
        complexity="simple",
        examples=[
            ("Total salary", "SELECT SUM(salary) FROM employee"),
        ]
    ),
    SQLTemplate(
        template_id="count_distinct",
        description="Count distinct values",
        pattern="(how many|number of) (different|distinct|unique) {attribute}",
        sql_template="SELECT COUNT(DISTINCT {column}) FROM {table}",
        slots=["attribute", "table", "column"],
        complexity="simple",
        examples=[
            ("How many different departments", "SELECT COUNT(DISTINCT dept_id) FROM student"),
        ]
    ),
    SQLTemplate(
        template_id="having_count",
        description="Filter groups by count",
        pattern="{entity} with (more than|at least|over) {n} {related}",
        sql_template="SELECT {group_col}, COUNT(*) FROM {table} GROUP BY {group_col} HAVING COUNT(*) > {n}",
        slots=["entity", "n", "related", "group_col", "table"],
        complexity="complex",
        examples=[
            ("Departments with more than 5 students",
             "SELECT dept_id, COUNT(*) FROM student GROUP BY dept_id HAVING COUNT(*) > 5"),
        ]
    ),
    SQLTemplate(
        template_id="group_by_multiple",
        description="Group by multiple columns",
        pattern="{entity} by {group1} and {group2}",
        sql_template="SELECT {col1}, {col2}, COUNT(*) FROM {table} GROUP BY {col1}, {col2}",
        slots=["entity", "group1", "group2", "col1", "col2", "table"],
        complexity="medium",
        examples=[
            ("Students by department and year",
             "SELECT dept_id, year, COUNT(*) FROM student GROUP BY dept_id, year"),
        ]
    ),
]

# Additional Join Templates
EXTENDED_JOIN_TEMPLATES = [
    SQLTemplate(
        template_id="three_table_join",
        description="Three-table join",
        pattern="{entity1} and their {entity2} and {entity3}",
        sql_template="SELECT * FROM {table1} t1 JOIN {table2} t2 ON t1.{fk1} = t2.{pk1} JOIN {table3} t3 ON t2.{fk2} = t3.{pk2}",
        slots=["entity1", "entity2", "entity3", "table1", "table2", "table3", "fk1", "pk1", "fk2", "pk2"],
        complexity="complex",
        examples=[
            ("Students and their enrollments and courses",
             "SELECT * FROM student s JOIN enrollment e ON s.id = e.student_id JOIN course c ON e.course_id = c.id"),
        ]
    ),
    SQLTemplate(
        template_id="left_join",
        description="Left outer join (include all from left)",
        pattern="all {entity1} (including|even) (those without|with no) {entity2}",
        sql_template="SELECT * FROM {table1} LEFT JOIN {table2} ON {table1}.{fk} = {table2}.{pk}",
        slots=["entity1", "entity2", "table1", "table2", "fk", "pk"],
        complexity="medium",
        examples=[
            ("All students including those without enrollments",
             "SELECT * FROM student LEFT JOIN enrollment ON student.id = enrollment.student_id"),
        ]
    ),
    SQLTemplate(
        template_id="join_aggregation",
        description="Join with aggregation",
        pattern="(how many|count) {entity1} for each {entity2}",
        sql_template="SELECT t2.{col}, COUNT(t1.id) FROM {table1} t1 JOIN {table2} t2 ON t1.{fk} = t2.{pk} GROUP BY t2.{col}",
        slots=["entity1", "entity2", "col", "table1", "table2", "fk", "pk"],
        complexity="complex",
        examples=[
            ("How many students for each department",
             "SELECT d.name, COUNT(s.id) FROM student s JOIN department d ON s.dept_id = d.id GROUP BY d.name"),
        ]
    ),
]

# Additional Complex Templates
EXTENDED_COMPLEX_TEMPLATES = [
    SQLTemplate(
        template_id="correlated_subquery",
        description="Correlated subquery",
        pattern="{entity1} that have {entity2} where {condition}",
        sql_template="SELECT * FROM {table1} t1 WHERE EXISTS (SELECT 1 FROM {table2} t2 WHERE t2.{fk} = t1.{pk} AND {condition})",
        slots=["entity1", "entity2", "table1", "table2", "fk", "pk", "condition"],
        complexity="expert",
        examples=[
            ("Students who have enrollments where grade is A",
             "SELECT * FROM student s WHERE EXISTS (SELECT 1 FROM enrollment e WHERE e.student_id = s.id AND e.grade = 'A')"),
        ]
    ),
    SQLTemplate(
        template_id="union",
        description="Union of two queries",
        pattern="{entity} that {condition1} or {condition2}",
        sql_template="SELECT * FROM {table} WHERE {condition1} UNION SELECT * FROM {table} WHERE {condition2}",
        slots=["entity", "table", "condition1", "condition2"],
        complexity="complex",
        examples=[
            ("Students in CS or with GPA above 3.5",
             "SELECT * FROM student WHERE dept = 'CS' UNION SELECT * FROM student WHERE gpa > 3.5"),
        ]
    ),
    SQLTemplate(
        template_id="case_when",
        description="Conditional expressions",
        pattern="{entity} categorized by {attribute}",
        sql_template="SELECT *, CASE WHEN {column} {op1} {val1} THEN '{cat1}' ELSE '{cat2}' END AS category FROM {table}",
        slots=["entity", "attribute", "column", "op1", "val1", "cat1", "cat2", "table"],
        complexity="complex",
        examples=[
            ("Students categorized by age",
             "SELECT *, CASE WHEN age < 20 THEN 'young' ELSE 'adult' END AS category FROM student"),
        ]
    ),
    SQLTemplate(
        template_id="window_function",
        description="Ranking with window functions",
        pattern="rank {entity} by {attribute}",
        sql_template="SELECT *, RANK() OVER (ORDER BY {column} DESC) AS rank FROM {table}",
        slots=["entity", "attribute", "column", "table"],
        complexity="expert",
        examples=[
            ("Rank students by GPA",
             "SELECT *, RANK() OVER (ORDER BY gpa DESC) AS rank FROM student"),
        ]
    ),
]

# Order Templates
ORDER_TEMPLATES = [
    SQLTemplate(
        template_id="order_asc",
        description="Order ascending",
        pattern="{entity} ordered by {attribute} (ascending|lowest to highest|smallest to largest)",
        sql_template="SELECT * FROM {table} ORDER BY {column} ASC",
        slots=["entity", "attribute", "table", "column"],
        complexity="simple",
        examples=[
            ("Students ordered by age ascending", "SELECT * FROM student ORDER BY age ASC"),
        ]
    ),
    SQLTemplate(
        template_id="order_desc",
        description="Order descending",
        pattern="{entity} ordered by {attribute} (descending|highest to lowest|largest to smallest)",
        sql_template="SELECT * FROM {table} ORDER BY {column} DESC",
        slots=["entity", "attribute", "table", "column"],
        complexity="simple",
        examples=[
            ("Students ordered by GPA descending", "SELECT * FROM student ORDER BY gpa DESC"),
        ]
    ),
]

# Limit Templates
LIMIT_TEMPLATES = [
    SQLTemplate(
        template_id="first_n",
        description="First N rows",
        pattern="(first|top) {n} {entity}",
        sql_template="SELECT * FROM {table} LIMIT {n}",
        slots=["n", "entity", "table"],
        complexity="simple",
        examples=[
            ("First 10 students", "SELECT * FROM student LIMIT 10"),
        ]
    ),
]

# Combine all templates
ALL_TEMPLATES = (
    SIMPLE_TEMPLATES +
    AGGREGATION_TEMPLATES +
    SUPERLATIVE_TEMPLATES +
    JOIN_TEMPLATES +
    COMPLEX_TEMPLATES +
    SET_TEMPLATES +
    COMPARISON_TEMPLATES +
    EXTENDED_AGGREGATION_TEMPLATES +
    EXTENDED_JOIN_TEMPLATES +
    EXTENDED_COMPLEX_TEMPLATES +
    ORDER_TEMPLATES +
    LIMIT_TEMPLATES
)


# ==========================
# TEMPLATE MATCHER
# ==========================

class SQLTemplateMatcher:
    """
    Match questions to SQL templates using pattern matching.

    Uses deterministic regex patterns - no ML.
    """

    def __init__(self, schema: SchemaGraph):
        self.schema = schema
        self.templates = ALL_TEMPLATES

        # Compile regex patterns
        self.compiled_patterns = {}
        for template in self.templates:
            self.compiled_patterns[template.template_id] = self._compile_pattern(template.pattern)

    def match_template(self, question: str, max_results: int = 5) -> List[Tuple[SQLTemplate, float, Dict[str, str]]]:
        """
        Find templates that match the question.

        Args:
            question: Natural language question
            max_results: Maximum templates to return

        Returns:
            List of (template, confidence, slot_bindings) tuples
        """
        matches = []
        question_lower = question.lower()

        for template in self.templates:
            pattern = self.compiled_patterns[template.template_id]
            match = pattern.search(question_lower)

            if match:
                # Extract slot bindings from regex groups
                slot_bindings = match.groupdict()

                # Calculate confidence based on pattern coverage
                coverage = len(match.group(0)) / len(question_lower)
                confidence = coverage * 0.9  # Max 90% from pattern match

                matches.append((template, confidence, slot_bindings))

        # Sort by confidence
        matches.sort(key=lambda m: m[1], reverse=True)

        return matches[:max_results]

    def _compile_pattern(self, pattern: str) -> re.Pattern:
        r"""
        Compile template pattern to regex.

        Converts {slot} to named capture groups.
        Example: "how many {entity}" -> r"how many (?P<entity>\w+)"
        """
        # Escape special regex chars except {, }
        escaped = re.escape(pattern)

        # Convert {slot} to named groups
        # Handle optional parts: (the )?
        regex_pattern = escaped
        regex_pattern = regex_pattern.replace(r"\{", "(?P<")
        regex_pattern = regex_pattern.replace(r"\}", r">\\w+)")

        # Restore optional groups
        regex_pattern = regex_pattern.replace(r"\(", "(")
        regex_pattern = regex_pattern.replace(r"\)", ")")
        regex_pattern = regex_pattern.replace(r"\|", "|")
        regex_pattern = regex_pattern.replace(r"\?", "?")

        return re.compile(regex_pattern, re.IGNORECASE)


# ==========================
# TEMPLATE-BASED GENERATOR
# ==========================

class TemplateBasedGenerator:
    """
    Generate SQL from templates with filled slots.

    Combines template matching with schema linking.
    """

    def __init__(self, schema: SchemaGraph):
        self.schema = schema
        self.matcher = SQLTemplateMatcher(schema)

    def generate_sql(self, question: str, schema_links: Dict[str, str]) -> Optional[str]:
        """
        Generate SQL using template matching.

        Args:
            question: Natural language question
            schema_links: Schema element mappings from schema linking

        Returns:
            Generated SQL or None if no template matches
        """
        # Find matching templates
        matches = self.matcher.match_template(question, max_results=3)

        if not matches:
            return None

        # Try each template in order of confidence
        for template, confidence, slot_bindings in matches:
            try:
                # Fill template with schema links
                sql = self._fill_template(template, slot_bindings, schema_links)
                if sql:
                    return sql
            except Exception as e:
                # Template filling failed, try next one
                continue

        return None

    def _fill_template(self, template: SQLTemplate, slot_bindings: Dict[str, str],
                      schema_links: Dict[str, str]) -> Optional[str]:
        """
        Fill template slots with schema elements.

        Args:
            template: SQL template
            slot_bindings: Extracted from question pattern
            schema_links: Schema element mappings

        Returns:
            Filled SQL string or None
        """
        sql = template.sql_template

        # Map slot bindings to schema elements
        for slot, value in slot_bindings.items():
            # Try to find schema element for this slot
            if slot in ["table", "table1", "table2"]:
                # Look up table name
                table_name = schema_links.get(value, value)
                sql = sql.replace(f"{{{slot}}}", table_name)

            elif slot in ["column", "fk", "pk"]:
                # Look up column name
                column_name = schema_links.get(value, value)
                sql = sql.replace(f"{{{slot}}}", column_name)

            else:
                # Direct substitution
                sql = sql.replace(f"{{{slot}}}", value)

        # Check if all placeholders were filled
        if "{" in sql and "}" in sql:
            # Some placeholders remain unfilled
            return None

        return sql


# ==========================
# TEMPLATE STATISTICS
# ==========================

def get_template_statistics() -> Dict[str, int]:
    """Get statistics about template library."""
    stats = {
        "total_templates": len(ALL_TEMPLATES),
        "simple": len([t for t in ALL_TEMPLATES if t.complexity == "simple"]),
        "medium": len([t for t in ALL_TEMPLATES if t.complexity == "medium"]),
        "complex": len([t for t in ALL_TEMPLATES if t.complexity == "complex"]),
        "expert": len([t for t in ALL_TEMPLATES if t.complexity == "expert"]),
    }
    return stats


# ==========================
# TESTING
# ==========================

if __name__ == "__main__":
    print("SQL Template Library v0.1")
    print("=" * 60)

    # Statistics
    stats = get_template_statistics()
    print(f"\nTemplate Library Statistics:")
    print(f"  Total templates: {stats['total_templates']}")
    print(f"  Simple: {stats['simple']}")
    print(f"  Medium: {stats['medium']}")
    print(f"  Complex: {stats['complex']}")
    print(f"  Expert: {stats['expert']}")

    # Test pattern matching
    print("\nTesting Template Matching:")
    from .schema_graph import SchemaGraph

    # Create dummy schema
    schema = SchemaGraph()
    matcher = SQLTemplateMatcher(schema)

    test_questions = [
        "How many students are there?",
        "What is the maximum age?",
        "Student with the highest GPA",
        "Students who do not take any courses",
    ]

    for question in test_questions:
        print(f"\n  Question: {question}")
        matches = matcher.match_template(question, max_results=2)
        for template, confidence, bindings in matches:
            print(f"    Template: {template.template_id} (confidence={confidence:.2f})")
            print(f"    Pattern: {template.pattern}")
            print(f"    Bindings: {bindings}")

    print("\n[OK] SQL template library ready")
    print(f"[INFO] {stats['total_templates']} templates loaded")


# ==========================
# IR-BASED TEMPLATE MATCHING
# ==========================

def match_template_to_ir(ir, analysis, schema):
    """
    Match IR representation to SQL templates for fast generation.

    This provides a simplified template matching based on IR intent and structure,
    allowing common patterns to bypass complex IR→SQL lowering.

    Args:
        ir: QueryIR object
        analysis: QuestionAnalysis object
        schema: SchemaGraph object

    Returns:
        Dict with {'template_id': str, 'sql': str} or None if no template matches
    """
    from .query_ir import QueryIntent

    # Import sqlglot for SQL construction
    from sqlglot import exp

    # ==========================
    # Simple COUNT queries
    # ==========================
    if ir.intent == QueryIntent.COUNT and len(ir.entities) == 1:
        table = ir.entities[0].table

        # No conditions → simple COUNT(*)
        if not ir.conditions and not ir.selection:
            return {
                'template_id': 'simple_count',
                'sql': f"SELECT COUNT(*) FROM {table}"
            }

        # With WHERE conditions
        if ir.selection:
            where_expr = ir.selection.to_expr()
            if where_expr:
                where_sql = exp.Where(this=where_expr).sql()
                return {
                    'template_id': 'count_with_where',
                    'sql': f"SELECT COUNT(*) FROM {table} {where_sql}"
                }

    # ==========================
    # Simple SELECT queries
    # ==========================
    if ir.intent == QueryIntent.SELECT and len(ir.entities) == 1 and ir.attributes:
        table = ir.entities[0].table

        # Build SELECT columns
        if len(ir.attributes) == 0:
            cols = "*"
        else:
            cols = ", ".join([f"{a.entity}.{a.column}" for a in ir.attributes[:5]])

        sql = f"SELECT {cols} FROM {table}"

        # Add WHERE if present
        if ir.selection:
            where_expr = ir.selection.to_expr()
            if where_expr:
                where_sql = exp.Where(this=where_expr).sql()
                sql += f" {where_sql}"

        # Add ORDER BY if present
        if ir.ordering and ir.ordering.attributes:
            order_attr = ir.ordering.attributes[0]
            direction = "DESC" if ir.ordering.descending else "ASC"
            sql += f" ORDER BY {order_attr.entity}.{order_attr.column} {direction}"

        # Add LIMIT if present
        if ir.limitation:
            sql += f" LIMIT {ir.limitation}"

        return {
            'template_id': 'simple_select',
            'sql': sql
        }

    # ==========================
    # Simple AGGREGATION queries
    # ==========================
    if ir.intent == QueryIntent.AGGREGATE and len(ir.entities) == 1:
        table = ir.entities[0].table

        if ir.projection and ir.projection.aggregations:
            agg_func, attr = ir.projection.aggregations[0]

            # Build aggregation expression
            if attr:
                agg_expr = f"{agg_func}({attr.entity}.{attr.column})"
            else:
                agg_expr = f"{agg_func}(*)"

            sql = f"SELECT {agg_expr} FROM {table}"

            # Add WHERE if present
            if ir.selection:
                where_expr = ir.selection.to_expr()
                if where_expr:
                    where_sql = exp.Where(this=where_expr).sql()
                    sql += f" {where_sql}"

            # Add GROUP BY if needed
            if ir.aggregation and ir.aggregation.group_by_attributes:
                group_cols = ", ".join([
                    f"{a.entity}.{a.column}" for a in ir.aggregation.group_by_attributes
                ])
                sql += f" GROUP BY {group_cols}"

                # Add HAVING if present
                if ir.aggregation.having_condition:
                    having_expr = ir.aggregation.having_condition.to_expr()
                    if having_expr:
                        having_sql = exp.Having(this=having_expr).sql()
                        sql += f" {having_sql}"

            return {
                'template_id': 'simple_aggregation',
                'sql': sql
            }

    # ==========================
    # Superlative queries (argmax/argmin)
    # ==========================
    if ir.intent == QueryIntent.SELECT and len(ir.entities) == 1 and ir.ordering:
        table = ir.entities[0].table

        # Detect superlative pattern: ORDER BY ... LIMIT 1
        if ir.limitation == 1 and ir.ordering.attributes:
            order_attr = ir.ordering.attributes[0]
            direction = "DESC" if ir.ordering.descending else "ASC"

            # Build SELECT columns
            if len(ir.attributes) == 0:
                cols = "*"
            else:
                cols = ", ".join([f"{a.entity}.{a.column}" for a in ir.attributes[:5]])

            sql = f"SELECT {cols} FROM {table} ORDER BY {order_attr.entity}.{order_attr.column} {direction} LIMIT 1"

            # Add WHERE if present
            if ir.selection:
                where_expr = ir.selection.to_expr()
                if where_expr:
                    # Insert WHERE before ORDER BY
                    sql = f"SELECT {cols} FROM {table} {exp.Where(this=where_expr).sql()} ORDER BY {order_attr.entity}.{order_attr.column} {direction} LIMIT 1"

            template_id = 'argmax' if ir.ordering.descending else 'argmin'
            return {
                'template_id': template_id,
                'sql': sql
            }

    # No template match
    return None
