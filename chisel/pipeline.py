"""
CHISEL Pipeline

Orchestrates all components:
1. Schema Graph (Task 1)
2. Question Analysis (Task 2)
3. Schema Linking (Task 3)
4. Skeleton Prediction (Task 4)
5. Slot Filling (Task 5)
6. Validation & Repair (Task 6)

Libraries do the work. We just compose them.
"""

from typing import Dict, Optional
import spacy

from .schema_graph import SchemaGraph, load_spider_schemas
from .question_analysis import QuestionAnalysis, analyze_question
from .schema_linking import link_question_to_schema, SchemaLink
from .skeleton_prediction import predict_skeleton, refine_skeleton_with_links, SQLSkeleton
from .slot_filling import fill_sql_skeleton, FilledSQL


class ChiselPipeline:
    """
    Main CHISEL pipeline.

    Deterministic text-to-SQL using only rules + libraries.
    """

    def __init__(self, tables_json_path: str, fuzzy_threshold: float = 75.0):
        """
        Initialize pipeline.

        Args:
            tables_json_path: Path to Spider's tables.json
            fuzzy_threshold: Minimum fuzzy match score (0-100)
        """
        # Load schemas (Task 1)
        print("Loading schemas from Spider...")
        self.schemas = load_spider_schemas(tables_json_path)
        print(f"Loaded {len(self.schemas)} database schemas")

        # Load spaCy model (used by Task 2) - MUST be pre-installed
        print("Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_sm")

        # lemminflect MUST be installed - it auto-extends spaCy tokens
        import lemminflect  # This import ensures it's available and integrated
        print("  [OK] lemminflect integration enabled")

        self.fuzzy_threshold = fuzzy_threshold

        print("Pipeline initialized.")

    def run(self, question: str, db_id: str, verbose: bool = False) -> Optional[str]:
        """
        Convert question to SQL.

        Args:
            question: English question
            db_id: Database ID (must be in loaded schemas)
            verbose: Print intermediate steps

        Returns: SQL string or None if failed
        """
        # Get schema for this database
        if db_id not in self.schemas:
            if verbose:
                print(f"ERROR: Database '{db_id}' not found in loaded schemas")
            return None

        schema = self.schemas[db_id]

        # ==========================
        # Task 2: Question Analysis
        # ==========================
        if verbose:
            print(f"\n[Task 2] Analyzing question...")
        analysis = analyze_question(question, self.nlp)

        if verbose:
            print(f"  Signals: {analysis.sql_signals}")
            print(f"  Content words: {analysis.content_words[:10]}")
            print(f"  Question type: {analysis.question_type}")

        # ==========================
        # Task 3: Schema Linking
        # ==========================
        if verbose:
            print(f"\n[Task 3] Linking to schema...")
        links = link_question_to_schema(analysis, schema, self.fuzzy_threshold)

        if verbose:
            print(f"  Found {len(links)} links")
            for link in links[:5]:
                print(f"    '{link.question_phrase}' -> {link.schema_element} "
                      f"({link.link_type.value}, {link.score:.1f})")

        # ==========================
        # Task 4: Skeleton Prediction
        # ==========================
        if verbose:
            print(f"\n[Task 4] Predicting SQL skeleton...")
        skeleton = predict_skeleton(analysis)

        # Refine with schema linking results
        from .schema_linking import get_tables_from_links
        involved_tables = get_tables_from_links(links)
        skeleton = refine_skeleton_with_links(skeleton, analysis, len(involved_tables))

        if verbose:
            print(f"  Aggregation: {skeleton.has_aggregation()}")
            print(f"  WHERE: {skeleton.needs_where}")
            print(f"  GROUP BY: {skeleton.needs_group_by}")
            print(f"  ORDER BY: {skeleton.needs_order_by}")
            print(f"  LIMIT: {skeleton.limit_value}")

        # ==========================
        # Task 5: Slot Filling
        # ==========================
        if verbose:
            print(f"\n[Task 5] Filling SQL slots...")
        filled = fill_sql_skeleton(skeleton, analysis, links, schema)

        if not filled:
            if verbose:
                print("  ERROR: Could not fill SQL skeleton")
            return None

        # Generate SQL (sqlglot does this)
        sql = filled.to_sql()

        if verbose:
            print(f"\n[Generated SQL]")
            print(f"  {sql}")

        # ==========================
        # Task 6: Validation (TODO)
        # ==========================
        # For now, return the generated SQL
        # In future: parse with sqlglot, validate, repair if needed

        return sql

    def run_batch(self, questions: list, db_ids: list, verbose: bool = False) -> list:
        """
        Run on multiple questions.

        Args:
            questions: List of questions
            db_ids: List of database IDs (same length as questions)
            verbose: Print progress

        Returns: List of SQL strings (None for failures)
        """
        if len(questions) != len(db_ids):
            raise ValueError("questions and db_ids must have same length")

        results = []
        for i, (question, db_id) in enumerate(zip(questions, db_ids)):
            if verbose:
                print(f"\n{'='*80}")
                print(f"Question {i+1}/{len(questions)}: {question}")
                print(f"Database: {db_id}")

            sql = self.run(question, db_id, verbose=verbose)
            results.append(sql)

            if verbose and not sql:
                print(f"  [WARN] Failed to generate SQL")

        return results


if __name__ == "__main__":
    import sys

    # Simple test
    if len(sys.argv) > 1:
        tables_path = sys.argv[1]
    else:
        tables_path = "data/spider/tables.json"

    print("Initializing CHISEL Pipeline...")
    pipeline = ChiselPipeline(tables_path)

    # Test questions
    test_cases = [
        ("How many students are there?", "student_1"),
        ("What are the names of all students?", "student_1"),
        ("What is the name of the student with the highest GPA?", "student_1"),
    ]

    print("\n" + "="*80)
    print("Running test cases...")
    print("="*80)

    for question, db_id in test_cases:
        print(f"\nQuestion: {question}")
        print(f"Database: {db_id}")

        sql = pipeline.run(question, db_id, verbose=False)

        if sql:
            print(f"SQL: {sql}")
        else:
            print("FAILED: Could not generate SQL")
