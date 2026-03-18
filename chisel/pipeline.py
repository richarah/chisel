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
# Old IR removed - using Lambda DCS only

# Enhanced modules (v0.5)
from .ontology_schema_linking import CompositeSchemaLinker, ValueNormalizer, enhanced_schema_linking
from .temporal_normalization import TemporalNormalizer
from .sql_templates import TemplateBasedGenerator, SQLTemplateMatcher
try:
    from .knowledge_base import KnowledgeBase, EntityType
    KB_AVAILABLE = True
except ImportError:
    KB_AVAILABLE = False


class ChiselPipeline:
    """
    Main CHISEL pipeline.

    Deterministic text-to-SQL using only rules + libraries.
    """

    def __init__(self, tables_json_path: str, fuzzy_threshold: float = 75.0,
                 use_ontology: bool = True, use_templates: bool = True,
                 use_knowledge_base: bool = True, use_lambda_dcs: bool = True,
                 domain: str = "university"):
        """
        Initialize pipeline.

        Args:
            tables_json_path: Path to Spider's tables.json
            fuzzy_threshold: Minimum fuzzy match score (0-100)
            use_ontology: Enable ontology-based schema linking
            use_templates: Enable SQL template matching
            use_knowledge_base: Enable external knowledge bases (requires SPARQLWrapper)
            domain: Domain ontology to use (university, geography, business)
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
        self.domain = domain

        # Initialize enhanced modules (v0.5)
        self.use_ontology = use_ontology
        self.use_templates = use_templates
        self.use_knowledge_base = use_knowledge_base and KB_AVAILABLE
        self.use_lambda_dcs = use_lambda_dcs

        # Ontology schema linker (one per schema)
        self.ontology_linkers = {}

        # Temporal normalizer
        self.temporal_normalizer = TemporalNormalizer()
        print("  [OK] Temporal normalization enabled")

        # Value normalizer
        self.value_normalizer = ValueNormalizer()

        # Knowledge base (optional)
        if self.use_knowledge_base:
            try:
                self.knowledge_base = KnowledgeBase(use_geonames=True, use_dbpedia=False)
                print("  [OK] Knowledge base integration enabled")
            except Exception as e:
                print(f"  [WARN] Knowledge base initialization failed: {e}")
                self.use_knowledge_base = False
        else:
            self.knowledge_base = None

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
        # Task 3: Schema Linking (Enhanced)
        # ==========================
        if verbose:
            print(f"\n[Task 3] Linking to schema...")

        # Original schema linking (fallback)
        basic_links = link_question_to_schema(analysis, schema, self.fuzzy_threshold)

        # Enhanced ontology-based linking with COMA++ composite matching (v0.5)
        if self.use_ontology:
            # Create ontology linker for this schema if not cached
            if db_id not in self.ontology_linkers:
                linker = CompositeSchemaLinker(schema)
                # Tune weights for better performance
                linker.weights = {
                    'string': 0.35,      # Name similarity (increased from 0.30)
                    'linguistic': 0.25,  # Semantic type
                    'structural': 0.20,  # Graph structure
                    'constraint': 0.20   # Keys/FKs
                }
                self.ontology_linkers[db_id] = linker

            # Use composite matching via match_all
            linker = self.ontology_linkers[db_id]
            composite_links = linker.match_all(
                question_terms=analysis.content_words,
                question_phrases=[chunk[0] for chunk in analysis.noun_chunks],
                threshold=self.fuzzy_threshold,
                domain=self.domain
            )

            # Merge: prioritize composite scores, fallback to basic
            from .schema_linking import merge_schema_links
            links = merge_schema_links(basic_links, composite_links)

            if verbose:
                print(f"  Basic links: {len(basic_links)}, Composite links: {len(composite_links)}, Merged: {len(links)}")
        else:
            links = basic_links

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

        # Old IR construction removed - Lambda DCS only

        # ==========================
        # Phase 3: Quantifier Scope Resolution
        # ==========================
        # Check for complex quantifiers needing continuation semantics
        if any(sig in analysis.sql_signals for sig in ["QUANTIFIER_ALL", "QUANTIFIER_NO", "EXISTS", "NOT_EXISTS"]):
            try:
                from .continuation_semantics import detect_quantifiers, generate_sql_with_quantifiers
                quantifiers = detect_quantifiers(analysis, schema)

                if len(quantifiers) >= 2:  # Multiple quantifiers need special handling
                    if verbose:
                        print(f"\n[Phase 3] Detected {len(quantifiers)} quantifiers")
                        for q in quantifiers:
                            print(f"  {q.type.value}: {q.domain} ({q.domain_table})")

                    quant_sql = generate_sql_with_quantifiers(quantifiers, None, schema)
                    if quant_sql:
                        return quant_sql.sql()
            except Exception as e:
                if verbose:
                    print(f"  [WARN] Quantifier handling failed: {e}")

        # ==========================
        # IR Layer: Compositional Semantics (deplambda-style)
        # ==========================
        if self.use_lambda_dcs:
            try:
                from .ir_composer import IRComposer
                from .ir_compiler import IRCompiler

                # Compose lambda expression from dependency tree
                composer = IRComposer(schema, self.nlp)
                lambda_expr = composer.compose(analysis.original_text, analysis, links)

                if lambda_expr:
                    if verbose:
                        print(f"\n[IR Composer] Built expression: {lambda_expr}")

                    # Compile to SQL
                    compiler = IRCompiler(schema)
                    sql = compiler.compile(lambda_expr, verbose=verbose)

                    if sql and "unknown" not in sql.lower():
                        if verbose:
                            print(f"  [OK] Compiled to SQL: {sql}")
                        return sql
            except Exception as e:
                if verbose:
                    print(f"  [WARN] IR layer failed: {e}")
                    import traceback
                    traceback.print_exc()

        # No fallback - IR only (peak performance principle)
        if verbose:
            print(f"\n[FAIL] IR layer did not generate valid SQL")

        return None

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
