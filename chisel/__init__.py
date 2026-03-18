"""
CHISEL: Compositional Heuristic Inference for SQL from English Language

A deterministic text-to-SQL solver using rules + libraries.
"""

__version__ = "0.5.0"

# Export main pipeline
from .pipeline import ChiselPipeline

# Export core components
from .schema_graph import SchemaGraph, load_spider_schemas
from .question_analysis import QuestionAnalysis, analyze_question
from .schema_linking import link_question_to_schema, SchemaLink
from .skeleton_prediction import predict_skeleton, SQLSkeleton
from .slot_filling import fill_sql_skeleton, FilledSQL
from .validation import validate_sql, validate_and_repair, check_sql_features

# Export enhanced components (v0.5)
from .ontology_schema_linking import (
    CompositeSchemaLinker,
    ValueNormalizer,
    enhanced_schema_linking,
    OntologyMapping
)
from .temporal_normalization import TemporalNormalizer, TemporalExpression
from .sql_templates import (
    SQLTemplate,
    SQLTemplateMatcher,
    TemplateBasedGenerator,
    get_template_statistics
)

# Knowledge base (optional)
try:
    from .knowledge_base import KnowledgeBase, EntityType, KnowledgeEntity
    KB_COMPONENTS = ["KnowledgeBase", "EntityType", "KnowledgeEntity"]
except ImportError:
    KB_COMPONENTS = []

__all__ = [
    # Main pipeline
    "ChiselPipeline",
    # Schema
    "SchemaGraph",
    "load_spider_schemas",
    # Question analysis
    "QuestionAnalysis",
    "analyze_question",
    # Schema linking
    "link_question_to_schema",
    "SchemaLink",
    # Skeleton
    "predict_skeleton",
    "SQLSkeleton",
    # Slot filling
    "fill_sql_skeleton",
    "FilledSQL",
    # Validation
    "validate_sql",
    "validate_and_repair",
    "check_sql_features",
    # Enhanced components (v0.5)
    "CompositeSchemaLinker",
    "ValueNormalizer",
    "enhanced_schema_linking",
    "OntologyMapping",
    "TemporalNormalizer",
    "TemporalExpression",
    "SQLTemplate",
    "SQLTemplateMatcher",
    "TemplateBasedGenerator",
    "get_template_statistics",
] + KB_COMPONENTS
