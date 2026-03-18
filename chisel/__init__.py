"""
CHISEL: Compositional Heuristic Inference for SQL from English Language

A deterministic text-to-SQL solver using rules + libraries.
"""

__version__ = "0.1.0"

# Export main pipeline
from .pipeline import ChiselPipeline

# Export core components
from .schema_graph import SchemaGraph, load_spider_schemas
from .question_analysis import QuestionAnalysis, analyze_question
from .schema_linking import link_question_to_schema, SchemaLink
from .skeleton_prediction import predict_skeleton, SQLSkeleton
from .slot_filling import fill_sql_skeleton, FilledSQL
from .validation import validate_sql, validate_and_repair, check_sql_features

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
]
