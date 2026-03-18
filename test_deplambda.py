#!/usr/bin/env python3
"""Test deplambda template integration."""

import sys
import spacy
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from chisel.schema_graph import SchemaGraph
from chisel.ir_composer import IRComposer

# Initialize
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

# Load schema (use a simple test schema)
print("Loading schema...")
schema_path = Path(__file__).parent / "data" / "databases" / "concert_singer" / "schema.json"
if not schema_path.exists():
    print(f"Schema not found: {schema_path}")
    sys.exit(1)

schema = SchemaGraph.from_spider_json(str(schema_path))

# Initialize composer
print("Initializing IR composer...")
composer = IRComposer(schema, nlp)

print(f"\nLoaded {len(composer.templates)} templates total")
print(f"Registered {len(composer.matcher)} patterns in DependencyMatcher")

# Print template summary
sql_specific = [t for t in composer.templates if t.priority >= 90]
linguistic = [t for t in composer.templates if t.priority < 90]

print(f"\nSQL-specific templates (priority >= 90): {len(sql_specific)}")
for t in sql_specific[:5]:
    print(f"  - {t.name} (priority {t.priority}): {t.description}")

print(f"\nLinguistic templates (priority < 90): {len(linguistic)}")
for t in linguistic[:10]:
    print(f"  - {t.name} (priority {t.priority}): {t.description}")

# Test a simple query
test_queries = [
    "How many singers are there?",
    "What is the average age of singers?",
    "Show the names of singers"
]

print("\n" + "="*60)
print("Testing simple queries...")
print("="*60)

from chisel.question_analysis import QuestionAnalyzer
from chisel.schema_linking import SchemaLinker

analyzer = QuestionAnalyzer(nlp)
linker = SchemaLinker(schema, nlp)

for query in test_queries:
    print(f"\nQuery: {query}")
    analysis = analyzer.analyze(query)
    links = linker.link(query, analysis)

    expr = composer.compose(query, analysis, links)
    if expr:
        print(f"  Lambda: {expr}")
    else:
        print("  [FAILED] No lambda expression generated")

print("\nDone!")
