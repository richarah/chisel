#!/usr/bin/env python3
"""
Quick benchmark script for CHISEL without requiring database files.

This provides a baseline by testing on Spider dev set with simple metrics.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chisel.question_analysis import analyze_question
import spacy


def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison."""
    return " ".join(sql.lower().strip().split())


def analyze_questions_only(data_file: Path, max_examples: int = 100):
    """
    Analyze questions without generating SQL (no schema needed).

    This tests question understanding capabilities.
    """
    print("="*80)
    print("CHISEL Question Analysis Benchmark")
    print("="*80)
    print(f"Dataset: {data_file}")
    print(f"Max examples: {max_examples}\n")

    # Load data
    with open(data_file) as f:
        data = json.load(f)

    # Load spaCy
    print("Loading spaCy...")
    nlp = spacy.load("en_core_web_sm")
    print("Ready\n")

    # Stats
    stats = {
        "total": 0,
        "with_ordinals": 0,
        "with_quantifiers": 0,
        "with_superlatives": 0,
        "with_comparatives": 0,
        "with_negation": 0,
        "with_numbers": 0,
        "with_dates": 0,
        "question_types": {},
        "sql_signals": {},
    }

    # Analyze each question
    for i, example in enumerate(data[:max_examples]):
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{min(max_examples, len(data))}...")

        question = example["question"]
        gold_sql = example["query"]

        # Analyze
        analysis = analyze_question(question, nlp)

        stats["total"] += 1

        # Collect stats
        if analysis.ordinal_values:
            stats["with_ordinals"] += 1

        if any(sig.startswith("QUANTIFIER_") for sig in analysis.sql_signals):
            stats["with_quantifiers"] += 1

        if analysis.superlatives:
            stats["with_superlatives"] += 1

        if analysis.comparatives:
            stats["with_comparatives"] += 1

        if analysis.negations:
            stats["with_negation"] += 1

        if analysis.numeric_values:
            stats["with_numbers"] += 1

        if analysis.date_values:
            stats["with_dates"] += 1

        # Question types
        qtype = analysis.question_type
        stats["question_types"][qtype] = stats["question_types"].get(qtype, 0) + 1

        # SQL signals
        for sig in analysis.sql_signals:
            stats["sql_signals"][sig] = stats["sql_signals"].get(sig, 0) + 1

    # Print results
    print("\n" + "="*80)
    print("Question Understanding Benchmark Results")
    print("="*80)

    print(f"\nTotal questions analyzed: {stats['total']}")
    print(f"\nFeature Detection:")
    print(f"  Ordinals:       {stats['with_ordinals']:4d} ({stats['with_ordinals']/stats['total']*100:5.1f}%)")
    print(f"  Quantifiers:    {stats['with_quantifiers']:4d} ({stats['with_quantifiers']/stats['total']*100:5.1f}%)")
    print(f"  Superlatives:   {stats['with_superlatives']:4d} ({stats['with_superlatives']/stats['total']*100:5.1f}%)")
    print(f"  Comparatives:   {stats['with_comparatives']:4d} ({stats['with_comparatives']/stats['total']*100:5.1f}%)")
    print(f"  Negation:       {stats['with_negation']:4d} ({stats['with_negation']/stats['total']*100:5.1f}%)")
    print(f"  Numbers:        {stats['with_numbers']:4d} ({stats['with_numbers']/stats['total']*100:5.1f}%)")
    print(f"  Dates:          {stats['with_dates']:4d} ({stats['with_dates']/stats['total']*100:5.1f}%)")

    print(f"\nQuestion Type Distribution:")
    for qtype, count in sorted(stats["question_types"].items(), key=lambda x: -x[1])[:10]:
        print(f"  {qtype:20s} {count:4d} ({count/stats['total']*100:5.1f}%)")

    print(f"\nTop SQL Signals Detected:")
    for sig, count in sorted(stats["sql_signals"].items(), key=lambda x: -x[1])[:15]:
        print(f"  {sig:20s} {count:4d} ({count/stats['total']*100:5.1f}%)")

    return stats


def quick_spider_test(data_file: Path, num_examples: int = 20):
    """
    Show sample question analysis on first N examples.
    """
    print("\n" + "="*80)
    print(f"Sample Analysis (first {num_examples} examples)")
    print("="*80)

    with open(data_file) as f:
        data = json.load(f)

    nlp = spacy.load("en_core_web_sm")

    for i, example in enumerate(data[:num_examples]):
        question = example["question"]
        gold_sql = example["query"]

        analysis = analyze_question(question, nlp)

        print(f"\n[{i+1}] {question}")
        print(f"    SQL: {gold_sql}")
        print(f"    Type: {analysis.question_type}")
        print(f"    Signals: {', '.join(sorted(analysis.sql_signals)[:5])}")
        if analysis.superlatives:
            print(f"    Superlatives: {analysis.superlatives}")
        if analysis.comparatives:
            print(f"    Comparatives: {analysis.comparatives}")
        if analysis.ordinal_values:
            print(f"    Ordinals: {analysis.ordinal_values}")
        if any(s.startswith("QUANTIFIER_") for s in analysis.sql_signals):
            quants = [s for s in analysis.sql_signals if s.startswith("QUANTIFIER_")]
            print(f"    Quantifiers: {quants}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CHISEL Benchmark")
    parser.add_argument("--spider-dev", type=Path,
                       default=Path("data/spider/dev.json"),
                       help="Path to Spider dev.json")
    parser.add_argument("--max", type=int, default=100,
                       help="Max examples to analyze")
    parser.add_argument("--samples", type=int, default=20,
                       help="Number of sample analyses to show")

    args = parser.parse_args()

    if not args.spider_dev.exists():
        print(f"ERROR: {args.spider_dev} not found")
        print("Run: python scripts/download_datasets.py")
        return

    # Run benchmark
    stats = analyze_questions_only(args.spider_dev, args.max)

    # Show samples
    quick_spider_test(args.spider_dev, args.samples)

    print("\n" + "="*80)
    print("Benchmark Complete!")
    print("="*80)
    print(f"\nNote: This benchmark tests question understanding only.")
    print(f"SQL generation requires schema files (tables.json).")


if __name__ == "__main__":
    main()
