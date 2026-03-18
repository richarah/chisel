#!/usr/bin/env python3
"""
Simple evaluation script for CHISEL that works without database files.

This evaluates exact SQL string matching on Spider dev set to establish a baseline.
Full evaluation with execution accuracy requires database files.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chisel import ChiselPipeline


def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison (case-insensitive, whitespace-normalized)."""
    return " ".join(sql.lower().strip().split())


def load_spider_dev(data_dir: Path) -> List[Dict]:
    """Load Spider dev set."""
    dev_file = data_dir / "spider" / "dev.json"
    if not dev_file.exists():
        raise FileNotFoundError(f"Dev file not found: {dev_file}")

    with open(dev_file) as f:
        return json.load(f)


def evaluate_exact_match(predictions: List[str], gold: List[str]) -> Tuple[int, int]:
    """Calculate exact match accuracy."""
    correct = 0
    total = len(predictions)

    for pred, gold_sql in zip(predictions, gold):
        if normalize_sql(pred) == normalize_sql(gold_sql):
            correct += 1

    return correct, total


def evaluate_chisel_simple(data_dir: Path, max_examples: int = 100) -> Dict:
    """
    Run simple evaluation on Spider dev set.

    Args:
        data_dir: Path to data directory containing spider/
        max_examples: Maximum number of examples to evaluate (for quick testing)

    Returns:
        Dictionary with evaluation results
    """
    print("="*80)
    print("CHISEL Simple Evaluation (Exact Match)")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Max examples: {max_examples}")
    print()

    # Load dev set
    print("[1/3] Loading Spider dev set...")
    try:
        dev_data = load_spider_dev(data_dir)
        print(f"  Loaded {len(dev_data)} examples")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        print("  Run: python scripts/download_datasets.py")
        return {}

    # Initialize pipeline
    print("\n[2/3] Initializing CHISEL pipeline...")
    try:
        pipeline = ChiselPipeline()
        print("  Pipeline initialized")
    except Exception as e:
        print(f"  ERROR: {e}")
        return {}

    # Evaluate
    print(f"\n[3/3] Evaluating on {min(max_examples, len(dev_data))} examples...")
    predictions = []
    gold = []
    errors = []

    for i, example in enumerate(dev_data[:max_examples]):
        question = example["question"]
        db_id = example["db_id"]
        gold_sql = example["query"]

        try:
            # Generate SQL
            result = pipeline.process(question, db_id)
            pred_sql = result.sql if hasattr(result, "sql") else ""

            predictions.append(pred_sql)
            gold.append(gold_sql)

            # Show progress
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{min(max_examples, len(dev_data))} examples...")

        except Exception as e:
            errors.append((i, question, str(e)))
            predictions.append("")
            gold.append(gold_sql)

    # Calculate accuracy
    print("\n" + "="*80)
    print("Results")
    print("="*80)

    correct, total = evaluate_exact_match(predictions, gold)
    accuracy = (correct / total * 100) if total > 0 else 0.0

    print(f"Exact Match Accuracy: {correct}/{total} ({accuracy:.2f}%)")
    print(f"Errors: {len(errors)}")

    # Show some errors
    if errors:
        print(f"\nFirst {min(5, len(errors))} errors:")
        for idx, question, error in errors[:5]:
            print(f"  [{idx}] {question}")
            print(f"       Error: {error}")

    # Show some examples
    print("\nSample predictions:")
    for i in range(min(3, len(predictions))):
        print(f"\n  Question: {dev_data[i]['question']}")
        print(f"  Gold:     {gold[i]}")
        print(f"  Predicted: {predictions[i]}")
        match = "MATCH" if normalize_sql(predictions[i]) == normalize_sql(gold[i]) else "MISMATCH"
        print(f"  Status:   [{match}]")

    return {
        "exact_match_accuracy": accuracy,
        "correct": correct,
        "total": total,
        "errors": len(errors),
    }


def main():
    """Run evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Simple CHISEL evaluation")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=100,
        help="Maximum number of examples to evaluate"
    )

    args = parser.parse_args()

    results = evaluate_chisel_simple(args.data_dir, args.max_examples)

    if results:
        print("\n" + "="*80)
        print("Evaluation complete!")
        print("="*80)
        print(f"Accuracy: {results['exact_match_accuracy']:.2f}%")
        print(f"Note: This is exact string match only.")
        print(f"For full evaluation with execution accuracy, database files are needed.")


if __name__ == "__main__":
    main()
