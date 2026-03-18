"""
Task 7: Evaluation on Spider Dev Set

Run CHISEL on Spider and compute metrics.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chisel.pipeline import ChiselPipeline
from chisel.validation import validate_sql


def load_spider_dev(dev_json_path: str) -> List[Dict]:
    """Load Spider dev.json."""
    with open(dev_json_path, 'r') as f:
        data = json.load(f)
    return data


def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison (basic normalization)."""
    if not sql:
        return ""

    # Convert to lowercase, remove extra whitespace
    sql = sql.lower()
    sql = " ".join(sql.split())

    return sql


def compare_sql(predicted: str, gold: str) -> bool:
    """
    Compare predicted and gold SQL.

    For now, use exact match on normalized SQL.
    In future: use Spider's official evaluation script.
    """
    if not predicted or not gold:
        return False

    pred_norm = normalize_sql(predicted)
    gold_norm = normalize_sql(gold)

    return pred_norm == gold_norm


def evaluate_on_spider(
    pipeline: ChiselPipeline,
    dev_data: List[Dict],
    limit: int = None,
    verbose: bool = False
) -> Dict:
    """
    Evaluate pipeline on Spider dev set.

    Args:
        pipeline: Initialized CHISEL pipeline
        dev_data: Spider dev.json data
        limit: Max number of examples to evaluate (None = all)
        verbose: Print each example

    Returns: Dictionary with metrics
    """
    total = 0
    correct = 0
    failed_to_generate = 0
    invalid_sql = 0

    correct_by_difficulty = defaultdict(int)
    total_by_difficulty = defaultdict(int)

    predictions = []

    # Limit examples if specified
    if limit:
        dev_data = dev_data[:limit]

    print(f"Evaluating on {len(dev_data)} examples...")

    for i, example in enumerate(dev_data):
        question = example["question"]
        db_id = example["db_id"]
        gold_sql = example["query"]
        difficulty = example.get("difficulty", "unknown")

        total += 1
        total_by_difficulty[difficulty] += 1

        if verbose or (i + 1) % 100 == 0:
            print(f"\n[{i+1}/{len(dev_data)}] {question}")
            print(f"  Database: {db_id}")
            print(f"  Difficulty: {difficulty}")

        # Generate SQL
        predicted_sql = pipeline.run(question, db_id, verbose=False)

        if not predicted_sql:
            failed_to_generate += 1
            predictions.append({
                "question": question,
                "db_id": db_id,
                "gold": gold_sql,
                "predicted": None,
                "correct": False,
                "error": "Failed to generate SQL"
            })
            if verbose:
                print(f"  [X] Failed to generate SQL")
            continue

        # Validate SQL
        is_valid, error = validate_sql(predicted_sql)
        if not is_valid:
            invalid_sql += 1
            if verbose:
                print(f"  [X] Invalid SQL: {error}")

        # Compare with gold
        is_correct = compare_sql(predicted_sql, gold_sql)

        if is_correct:
            correct += 1
            correct_by_difficulty[difficulty] += 1

        predictions.append({
            "question": question,
            "db_id": db_id,
            "gold": gold_sql,
            "predicted": predicted_sql,
            "correct": is_correct,
            "valid": is_valid,
            "difficulty": difficulty
        })

        if verbose:
            print(f"  Gold:      {gold_sql}")
            print(f"  Predicted: {predicted_sql}")
            print(f"  {'[OK]' if is_correct else '[X]'} {'CORRECT' if is_correct else 'INCORRECT'}")

    # Compute metrics
    accuracy = correct / total if total > 0 else 0.0

    metrics = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "failed_to_generate": failed_to_generate,
        "invalid_sql": invalid_sql,
        "by_difficulty": {}
    }

    # Accuracy by difficulty
    for difficulty in total_by_difficulty:
        diff_total = total_by_difficulty[difficulty]
        diff_correct = correct_by_difficulty[difficulty]
        diff_acc = diff_correct / diff_total if diff_total > 0 else 0.0

        metrics["by_difficulty"][difficulty] = {
            "total": diff_total,
            "correct": diff_correct,
            "accuracy": diff_acc
        }

    return metrics, predictions


def print_metrics(metrics: Dict):
    """Print evaluation metrics."""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    print(f"\nOverall:")
    print(f"  Total:                {metrics['total']}")
    print(f"  Correct:              {metrics['correct']}")
    print(f"  Accuracy:             {metrics['accuracy']:.2%}")
    print(f"  Failed to generate:   {metrics['failed_to_generate']}")
    print(f"  Invalid SQL:          {metrics['invalid_sql']}")

    print(f"\nBy Difficulty:")
    for difficulty, diff_metrics in metrics["by_difficulty"].items():
        print(f"  {difficulty.upper()}:")
        print(f"    Total:    {diff_metrics['total']}")
        print(f"    Correct:  {diff_metrics['correct']}")
        print(f"    Accuracy: {diff_metrics['accuracy']:.2%}")

    print("\n" + "="*80)


def save_predictions(predictions: List[Dict], output_path: str):
    """Save predictions to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"\nPredictions saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate CHISEL on Spider dev set")
    parser.add_argument("--tables", default="data/spider/tables.json",
                       help="Path to Spider tables.json")
    parser.add_argument("--dev", default="data/spider/dev.json",
                       help="Path to Spider dev.json")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of examples (for quick testing)")
    parser.add_argument("--output", default="predictions.json",
                       help="Output file for predictions")
    parser.add_argument("--verbose", action="store_true",
                       help="Print each example")

    args = parser.parse_args()

    # Initialize pipeline
    print("Initializing CHISEL Pipeline...")
    pipeline = ChiselPipeline(args.tables)

    # Load dev set
    print(f"\nLoading Spider dev set from {args.dev}...")
    dev_data = load_spider_dev(args.dev)
    print(f"Loaded {len(dev_data)} examples")

    # Evaluate
    print("\nStarting evaluation...")
    metrics, predictions = evaluate_on_spider(
        pipeline,
        dev_data,
        limit=args.limit,
        verbose=args.verbose
    )

    # Print results
    print_metrics(metrics)

    # Save predictions
    save_predictions(predictions, args.output)

    # Print expected performance comparison
    print("\nExpected Performance (from PRINCIPLES.md):")
    print("  Overall:   30-45%")
    print("  Easy:      ~60%")
    print("  Medium:    ~35%")
    print("  Hard:      ~20%")
    print("  Extra:     ~10%")
