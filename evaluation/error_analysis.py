"""
Task 8: Error Analysis

Categorize errors to guide rule improvements.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from collections import defaultdict, Counter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chisel.validation import check_sql_features


def load_predictions(predictions_path: str) -> List[Dict]:
    """Load predictions from JSON."""
    with open(predictions_path, 'r') as f:
        return json.load(f)


def categorize_error(pred: Dict) -> str:
    """
    Categorize error type.

    Categories:
    - no_sql: Failed to generate SQL
    - invalid_sql: Generated invalid SQL
    - missing_join: Gold has JOIN, predicted doesn't
    - missing_where: Gold has WHERE, predicted doesn't
    - missing_aggregation: Gold has aggregation, predicted doesn't
    - wrong_column: Wrong columns selected
    - wrong_table: Wrong tables used
    - other: Other mismatch
    """
    if not pred["predicted"]:
        return "no_sql"

    if not pred.get("valid", True):
        return "invalid_sql"

    if pred["correct"]:
        return "correct"

    # Compare features
    try:
        gold_features = check_sql_features(pred["gold"])
        pred_features = check_sql_features(pred["predicted"])

        # Check for missing features
        if gold_features["has_join"] and not pred_features["has_join"]:
            return "missing_join"

        if gold_features["has_where"] and not pred_features["has_where"]:
            return "missing_where"

        if gold_features["has_aggregation"] and not pred_features["has_aggregation"]:
            return "missing_aggregation"

        if gold_features["has_group_by"] and not pred_features["has_group_by"]:
            return "missing_group_by"

        if gold_features["has_order_by"] and not pred_features["has_order_by"]:
            return "missing_order_by"

        # Extra features
        if not gold_features["has_join"] and pred_features["has_join"]:
            return "extra_join"

        if not gold_features["has_where"] and pred_features["has_where"]:
            return "extra_where"

        # Column count mismatch
        if gold_features["column_count"] != pred_features["column_count"]:
            return "wrong_column_count"

        # Table count mismatch
        if gold_features["table_count"] != pred_features["table_count"]:
            return "wrong_table_count"

    except:
        pass

    return "other"


def analyze_errors(predictions: List[Dict]) -> Dict:
    """
    Analyze prediction errors.

    Returns dict with:
    - error_categories: Counter of error types
    - errors_by_difficulty: Error types per difficulty level
    - example_errors: Sample errors for each category
    """
    error_categories = Counter()
    errors_by_difficulty = defaultdict(Counter)
    example_errors = defaultdict(list)

    for pred in predictions:
        category = categorize_error(pred)
        error_categories[category] += 1

        difficulty = pred.get("difficulty", "unknown")
        errors_by_difficulty[difficulty][category] += 1

        # Store examples (max 3 per category)
        if len(example_errors[category]) < 3:
            example_errors[category].append({
                "question": pred["question"],
                "gold": pred["gold"],
                "predicted": pred.get("predicted"),
                "db_id": pred["db_id"]
            })

    analysis = {
        "error_categories": error_categories,
        "errors_by_difficulty": errors_by_difficulty,
        "example_errors": dict(example_errors)
    }

    return analysis


def print_error_analysis(analysis: Dict, predictions: List[Dict]):
    """Print error analysis report."""
    print("\n" + "="*80)
    print("ERROR ANALYSIS")
    print("="*80)

    total = len(predictions)
    error_categories = analysis["error_categories"]

    print(f"\nError Categories (Total: {total}):")
    for category, count in error_categories.most_common():
        pct = count / total * 100
        print(f"  {category:25s}: {count:4d} ({pct:5.1f}%)")

    print(f"\nErrors by Difficulty:")
    for difficulty, counts in analysis["errors_by_difficulty"].items():
        print(f"\n  {difficulty.upper()}:")
        for category, count in counts.most_common(5):
            print(f"    {category:25s}: {count:4d}")

    print(f"\n" + "="*80)
    print("EXAMPLE ERRORS (for rule improvement)")
    print("="*80)

    for category, examples in analysis["example_errors"].items():
        if category == "correct":
            continue

        print(f"\n{category.upper()}:")
        for i, ex in enumerate(examples, 1):
            print(f"\n  Example {i}:")
            print(f"    Question:  {ex['question']}")
            print(f"    Database:  {ex['db_id']}")
            print(f"    Gold:      {ex['gold']}")
            print(f"    Predicted: {ex['predicted']}")


def suggest_improvements(analysis: Dict):
    """
    Suggest rule improvements based on error categories.
    """
    print("\n" + "="*80)
    print("SUGGESTED IMPROVEMENTS")
    print("="*80)

    error_categories = analysis["error_categories"]

    suggestions = []

    # Prioritize by frequency
    for category, count in error_categories.most_common(5):
        if category == "correct":
            continue

        if category == "no_sql":
            suggestions.append({
                "category": category,
                "count": count,
                "suggestion": "Improve schema linking - many questions fail to link to any schema elements. "
                             "Consider: (1) Better fuzzy matching thresholds, (2) More synonym expansion, "
                             "(3) Handling common paraphrases."
            })

        elif category == "missing_join":
            suggestions.append({
                "category": category,
                "count": count,
                "suggestion": "Multi-table queries not detected. Improve schema linking to identify multiple "
                             "tables from question words. Check FK graph coverage."
            })

        elif category == "missing_where":
            suggestions.append({
                "category": category,
                "count": count,
                "suggestion": "Value detection failing. Improve: (1) NER for values, (2) Number/date parsing, "
                             "(3) Quoted string detection."
            })

        elif category == "missing_aggregation":
            suggestions.append({
                "category": category,
                "count": count,
                "suggestion": "Aggregation signals not detected. Expand SQL indicator seeds in question_analysis.py "
                             "with more synonyms for COUNT/MAX/MIN/AVG/SUM."
            })

        elif category == "missing_group_by":
            suggestions.append({
                "category": category,
                "count": count,
                "suggestion": "GROUP BY logic needs improvement. Check detection of 'each', 'every', 'per' patterns. "
                             "Review skeleton_prediction.py rules."
            })

    print("\nTop priorities for rule improvements:\n")
    for i, sugg in enumerate(suggestions, 1):
        print(f"{i}. {sugg['category'].upper()} ({sugg['count']} errors)")
        print(f"   -> {sugg['suggestion']}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze CHISEL errors")
    parser.add_argument("predictions", help="Path to predictions.json from evaluate.py")
    parser.add_argument("--output", default="error_analysis.json",
                       help="Output file for error analysis")

    args = parser.parse_args()

    # Load predictions
    print(f"Loading predictions from {args.predictions}...")
    predictions = load_predictions(args.predictions)
    print(f"Loaded {len(predictions)} predictions")

    # Analyze errors
    print("\nAnalyzing errors...")
    analysis = analyze_errors(predictions)

    # Print analysis
    print_error_analysis(analysis, predictions)

    # Suggest improvements
    suggest_improvements(analysis)

    # Save analysis
    # Convert Counter to dict for JSON serialization
    analysis_for_json = {
        "error_categories": dict(analysis["error_categories"]),
        "errors_by_difficulty": {
            diff: dict(counts) for diff, counts in analysis["errors_by_difficulty"].items()
        },
        "example_errors": analysis["example_errors"]
    }

    with open(args.output, 'w') as f:
        json.dump(analysis_for_json, f, indent=2)

    print(f"\nError analysis saved to: {args.output}")
