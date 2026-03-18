"""
Debug script to identify which queries regressed from baseline to enhanced.

Runs both pipelines on the first 50 Spider questions and compares results.
"""

import json
from pathlib import Path
from chisel.pipeline import ChiselPipeline


def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison."""
    if not sql:
        return ""
    sql = sql.lower()
    sql = ' '.join(sql.split())
    sql = sql.rstrip(';')
    return sql


def main():
    """Debug the regression."""
    tables_path = "data/spider/tables.json"
    dev_path = "data/spider/dev.json"

    # Load dev data
    print("Loading Spider dev set...")
    with open(dev_path, 'r') as f:
        dev_data = json.load(f)[:50]  # First 50 questions

    print(f"Loaded {len(dev_data)} examples\n")

    # Create both pipelines
    print("Creating pipelines...")
    baseline_pipeline = ChiselPipeline(
        tables_path,
        use_ontology=False,
        use_templates=False,
        use_knowledge_base=False,
        use_lambda_dcs=False  # Disable Lambda DCS for baseline
    )

    enhanced_pipeline = ChiselPipeline(
        tables_path,
        use_ontology=True,
        use_templates=True,
        use_knowledge_base=False,
        use_lambda_dcs=True  # Enable Lambda DCS for enhanced
    )

    print("[OK] Pipelines created\n")

    # Track results
    baseline_correct = []
    enhanced_correct = []
    regressions = []
    improvements = []

    # Run both pipelines on each question
    for i, example in enumerate(dev_data):
        question = example['question']
        db_id = example['db_id']
        gold_sql = example['query']

        print(f"\n{'='*80}")
        print(f"[{i+1}/{len(dev_data)}] Question: {question}")
        print(f"DB: {db_id}")
        print(f"Gold SQL: {gold_sql}")
        print(f"{'='*80}")

        # Run baseline
        print("\n[BASELINE] Running v0.4 (no Lambda DCS)...")
        try:
            baseline_sql = baseline_pipeline.run(question, db_id, verbose=True)
            print(f"[BASELINE] Generated: {baseline_sql}")
        except Exception as e:
            baseline_sql = None
            print(f"[BASELINE] ERROR: {e}")

        # Run enhanced
        print("\n[ENHANCED] Running v0.5 (with Lambda DCS)...")
        try:
            enhanced_sql = enhanced_pipeline.run(question, db_id, verbose=True)
            print(f"[ENHANCED] Generated: {enhanced_sql}")
        except Exception as e:
            enhanced_sql = None
            print(f"[ENHANCED] ERROR: {e}")

        # Check exact match
        baseline_match = normalize_sql(baseline_sql) == normalize_sql(gold_sql)
        enhanced_match = normalize_sql(enhanced_sql) == normalize_sql(gold_sql)

        print(f"\n[RESULTS]")
        print(f"  Baseline correct: {baseline_match}")
        print(f"  Enhanced correct: {enhanced_match}")

        # Track results
        if baseline_match:
            baseline_correct.append(i)
        if enhanced_match:
            enhanced_correct.append(i)

        # Detect regressions and improvements
        if baseline_match and not enhanced_match:
            print(f"  ⚠ REGRESSION: Baseline was correct, enhanced is wrong!")
            regressions.append({
                "index": i,
                "question": question,
                "db_id": db_id,
                "gold_sql": gold_sql,
                "baseline_sql": baseline_sql,
                "enhanced_sql": enhanced_sql
            })
        elif not baseline_match and enhanced_match:
            print(f"  ✓ IMPROVEMENT: Baseline was wrong, enhanced is correct!")
            improvements.append({
                "index": i,
                "question": question,
                "db_id": db_id,
                "gold_sql": gold_sql,
                "baseline_sql": baseline_sql,
                "enhanced_sql": enhanced_sql
            })

    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Baseline correct: {len(baseline_correct)}/{len(dev_data)} ({100.0 * len(baseline_correct) / len(dev_data):.1f}%)")
    print(f"Enhanced correct: {len(enhanced_correct)}/{len(dev_data)} ({100.0 * len(enhanced_correct) / len(dev_data):.1f}%)")
    print(f"Regressions: {len(regressions)}")
    print(f"Improvements: {len(improvements)}")
    print(f"Net change: {len(improvements) - len(regressions)}")

    # Detailed regression analysis
    if regressions:
        print(f"\n{'='*80}")
        print("REGRESSION DETAILS")
        print(f"{'='*80}")
        for reg in regressions:
            print(f"\n[Question {reg['index']+1}] {reg['question']}")
            print(f"DB: {reg['db_id']}")
            print(f"Gold:     {reg['gold_sql']}")
            print(f"Baseline: {reg['baseline_sql']}")
            print(f"Enhanced: {reg['enhanced_sql']}")

    # Detailed improvement analysis
    if improvements:
        print(f"\n{'='*80}")
        print("IMPROVEMENT DETAILS")
        print(f"{'='*80}")
        for imp in improvements:
            print(f"\n[Question {imp['index']+1}] {imp['question']}")
            print(f"DB: {imp['db_id']}")
            print(f"Gold:     {imp['gold_sql']}")
            print(f"Baseline: {imp['baseline_sql']}")
            print(f"Enhanced: {imp['enhanced_sql']}")


if __name__ == "__main__":
    main()
