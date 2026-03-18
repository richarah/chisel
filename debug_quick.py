"""Quick debug - test first 3 queries only."""

import json
from chisel.pipeline import ChiselPipeline


def normalize_sql(sql: str) -> str:
    if not sql:
        return ""
    sql = sql.lower().strip()
    sql = ' '.join(sql.split())
    sql = sql.rstrip(';')
    return sql


# Load first 3 examples
with open("data/spider/dev.json", 'r') as f:
    dev_data = json.load(f)[:3]

print(f"Testing {len(dev_data)} examples\n")

# Create pipelines
print("Creating baseline (no Lambda DCS)...")
baseline = ChiselPipeline(
    "data/spider/tables.json",
    use_ontology=False,
    use_templates=False,
    use_lambda_dcs=False
)

print("Creating enhanced (with Lambda DCS)...\n")
enhanced = ChiselPipeline(
    "data/spider/tables.json",
    use_ontology=False,
    use_templates=False,
    use_lambda_dcs=True
)

for i, ex in enumerate(dev_data):
    print(f"\n{'='*80}")
    print(f"[{i+1}] {ex['question']}")
    print(f"DB: {ex['db_id']}")
    print(f"Gold: {ex['query']}")
    print(f"{'='*80}")

    # Baseline
    print("\n[BASELINE - No Lambda DCS]")
    base_sql = baseline.run(ex['question'], ex['db_id'], verbose=True)
    print(f"Result: {base_sql}")
    base_match = normalize_sql(base_sql) == normalize_sql(ex['query'])
    print(f"Correct: {base_match}")

    # Enhanced
    print("\n[ENHANCED - With Lambda DCS]")
    enh_sql = enhanced.run(ex['question'], ex['db_id'], verbose=True)
    print(f"Result: {enh_sql}")
    enh_match = normalize_sql(enh_sql) == normalize_sql(ex['query'])
    print(f"Correct: {enh_match}")

    # Compare
    if base_match and not enh_match:
        print("\n⚠️ REGRESSION: Lambda DCS broke this query!")
    elif not base_match and enh_match:
        print("\n✅ IMPROVEMENT: Lambda DCS fixed this query!")
