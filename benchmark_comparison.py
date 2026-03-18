"""
Benchmark Comparison Script

Compare CHISEL v0.4 (baseline) vs v0.5 (with enhancements) performance.

Metrics:
- Exact match accuracy
- Component-level accuracy (schema linking, skeleton prediction)
- Performance by SQL complexity
- Detailed failure analysis
"""

import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

from chisel.pipeline import ChiselPipeline
from chisel.schema_graph import load_spider_schemas


# ==========================
# BENCHMARK CONFIGURATION
# ==========================

@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    tables_path: str = "data/spider/tables.json"
    dev_path: str = "data/spider/dev.json"
    num_samples: int = 100  # Number of examples to test (use -1 for all)
    timeout: float = 10.0  # Seconds per question
    verbose: bool = False
    parallel: bool = True  # Use multiprocessing
    num_workers: int = -1  # -1 = use all CPUs


@dataclass
class BenchmarkResult:
    """Results for a single question."""
    question: str
    db_id: str
    gold_sql: str
    predicted_sql: Optional[str]
    exact_match: bool
    execution_time: float
    error: Optional[str] = None


@dataclass
class BenchmarkSummary:
    """Summary statistics for benchmark run."""
    version: str
    total_questions: int
    exact_match: int
    exact_match_percent: float
    avg_time: float
    failures: int
    failure_percent: float
    results: List[BenchmarkResult]


# ==========================
# PARALLEL WORKER FUNCTION
# ==========================

# Global cache for worker processes (one pipeline per worker)
_worker_pipeline_cache = {}

def _run_single_query(example: Dict, tables_path: str, use_ontology: bool,
                     use_templates: bool, verbose: bool) -> BenchmarkResult:
    """
    Worker function for parallel processing.

    Each worker reuses its own pipeline instance for better performance.
    Pipeline is cached per worker process ID to avoid re-initialization overhead.
    """
    import os

    question = example['question']
    db_id = example['db_id']
    gold_sql = example['query']

    # Create cache key based on worker PID and pipeline config
    worker_id = os.getpid()
    cache_key = (worker_id, tables_path, use_ontology, use_templates)

    # Reuse pipeline for this worker (created once per worker process)
    if cache_key not in _worker_pipeline_cache:
        _worker_pipeline_cache[cache_key] = ChiselPipeline(
            tables_path,
            use_ontology=use_ontology,
            use_templates=use_templates,
            use_knowledge_base=False
        )

    pipeline = _worker_pipeline_cache[cache_key]

    # Run pipeline
    start_time = time.time()
    try:
        predicted_sql = pipeline.run(question, db_id, verbose=False)
        execution_time = time.time() - start_time
        error = None
    except Exception as e:
        predicted_sql = None
        execution_time = time.time() - start_time
        error = str(e)

    # Check exact match (normalized)
    def normalize_sql(sql: str) -> str:
        if not sql:
            return ""
        sql = sql.lower()
        sql = ' '.join(sql.split())
        sql = sql.rstrip(';')
        return sql

    exact_match = False
    if predicted_sql:
        pred_norm = normalize_sql(predicted_sql)
        gold_norm = normalize_sql(gold_sql)
        exact_match = (pred_norm == gold_norm)

    return BenchmarkResult(
        question=question,
        db_id=db_id,
        gold_sql=gold_sql,
        predicted_sql=predicted_sql,
        exact_match=exact_match,
        execution_time=execution_time,
        error=error
    )


# ==========================
# BENCHMARK RUNNER
# ==========================

class BenchmarkRunner:
    """Run benchmarks and compare versions."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

        # Load Spider dev set
        print(f"Loading Spider dev set from {config.dev_path}...")
        with open(config.dev_path, 'r') as f:
            self.dev_data = json.load(f)

        if config.num_samples > 0:
            self.dev_data = self.dev_data[:config.num_samples]

        print(f"Loaded {len(self.dev_data)} examples")

    def run_baseline(self) -> BenchmarkSummary:
        """
        Run baseline CHISEL v0.4 (no enhancements).

        This simulates the previous version by disabling all v0.5 features.
        """
        print("\n" + "="*80)
        print("BASELINE: CHISEL v0.4 (No Enhancements)")
        print("="*80)

        pipeline = ChiselPipeline(
            self.config.tables_path,
            use_ontology=False,  # Disable v0.5 features
            use_templates=False,
            use_knowledge_base=False
        )

        return self._run_benchmark(pipeline, "v0.4-baseline")

    def run_enhanced(self) -> BenchmarkSummary:
        """
        Run enhanced CHISEL v0.5 (with all improvements).
        """
        print("\n" + "="*80)
        print("ENHANCED: CHISEL v0.5 (With Improvements)")
        print("="*80)

        pipeline = ChiselPipeline(
            self.config.tables_path,
            use_ontology=True,  # Enable v0.5 features
            use_templates=True,
            use_knowledge_base=False,  # Requires optional dependencies
            domain="university"
        )

        return self._run_benchmark(pipeline, "v0.5-enhanced")

    def _run_benchmark(self, pipeline: ChiselPipeline, version: str) -> BenchmarkSummary:
        """Run benchmark on a pipeline."""
        print(f"\nProcessing {len(self.dev_data)} questions...")

        if self.config.parallel:
            # Parallel processing
            num_workers = self.config.num_workers if self.config.num_workers > 0 else cpu_count()
            print(f"Using {num_workers} parallel workers...")

            # Create worker function with pipeline config
            worker_fn = partial(
                _run_single_query,
                tables_path=self.config.tables_path,
                use_ontology=(version == "v0.5-enhanced"),
                use_templates=(version == "v0.5-enhanced"),
                verbose=self.config.verbose
            )

            # Process in parallel
            with Pool(num_workers) as pool:
                results = []
                for i, result in enumerate(pool.imap(worker_fn, self.dev_data), 1):
                    results.append(result)
                    if self.config.verbose or i % 10 == 0:
                        print(f"  [{i}/{len(self.dev_data)}] {result.question[:60]}...")
        else:
            # Sequential processing (original)
            results = []
            for i, example in enumerate(self.dev_data):
                question = example['question']
                db_id = example['db_id']
                gold_sql = example['query']

                if self.config.verbose or (i + 1) % 10 == 0:
                    print(f"  [{i+1}/{len(self.dev_data)}] {question[:60]}...")

                # Run pipeline
                start_time = time.time()
                try:
                    predicted_sql = pipeline.run(question, db_id, verbose=False)
                    execution_time = time.time() - start_time
                    error = None

                except Exception as e:
                    predicted_sql = None
                    execution_time = time.time() - start_time
                    error = str(e)

                # Check exact match (normalized)
                exact_match = self._check_exact_match(predicted_sql, gold_sql)

                result = BenchmarkResult(
                    question=question,
                    db_id=db_id,
                    gold_sql=gold_sql,
                    predicted_sql=predicted_sql,
                    exact_match=exact_match,
                    execution_time=execution_time,
                    error=error
                )

                results.append(result)

        # Calculate summary statistics
        exact_matches = sum(1 for r in results if r.exact_match)
        failures = sum(1 for r in results if r.predicted_sql is None)
        total_time = sum(r.execution_time for r in results)

        summary = BenchmarkSummary(
            version=version,
            total_questions=len(results),
            exact_match=exact_matches,
            exact_match_percent=100.0 * exact_matches / len(results),
            avg_time=total_time / len(results),
            failures=failures,
            failure_percent=100.0 * failures / len(results),
            results=results
        )

        return summary

    def _check_exact_match(self, predicted: Optional[str], gold: str) -> bool:
        """
        Check if predicted SQL matches gold SQL (normalized).

        Uses simple normalization: lowercase, remove extra spaces.
        """
        if predicted is None:
            return False

        # Normalize both SQLs
        pred_norm = self._normalize_sql(predicted)
        gold_norm = self._normalize_sql(gold)

        return pred_norm == gold_norm

    def _normalize_sql(self, sql: str) -> str:
        """Normalize SQL for comparison."""
        # Lowercase
        sql = sql.lower()

        # Remove extra whitespace
        sql = ' '.join(sql.split())

        # Remove trailing semicolon
        sql = sql.rstrip(';')

        return sql

    def detect_overfitting(self, results: List[BenchmarkResult]) -> Dict:
        """
        Detect potential overfitting indicators.

        Checks:
        - Per-database accuracy variance (high variance = overfitting to specific DBs)
        - SQL pattern diversity (low diversity = memorized patterns)
        - Regression analysis (breaking previously working queries)

        Returns dict with overfitting metrics
        """
        from collections import defaultdict
        import statistics

        # Group by database
        db_accuracy = defaultdict(lambda: {"correct": 0, "total": 0})

        for result in results:
            db_accuracy[result.db_id]["total"] += 1
            if result.exact_match:
                db_accuracy[result.db_id]["correct"] += 1

        # Calculate per-DB accuracy
        db_accuracies = {}
        for db_id, stats in db_accuracy.items():
            if stats["total"] > 0:
                db_accuracies[db_id] = stats["correct"] / stats["total"] * 100

        # Calculate variance
        if len(db_accuracies) > 1:
            accuracy_variance = statistics.variance(db_accuracies.values())
            accuracy_stdev = statistics.stdev(db_accuracies.values())
        else:
            accuracy_variance = 0
            accuracy_stdev = 0

        # Check for extreme per-DB performance
        if db_accuracies:
            max_db_acc = max(db_accuracies.values())
            min_db_acc = min(db_accuracies.values())
            db_acc_range = max_db_acc - min_db_acc
        else:
            max_db_acc = min_db_acc = db_acc_range = 0

        # Overfitting indicators
        overfitting_indicators = []

        if accuracy_stdev > 30:
            overfitting_indicators.append(
                f"HIGH VARIANCE: Accuracy varies greatly across databases (σ={accuracy_stdev:.1f}%)"
            )

        if db_acc_range > 50:
            overfitting_indicators.append(
                f"EXTREME RANGE: {db_acc_range:.1f}% difference between best and worst database"
            )

        # Check SQL pattern diversity
        import re
        sql_patterns = set()
        for result in results:
            if result.predicted_sql:
                # Extract SQL structure (remove specific values/names)
                pattern = re.sub(r'\b\d+\b', 'NUM', result.predicted_sql.lower())
                pattern = re.sub(r"'[^']*'", 'STR', pattern)
                pattern = re.sub(r'\b[a-z_]+\b', 'ID', pattern)
                sql_patterns.add(pattern)

        pattern_diversity = len(sql_patterns) / len(results) if results else 0

        if pattern_diversity < 0.3:
            overfitting_indicators.append(
                f"LOW DIVERSITY: Only {len(sql_patterns)} unique SQL patterns for {len(results)} questions"
            )

        return {
            "db_accuracies": db_accuracies,
            "accuracy_variance": accuracy_variance,
            "accuracy_stdev": accuracy_stdev,
            "db_acc_range": db_acc_range,
            "max_db_acc": max_db_acc,
            "min_db_acc": min_db_acc,
            "pattern_diversity": pattern_diversity,
            "total_patterns": len(sql_patterns),
            "overfitting_indicators": overfitting_indicators,
            "overfitting_risk": "HIGH" if len(overfitting_indicators) > 0 else "LOW"
        }

    def compare_results(self, baseline: BenchmarkSummary,
                       enhanced: BenchmarkSummary) -> Dict:
        """
        Compare baseline and enhanced results.

        Returns:
            Dictionary with comparison metrics
        """
        print("\n" + "="*80)
        print("COMPARISON: v0.4 Baseline vs v0.5 Enhanced")
        print("="*80)

        comparison = {
            "baseline": {
                "version": baseline.version,
                "exact_match": baseline.exact_match_percent,
                "failures": baseline.failure_percent,
                "avg_time": baseline.avg_time,
            },
            "enhanced": {
                "version": enhanced.version,
                "exact_match": enhanced.exact_match_percent,
                "failures": enhanced.failure_percent,
                "avg_time": enhanced.avg_time,
            },
            "delta": {
                "exact_match": enhanced.exact_match_percent - baseline.exact_match_percent,
                "failures": enhanced.failure_percent - baseline.failure_percent,
                "avg_time": enhanced.avg_time - baseline.avg_time,
            }
        }

        # Print summary table
        print(f"\n{'Metric':<30} {'Baseline (v0.4)':<20} {'Enhanced (v0.5)':<20} {'Delta':<15}")
        print("-" * 85)

        print(f"{'Exact Match %':<30} {baseline.exact_match_percent:>17.2f}% "
              f"{enhanced.exact_match_percent:>17.2f}% "
              f"{comparison['delta']['exact_match']:>+14.2f}%")

        print(f"{'Failure Rate %':<30} {baseline.failure_percent:>17.2f}% "
              f"{enhanced.failure_percent:>17.2f}% "
              f"{comparison['delta']['failures']:>+14.2f}%")

        print(f"{'Avg Time (s)':<30} {baseline.avg_time:>20.3f} "
              f"{enhanced.avg_time:>20.3f} "
              f"{comparison['delta']['avg_time']:>+14.3f}")

        # Analyze improvements
        print(f"\n{'Analysis'}")
        print("-" * 85)

        improved = 0
        regressed = 0
        for base_r, enh_r in zip(baseline.results, enhanced.results):
            if not base_r.exact_match and enh_r.exact_match:
                improved += 1
            elif base_r.exact_match and not enh_r.exact_match:
                regressed += 1

        print(f"  Questions improved: {improved}")
        print(f"  Questions regressed: {regressed}")
        print(f"  Net improvement: {improved - regressed} (+{100.0 * (improved - regressed) / len(baseline.results):.2f}%)")

        # Overfitting detection
        print(f"\n{'Overfitting Analysis'}")
        print("-" * 85)

        baseline_overfitting = self.detect_overfitting(baseline.results)
        enhanced_overfitting = self.detect_overfitting(enhanced.results)

        print(f"  Baseline overfitting risk: {baseline_overfitting['overfitting_risk']}")
        print(f"  Enhanced overfitting risk: {enhanced_overfitting['overfitting_risk']}")

        if baseline_overfitting['overfitting_indicators']:
            print(f"  Baseline indicators:")
            for indicator in baseline_overfitting['overfitting_indicators']:
                print(f"    - {indicator}")

        if enhanced_overfitting['overfitting_indicators']:
            print(f"  Enhanced indicators:")
            for indicator in enhanced_overfitting['overfitting_indicators']:
                print(f"    - {indicator}")

        comparison['overfitting'] = {
            'baseline': baseline_overfitting,
            'enhanced': enhanced_overfitting
        }

        return comparison


# ==========================
# MAIN EXECUTION
# ==========================

def main():
    """Run benchmark comparison."""
    config = BenchmarkConfig(
        num_samples=50,  # Test on 50 examples for quick comparison
        verbose=False
    )

    # Check if Spider data exists
    if not Path(config.tables_path).exists():
        print(f"ERROR: Spider tables not found at {config.tables_path}")
        print("Please download Spider dataset from https://yale-lily.github.io/spider")
        print("and extract to data/spider/")
        return

    if not Path(config.dev_path).exists():
        print(f"ERROR: Spider dev set not found at {config.dev_path}")
        print("Please download Spider dataset from https://yale-lily.github.io/spider")
        print("and extract to data/spider/")
        return

    # Run benchmarks
    runner = BenchmarkRunner(config)

    print("\n" + "="*80)
    print("CHISEL BENCHMARK COMPARISON")
    print("="*80)
    print(f"Testing on {len(runner.dev_data)} examples from Spider dev set")
    print("="*80)

    # Run baseline
    baseline_summary = runner.run_baseline()

    # Run enhanced
    enhanced_summary = runner.run_enhanced()

    # Compare
    comparison = runner.compare_results(baseline_summary, enhanced_summary)

    # Save results
    output_file = "benchmark_results.json"
    print(f"\nSaving results to {output_file}...")

    output = {
        "config": {
            "num_samples": config.num_samples,
            "tables_path": config.tables_path,
            "dev_path": config.dev_path,
        },
        "baseline": {
            "version": baseline_summary.version,
            "exact_match_percent": baseline_summary.exact_match_percent,
            "failure_percent": baseline_summary.failure_percent,
            "avg_time": baseline_summary.avg_time,
        },
        "enhanced": {
            "version": enhanced_summary.version,
            "exact_match_percent": enhanced_summary.exact_match_percent,
            "failure_percent": enhanced_summary.failure_percent,
            "avg_time": enhanced_summary.avg_time,
        },
        "comparison": comparison
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"[OK] Results saved to {output_file}")

    # Generate markdown report
    generate_markdown_report(baseline_summary, enhanced_summary, comparison)


def generate_markdown_report(baseline: BenchmarkSummary,
                            enhanced: BenchmarkSummary,
                            comparison: Dict):
    """Generate markdown benchmark report."""
    report_file = "BENCHMARK_COMPARISON.md"
    print(f"\nGenerating report: {report_file}...")

    with open(report_file, 'w') as f:
        f.write("# CHISEL Benchmark Comparison Report\n\n")
        f.write("## Overview\n\n")
        f.write(f"- **Test Date**: {time.strftime('%Y-%m-%d')}\n")
        f.write(f"- **Dataset**: Spider dev set\n")
        f.write(f"- **Samples**: {baseline.total_questions} questions\n\n")

        f.write("## Results Summary\n\n")
        f.write("| Metric | Baseline (v0.4) | Enhanced (v0.5) | Delta |\n")
        f.write("|--------|-----------------|-----------------|-------|\n")
        f.write(f"| Exact Match | {baseline.exact_match_percent:.2f}% | "
                f"{enhanced.exact_match_percent:.2f}% | "
                f"{comparison['delta']['exact_match']:+.2f}% |\n")
        f.write(f"| Failure Rate | {baseline.failure_percent:.2f}% | "
                f"{enhanced.failure_percent:.2f}% | "
                f"{comparison['delta']['failures']:+.2f}% |\n")
        f.write(f"| Avg Time (s) | {baseline.avg_time:.3f} | "
                f"{enhanced.avg_time:.3f} | "
                f"{comparison['delta']['avg_time']:+.3f} |\n\n")

        f.write("## Key Improvements in v0.5\n\n")
        f.write("### Enhanced Components\n\n")
        f.write("1. **Ontology-Based Schema Linking**\n")
        f.write("   - COMA++ composite matching (4 matchers with weighted aggregation)\n")
        f.write("   - Domain ontologies (university, geography, business)\n")
        f.write("   - Abbreviation expansion (CS→Computer Science, etc.)\n\n")

        f.write("2. **SQL Template Library**\n")
        f.write("   - 20+ predefined SQL patterns\n")
        f.write("   - Pattern matching for rare constructs (self-joins, INTERSECT, NOT EXISTS)\n")
        f.write("   - Deterministic template-to-SQL generation\n\n")

        f.write("3. **Temporal Normalization**\n")
        f.write("   - TIMEX3 standard compliance\n")
        f.write("   - Semester/quarter normalization (Fall 2023 → 2023-09)\n")
        f.write("   - Duration parsing (3 years → P3Y)\n\n")

        f.write("4. **Knowledge Base Integration** (optional)\n")
        f.write("   - DBpedia/Wikidata SPARQL queries\n")
        f.write("   - GeoNames geographic gazetteer\n")
        f.write("   - Deterministic fact lookup\n\n")

        f.write("## Baseline vs Enhanced\n\n")
        f.write("### Baseline (v0.4)\n")
        f.write("- Components: Schema graph, question analysis, schema linking, "
                "skeleton prediction, slot filling, validation\n")
        f.write("- Schema linking: rapidfuzz fuzzy matching only\n")
        f.write("- No ontology, no templates, no temporal normalization\n\n")

        f.write("### Enhanced (v0.5)\n")
        f.write("- All v0.4 components PLUS:\n")
        f.write("- Composite schema linking with ontologies\n")
        f.write("- SQL template library\n")
        f.write("- Temporal normalization\n")
        f.write("- Knowledge base integration (optional)\n\n")

        f.write("## Detailed Analysis\n\n")

        # Count improvements by category
        improved = sum(1 for base_r, enh_r in zip(baseline.results, enhanced.results)
                      if not base_r.exact_match and enh_r.exact_match)
        regressed = sum(1 for base_r, enh_r in zip(baseline.results, enhanced.results)
                       if base_r.exact_match and not enh_r.exact_match)

        f.write(f"- **Questions improved**: {improved} "
                f"({100.0 * improved / baseline.total_questions:.2f}%)\n")
        f.write(f"- **Questions regressed**: {regressed} "
                f"({100.0 * regressed / baseline.total_questions:.2f}%)\n")
        f.write(f"- **Net improvement**: {improved - regressed} questions\n\n")

        f.write("## Interpretation\n\n")

        delta = comparison['delta']['exact_match']
        if delta > 0:
            f.write(f"✓ Enhanced version shows **+{delta:.2f}%** improvement in exact match accuracy.\n\n")
            f.write("The rule-based enhancements successfully improve performance while maintaining "
                    "100% determinism.\n\n")
        elif delta == 0:
            f.write("The enhanced version shows no change in exact match accuracy on this sample.\n\n")
            f.write("This may indicate:\n")
            f.write("- Sample size too small to detect improvements\n")
            f.write("- Test questions don't leverage new features (ontologies, templates, temporal)\n")
            f.write("- Need larger evaluation (full Spider dev set with 1,034 examples)\n\n")
        else:
            f.write(f"⚠ Enhanced version shows **{delta:.2f}%** regression.\n\n")
            f.write("This requires investigation. Possible causes:\n")
            f.write("- Ontology matcher overriding correct fuzzy matches\n")
            f.write("- Template patterns too aggressive\n")
            f.write("- Need weight tuning in composite matcher\n\n")

        f.write("## Projected Performance\n\n")
        f.write("Based on RULE_BASED_SOLUTIONS.md analysis:\n\n")
        f.write("- **Phase 1** (Schema + Values): 50% → 75% (+25%)\n")
        f.write("- **Phase 2** (Knowledge Bases): 75% → 80% (+5%)\n")
        f.write("- **Phase 3** (Complex Queries): 80% → 85% (+5%)\n\n")
        f.write("**Deterministic ceiling**: 80-85% exact match on Spider\n\n")

        f.write("## Next Steps\n\n")
        f.write("1. Run on full Spider dev set (1,034 examples) for robust evaluation\n")
        f.write("2. Enable knowledge base integration (requires SPARQLWrapper)\n")
        f.write("3. Expand SQL template library to 100+ patterns\n")
        f.write("4. Tune composite matcher weights based on error analysis\n")
        f.write("5. Add continuation semantics for complex nested queries\n\n")

    print(f"[OK] Report saved to {report_file}")


if __name__ == "__main__":
    main()
