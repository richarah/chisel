# CHISEL: Compositional Heuristic Inference for SQL from English Language

A deterministic text-to-SQL solver. No LLM, no neural net, no trained weights. Rules + libraries only.

**Target**: Spider benchmark
**Approach**: Rules that compose powerful libraries

## Core Philosophy

**We write ~4,600 lines of glue code. Libraries write ~500K lines.** (1:109 ratio)

- [PRINCIPLES.md](PRINCIPLES.md) - Design philosophy and constraints
- [PEAK_PERFORMANCE.md](PEAK_PERFORMANCE.md) - No graceful degradation principle
- [SYSTEM_STATUS.md](SYSTEM_STATUS.md) - Current implementation status (v0.4)
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guidelines

## Setup

**Quick Start:**
```bash
./setup.sh  # Automated setup (steps 1-2)
python test_standalone.py  # Test installation
```

**Manual Setup:**
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('stopwords')"

# 3. Test (no Spider data required)
python test_standalone.py
```

**For Evaluation (Optional):**
- Download Spider: https://yale-lily.github.io/spider
- Extract to `data/spider/`
- See [DATASETS_STATUS.md](DATASETS_STATUS.md) for dataset information

## Quick Start

```python
from chisel.pipeline import ChiselPipeline

# Initialize pipeline
pipeline = ChiselPipeline("data/spider/tables.json")

# Run on a question
question = "How many students are there?"
db_id = "student_1"
sql = pipeline.run(question, db_id, verbose=True)
print(sql)
# Output: SELECT COUNT(*) FROM student
```

## Command Line Usage

```bash
# Run pipeline test
python -m chisel.pipeline data/spider/tables.json

# Run tests
python -m pytest tests/

# Evaluate on Spider dev set
python evaluation/evaluate.py --tables data/spider/tables.json --dev data/spider/dev.json

# Error analysis
python evaluation/error_analysis.py predictions.json
```

## Architecture

### Enhanced Libraries (the heavy lifters):
- **spacy** -> tokenize, POS tag, dependency parse, NER
- **lemminflect** -> better lemmatization (95.6% accuracy vs spaCy's 84.7%)
- **nltk.wordnet** -> synonym expansion (155k words), semantic similarity
- **word2number** -> parse number words ("twenty" -> 20)
- **dateparser** -> parse date expressions ("last year" -> datetime)
- **inflect** -> pluralization/singularization ("countries" -> "country")
- **sqlglot** -> parse SQL, build SQL ASTs, validate, optimize
- **rapidfuzz** -> fuzzy string matching (WRatio algorithm)
- **networkx** -> FK graph, shortest_path for join resolution

### Our Code (the glue):
1. **Schema Graph** (Task 1): Parse DDL -> FK graph
2. **Question Analysis** (Task 2): spaCy parse -> SQL signals
3. **Schema Linking** (Task 3): Match question words -> schema elements
4. **Skeleton Prediction** (Task 4): Signals -> SQL clause structure
5. **Slot Filling** (Task 5): Fill skeleton, resolve joins, build AST
6. **Validate & Repair** (Task 6): Parse back, fix errors
7. **Evaluation** (Task 7): Run on Spider dev set
8. **Error-Driven Improvement** (Task 8): Iterative rule refinement

## System Status

**Version**: v0.5 (rule-based enhancements)
**Modules**: 17 modules, 6,800+ lines
**Test Coverage**: 770+ lines of tests

**Features Implemented**:
- Core pipeline (6 components)
- Compositional semantics (CCG + FOL)
- Advanced SQL generation (comparatives, joins, negation, sets)
- Multi-turn dialogue (coreference resolution)
- **NEW in v0.5**:
  - Ontology-based schema linking (COMA++ composite matching)
  - SQL template library (20+ patterns for rare constructs)
  - Temporal normalization (TIMEX3 standard)
  - Knowledge base integration (DBpedia/Wikidata/GeoNames)

**Benchmark Results** (Question Understanding):
- Spider dev: 95%+ feature detection accuracy
- SParc dev: 95%+ feature detection accuracy
- See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for details

**Expected SQL Generation** (v0.5 projected):
- Overall: 60-75% exact match (Phase 1 improvements)
- Phase 2 target: 75-80% (with knowledge bases)
- Phase 3 target: 80-85% (deterministic ceiling)
- Simple SELECT: 70-80%
- Aggregations: 65-75%
- Multi-table joins: 55-65%
- Complex queries: 40-50%

## Contributing

All rules must be:
1. **General** (not Spider-specific)
2. **Deterministic** (same input -> same output)
3. **Library-first** (use existing tools before writing code)

See [PRINCIPLES.md](PRINCIPLES.md) for details.

## License

MIT
