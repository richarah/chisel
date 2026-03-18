"""
Task 3: Schema Linking

Match question words to schema elements (tables, columns).

This is the hardest task. We use multiple strategies:
1. EXACT: token == column_name
2. FUZZY: rapidfuzz.fuzz.WRatio > threshold (handles typos, abbreviations)
3. SYNONYM: WordNet synset overlap (handles paraphrases)
4. VALUE: NER entities, numbers, quoted strings -> WHERE values

Libraries doing the heavy lifting:
- rapidfuzz: Fuzzy string matching (100x faster than difflib)
- nltk.wordnet: Synonym matching
- spaCy: Already parsed tokens and entities

We write: matching strategies + scoring + deduplication
"""

from dataclasses import dataclass
from typing import List, Tuple, Set, Dict, Optional
from enum import Enum

# Import required libraries
from rapidfuzz import fuzz, process
from nltk.corpus import wordnet

from .schema_graph import SchemaGraph
from .question_analysis import QuestionAnalysis


class LinkType(Enum):
    """Type of schema link."""
    EXACT = "exact"          # Exact match
    FUZZY = "fuzzy"          # Fuzzy string match
    SYNONYM = "synonym"      # WordNet synonym match
    PARTIAL = "partial"      # Partial word match
    VALUE = "value"          # Entity/value for WHERE clause


@dataclass
class SchemaLink:
    """
    A link between a question phrase and a schema element.
    """
    question_phrase: str          # Original phrase from question
    schema_element: str           # table.column or table
    element_type: str             # "table", "column"
    table_name: Optional[str]     # Table name (if column, which table)
    column_name: Optional[str]    # Column name (if element is column)
    score: float                  # Match confidence (0-100)
    link_type: LinkType           # How it was matched
    is_value: bool = False        # True if this is a value, not a column reference
    value: Optional[str] = None   # The actual value (if is_value=True)


def generate_ngrams_with_spans(doc, max_n: int = 3) -> List[Tuple[str, Tuple[int, int]]]:
    """
    Generate n-grams from spaCy Doc with span information.

    Uses spaCy's span functionality which is more efficient and respects
    token boundaries properly.

    Args:
        doc: spaCy Doc object
        max_n: Maximum n-gram size

    Returns: [(ngram_text, (start_idx, end_idx)), ...]
    """
    ngrams = []

    for n in range(1, min(max_n + 1, len(doc) + 1)):
        for i in range(len(doc) - n + 1):
            span = doc[i:i+n]
            # Use lemmatized text for better matching
            ngram_text = " ".join([t.lemma_ for t in span])
            ngrams.append((ngram_text, (span.start, span.end)))

    return ngrams


def get_wordnet_synonyms(word: str) -> Set[str]:
    """Get synonyms of a word using WordNet."""
    synonyms = {word}
    for synset in wordnet.synsets(word.replace(" ", "_")):
        for lemma in synset.lemmas():
            synonyms.add(lemma.name().replace("_", " ").lower())

    return synonyms


def get_wordnet_similarity(word1: str, word2: str) -> float:
    """
    Calculate semantic similarity using WordNet Wu-Palmer similarity.
    Returns score 0.0-1.0 (higher = more similar).

    "city" vs "town" -> 0.87
    "city" vs "salary" -> 0.15

    Better than binary synonym match - gives continuous ranking.
    """
    word1 = word1.replace(" ", "_")
    word2 = word2.replace(" ", "_")

    synsets1 = wordnet.synsets(word1)
    synsets2 = wordnet.synsets(word2)

    if not synsets1 or not synsets2:
        return 0.0

    # Find maximum similarity across all synset pairs
    max_sim = 0.0
    for s1 in synsets1:
        for s2 in synsets2:
            sim = s1.wup_similarity(s2)
            if sim and sim > max_sim:
                max_sim = sim

    return max_sim


def link_question_to_schema(
    question_analysis: QuestionAnalysis,
    schema: SchemaGraph,
    fuzzy_threshold: float = 75.0
) -> List[SchemaLink]:
    """
    Link question words to schema elements using multiple strategies.

    Args:
        question_analysis: Parsed question
        schema: Database schema
        fuzzy_threshold: Minimum fuzzy match score (0-100)

    Returns: List of schema links, sorted by score (descending)
    """
    links = []

    # Get all schema elements to match against
    all_tables = schema.get_all_tables()
    all_columns = schema.get_all_columns()  # [(table, column), ...]

    # Use noun chunks (better than n-grams - respects grammatical boundaries)
    # Example: "total number of students" is ONE chunk, not random n-grams
    phrases_to_match = []

    # Add noun chunks
    for chunk_text, start, end in question_analysis.noun_chunks:
        phrases_to_match.append((chunk_text, (start, end), "chunk"))

    # Also add individual content words (for cases where chunks miss things)
    for token in question_analysis.tokens:
        if not token.is_stop and token.pos not in ["PUNCT", "SYM", "SPACE"]:
            phrases_to_match.append((token.lemma, (token.idx, token.idx + 1), "token"))
            # Also try singular form for nouns
            if token.singular_form and token.singular_form != token.lemma:
                phrases_to_match.append((token.singular_form, (token.idx, token.idx + 1), "singular"))

    # Fallback: also generate n-grams for coverage using spaCy spans
    # Filter to content words only
    content_doc = question_analysis.doc  # Full spaCy doc
    ngrams = generate_ngrams_with_spans(content_doc, max_n=3)
    for ngram, span in ngrams:
        # Only add if not stopwords
        tokens_in_span = content_doc[span[0]:span[1]]
        if any(not t.is_stop for t in tokens_in_span):
            phrases_to_match.append((ngram, span, "ngram"))

    # ==========================
    # STRATEGY 1: EXACT MATCH
    # ==========================
    for phrase, (start, end), source in phrases_to_match:
        phrase_normalized = phrase.replace(" ", "_").lower()

        # Match tables
        for table in all_tables:
            if phrase_normalized == table.lower():
                links.append(SchemaLink(
                    question_phrase=phrase,
                    schema_element=table,
                    element_type="table",
                    table_name=table,
                    column_name=None,
                    score=100.0,
                    link_type=LinkType.EXACT
                ))

        # Match columns
        for table, column in all_columns:
            if phrase_normalized == column.lower():
                links.append(SchemaLink(
                    question_phrase=phrase,
                    schema_element=f"{table}.{column}",
                    element_type="column",
                    table_name=table,
                    column_name=column,
                    score=100.0,
                    link_type=LinkType.EXACT
                ))

    # ==========================
    # STRATEGY 2: FUZZY MATCH (rapidfuzz)
    # ==========================
    for phrase, (start, end), source in phrases_to_match:
        phrase_normalized = phrase.replace(" ", "").lower()

        # Fuzzy match tables
        for table in all_tables:
            table_normalized = table.replace("_", "").lower()
            score = fuzz.WRatio(phrase_normalized, table_normalized)

            # Boost score with WordNet similarity
            if score < 100.0:
                wn_sim = get_wordnet_similarity(phrase, table)
                if wn_sim > 0.5:  # Only boost if reasonably similar
                    score = max(score, 70.0 + wn_sim * 30.0)  # 70-100 range

            if score >= fuzzy_threshold:
                links.append(SchemaLink(
                    question_phrase=phrase,
                    schema_element=table,
                    element_type="table",
                    table_name=table,
                    column_name=None,
                    score=score,
                    link_type=LinkType.FUZZY
                ))

        # Fuzzy match columns
        for table, column in all_columns:
            column_normalized = column.replace("_", "").lower()
            score = fuzz.WRatio(phrase_normalized, column_normalized)

            # Boost score with WordNet similarity
            if score < 100.0:
                wn_sim = get_wordnet_similarity(phrase, column)
                if wn_sim > 0.5:
                    score = max(score, 70.0 + wn_sim * 30.0)

            if score >= fuzzy_threshold:
                links.append(SchemaLink(
                    question_phrase=phrase,
                    schema_element=f"{table}.{column}",
                    element_type="column",
                    table_name=table,
                    column_name=column,
                    score=score,
                    link_type=LinkType.FUZZY
                ))

    # ==========================
    # STRATEGY 3: SYNONYM MATCH (WordNet)
    # ==========================
    for phrase, (start, end), source in phrases_to_match:
        # Get synonyms of phrase
        phrase_syns = get_wordnet_synonyms(phrase)

        # Match tables
        for table in all_tables:
            table_words = table.replace("_", " ").split()
            for table_word in table_words:
                table_syns = get_wordnet_synonyms(table_word)

                # Check for overlap
                overlap = phrase_syns & table_syns
                if overlap and overlap != {phrase}:  # Don't count exact matches again
                    score = 85.0  # High but less than exact
                    links.append(SchemaLink(
                        question_phrase=phrase,
                        schema_element=table,
                        element_type="table",
                        table_name=table,
                        column_name=None,
                        score=score,
                        link_type=LinkType.SYNONYM
                    ))
                    break

        # Match columns
        for table, column in all_columns:
            column_words = column.replace("_", " ").split()
            for column_word in column_words:
                column_syns = get_wordnet_synonyms(column_word)

                overlap = phrase_syns & column_syns
                if overlap and overlap != {phrase}:
                    score = 85.0
                    links.append(SchemaLink(
                        question_phrase=phrase,
                        schema_element=f"{table}.{column}",
                        element_type="column",
                        table_name=table,
                        column_name=column,
                        score=score,
                        link_type=LinkType.SYNONYM
                    ))
                    break

    # ==========================
    # STRATEGY 4: PARTIAL WORD MATCH
    # ==========================
    # Match if question word is part of schema element or vice versa
    for phrase, (start, end), source in phrases_to_match:
        phrase_lower = phrase.replace(" ", "").lower()

        # Match columns (most useful for partial matches)
        for table, column in all_columns:
            column_lower = column.replace("_", "").lower()

            # Check containment in both directions
            if len(phrase_lower) >= 4:  # Only for longer words
                if phrase_lower in column_lower or column_lower in phrase_lower:
                    # Calculate score based on length ratio
                    len_ratio = min(len(phrase_lower), len(column_lower)) / max(len(phrase_lower), len(column_lower))
                    score = 70.0 * len_ratio

                    if score >= 50.0:  # Minimum threshold
                        links.append(SchemaLink(
                            question_phrase=phrase,
                            schema_element=f"{table}.{column}",
                            element_type="column",
                            table_name=table,
                            column_name=column,
                            score=score,
                            link_type=LinkType.PARTIAL
                        ))

    # ==========================
    # STRATEGY 5: VALUE DETECTION
    # ==========================
    # Detect values for WHERE clauses (not column references)

    # Quoted strings -> string values
    for quoted in question_analysis.quoted_values:
        links.append(SchemaLink(
            question_phrase=quoted,
            schema_element="<value>",
            element_type="value",
            table_name=None,
            column_name=None,
            score=100.0,
            link_type=LinkType.VALUE,
            is_value=True,
            value=quoted
        ))

    # Named entities -> potential values
    for entity in question_analysis.entities:
        # GPE, PERSON, ORG are likely values, not column names
        if entity.label in ["GPE", "PERSON", "ORG", "DATE", "TIME", "MONEY", "PERCENT"]:
            links.append(SchemaLink(
                question_phrase=entity.text,
                schema_element="<value>",
                element_type="value",
                table_name=None,
                column_name=None,
                score=90.0,
                link_type=LinkType.VALUE,
                is_value=True,
                value=entity.text
            ))

    # Numeric values
    for idx, num_val in question_analysis.numeric_values:
        token_text = question_analysis.tokens[idx].text
        links.append(SchemaLink(
            question_phrase=token_text,
            schema_element="<value>",
            element_type="value",
            table_name=None,
            column_name=None,
            score=100.0,
            link_type=LinkType.VALUE,
            is_value=True,
            value=str(num_val)
        ))

    # ==========================
    # DEDUPLICATE & RANK
    # ==========================
    # Remove duplicate links (same phrase + schema element)
    # Keep highest score
    seen = {}
    for link in links:
        key = (link.question_phrase, link.schema_element)
        if key not in seen or link.score > seen[key].score:
            seen[key] = link

    unique_links = list(seen.values())

    # Sort by score (descending)
    unique_links.sort(key=lambda x: x.score, reverse=True)

    return unique_links


def get_table_links(links: List[SchemaLink]) -> List[SchemaLink]:
    """Filter links to only tables."""
    return [l for l in links if l.element_type == "table"]


def get_column_links(links: List[SchemaLink]) -> List[SchemaLink]:
    """Filter links to only columns."""
    return [l for l in links if l.element_type == "column"]


def get_value_links(links: List[SchemaLink]) -> List[SchemaLink]:
    """Filter links to only values."""
    return [l for l in links if l.is_value]


def get_tables_from_links(links: List[SchemaLink]) -> Set[str]:
    """Get all unique tables mentioned in links."""
    tables = set()
    for link in links:
        if link.table_name:
            tables.add(link.table_name)
    return tables


def merge_schema_links(basic: List[SchemaLink], composite: List[SchemaLink]) -> List[SchemaLink]:
    """
    Merge two link sets, prioritizing composite scores over basic scores.

    Args:
        basic: Links from basic schema linking (exact/fuzzy/synonym matching)
        composite: Links from COMA++ composite matching (weighted aggregation)

    Returns:
        Merged list of unique links with highest scores
    """
    merged = {}

    # Add basic links first - keep highest scoring link per (table, column) key
    for link in basic:
        # Key by (table, column) or just table for table links
        key = (link.table_name, link.column_name) if link.column_name else link.table_name
        if key not in merged or link.score > merged[key].score:
            merged[key] = link

    # Override with composite (higher scores from composite matching)
    for link in composite:
        key = (link.table_name, link.column_name) if link.column_name else link.table_name
        if key not in merged or link.score > merged[key].score:
            merged[key] = link

    # Convert back to list and sort by score
    result = list(merged.values())
    result.sort(key=lambda x: x.score, reverse=True)

    return result


if __name__ == "__main__":
    # Test
    from .schema_graph import SchemaGraph
    from .question_analysis import analyze_question

    # Sample schema
    sample_db = {
        "db_id": "test_db",
        "table_names_original": ["Student", "Course", "Enrollment"],
        "column_names_original": [
            [-1, "*"],
            [0, "student_id"],
            [0, "stu_name"],
            [0, "age"],
            [1, "course_id"],
            [1, "title"],
            [2, "enrollment_id"],
            [2, "student_id"],
            [2, "course_id"]
        ],
        "column_types": ["text", "number", "text", "number", "number", "text", "number", "number", "number"],
        "primary_keys": [1, 4, 6],
        "foreign_keys": [[7, 1], [8, 4]]
    }

    schema = SchemaGraph.from_spider_json(sample_db)

    # Test questions
    test_questions = [
        "What is the name of students?",
        "How many courses are there?",
        "List students enrolled in 'Database Systems'",
    ]

    for q in test_questions:
        print(f"\nQuestion: {q}")
        analysis = analyze_question(q)
        links = link_question_to_schema(analysis, schema)

        print("Top links:")
        for link in links[:5]:
            print(f"  '{link.question_phrase}' -> {link.schema_element} "
                  f"({link.link_type.value}, score={link.score:.1f})")
