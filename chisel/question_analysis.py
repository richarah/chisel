"""
Task 2: Question Analysis

Parse questions using spaCy and detect SQL signals using WordNet expansion.

Libraries doing the heavy lifting:
- spaCy: tokenization, POS tagging, dependency parsing, NER, lemmatization
- nltk.wordnet: synonym expansion for SQL indicators (155k words -> 3-5x coverage)
- nltk.corpus.stopwords: filter noise words

We write: seed word lists + expansion logic + signal detection rules
"""

from dataclasses import dataclass, field
from typing import List, Set, Tuple, Dict, Optional
import re

# Import required libraries (NO graceful degradation - these are REQUIRED)
import spacy
from spacy.tokens import Doc
from nltk.corpus import wordnet, stopwords
import nltk
import lemminflect  # Better lemmatization
from word2number import w2n  # Number word parsing
import dateparser  # Date/time parsing
import inflect  # Pluralization/singularization

# Initialize inflect engine
INFLECT_ENGINE = inflect.engine()


# ==========================
# SQL INDICATOR EXPANSION
# ==========================
# Start with seed words, expand via WordNet synsets
# This gives us 3-5x more coverage for free

SQL_INDICATOR_SEEDS = {
    "COUNT": ["count", "number", "total", "amount", "quantity", "how many"],
    "MAX": ["most", "biggest", "largest", "highest", "greatest", "maximum", "top"],
    "MIN": ["least", "smallest", "lowest", "fewest", "minimum", "bottom"],
    "AVG": ["average", "mean", "typical"],
    "SUM": ["sum", "total", "combined", "overall", "aggregate"],
    "ORDER_DESC": ["descending", "decreasing", "highest", "most", "largest", "biggest", "top"],
    "ORDER_ASC": ["ascending", "increasing", "lowest", "least", "smallest", "bottom"],
    "WHERE_GT": ["more than", "greater than", "above", "over", "exceeds", "higher than"],
    "WHERE_LT": ["less than", "below", "under", "fewer than", "lower than"],
    "WHERE_EQ": ["equal", "equals", "is", "same"],
    "WHERE_NEQ": ["not", "other than", "except", "excluding", "different"],
    "WHERE_LIKE": ["contains", "includes", "like", "has", "with"],
    "GROUP_BY": ["each", "every", "per", "for each", "by"],
    "DISTINCT": ["different", "distinct", "unique", "separate"],
    "INTERSECT": ["both", "in common", "shared", "intersection"],
    "EXCEPT": ["but not", "excluding", "except", "without"],
    "UNION": ["or", "either", "combined"],
    "EXISTS": ["any", "at least one", "some", "exists"],
    "NOT_EXISTS": ["no", "none", "not any", "does not exist"],
    "LIMIT": ["top", "first", "bottom", "last"],
    # Quantifiers - map to SQL constructs
    "QUANTIFIER_ALL": ["all", "every"],  # NOT EXISTS ... NOT
    "QUANTIFIER_ANY": ["any", "some"],   # EXISTS
    "QUANTIFIER_NO": ["no", "none"],     # NOT EXISTS
    "QUANTIFIER_ATLEAST": ["at least"],  # >= N (HAVING)
    "QUANTIFIER_ATMOST": ["at most"],    # <= N (HAVING)
    "QUANTIFIER_EXACTLY": ["exactly"],   # = N (HAVING)
}


def expand_with_wordnet(seeds: List[str]) -> Set[str]:
    """
    Expand seed words using WordNet synonyms.

    WordNet gives us 117k synsets, 155k words. This is free coverage.
    Example: "biggest" -> synset.lemmas() -> ["largest", "greatest", "biggest"]
    """
    expanded = set(seeds)

    for seed in seeds:
        # Get synsets for this word
        synsets = wordnet.synsets(seed.replace(" ", "_"))

        for synset in synsets:
            # Add all lemma names from this synset
            for lemma in synset.lemmas():
                lemma_name = lemma.name().replace("_", " ").lower()
                expanded.add(lemma_name)

    return expanded


# Build expanded SQL indicators at module load time (deterministic)
# NLTK data MUST be pre-installed via setup (no auto-download)
SQL_INDICATORS: Dict[str, Set[str]] = {}
for signal, seeds in SQL_INDICATOR_SEEDS.items():
    SQL_INDICATORS[signal] = expand_with_wordnet(seeds)

# Load stopwords (MUST be pre-installed)
STOPWORDS = set(stopwords.words('english'))


# ==========================
# QUESTION ANALYSIS
# ==========================

def get_better_lemma(token) -> str:
    """
    Get improved lemma using lemminflect (95.6% accuracy vs spaCy's 84.7%).
    lemminflect integrates with spaCy via token._.lemma()
    """
    if hasattr(token, '_'):
        lemmas = token._.lemma()
        if lemmas:
            return lemmas.lower()
    return token.lemma_.lower()


def parse_number_word(text: str) -> Optional[float]:
    """
    Parse number words to numeric values.
    "twenty" -> 20, "three hundred" -> 300
    """
    try:
        return float(w2n.word_to_num(text))
    except ValueError:
        return None


def parse_date_expression(text: str) -> Optional[str]:
    """
    Parse date expressions to normalized form.
    "last year" -> "2025", "March 2020" -> "2020-03-01"
    """
    parsed = dateparser.parse(text)
    if parsed:
        return parsed.isoformat()
    return None


def get_singular_form(word: str) -> str:
    """
    Get singular form of a noun.
    "countries" -> "country", "students" -> "student"
    Better than lemmatization for schema linking.
    """
    singular = INFLECT_ENGINE.singular_noun(word)
    return singular if singular else word


def parse_ordinal(text: str) -> Optional[int]:
    """
    Parse ordinal words to numeric values.
    "first" -> 1, "third" -> 3, "tenth" -> 10
    Used for queries like "the third highest salary" -> LIMIT 1 OFFSET 2
    """
    try:
        from num2words import num2words
        # num2words can convert numbers to ordinals, but we need the reverse
        # Try a simple mapping first (most common ordinals)
        ordinal_map = {
            "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
            "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
            "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
            "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
            "nineteenth": 19, "twentieth": 20,
        }

        text_lower = text.lower().strip()

        # Check direct mapping
        if text_lower in ordinal_map:
            return ordinal_map[text_lower]

        # Handle patterns like "1st", "2nd", "3rd", "21st" etc.
        import re
        match = re.match(r'(\d+)(?:st|nd|rd|th)', text_lower)
        if match:
            return int(match.group(1))

        return None
    except (ImportError, ValueError):
        return None


@dataclass
class Token:
    """Parsed token information."""
    text: str
    lemma: str
    pos: str  # Part of speech
    tag: str  # Fine-grained tag
    dep: str  # Dependency relation
    head_idx: int  # Index of head token
    idx: int  # Position in sentence
    is_stop: bool = False
    singular_form: Optional[str] = None  # For nouns
    numeric_value: Optional[float] = None  # Parsed number (including word numbers)


@dataclass
class Entity:
    """Named entity."""
    text: str
    label: str  # PERSON, ORG, GPE, DATE, etc.
    start: int
    end: int


@dataclass
class QuestionAnalysis:
    """
    Parsed question with SQL signals detected.

    spaCy does: tokenization, POS, dependencies, NER
    We do: map parsed info -> SQL signals using expanded indicators
    """
    original_text: str
    tokens: List[Token] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    sql_signals: Set[str] = field(default_factory=set)
    content_words: List[str] = field(default_factory=list)  # Non-stopword lemmas
    question_type: str = "select"  # "count", "select", "yes_no", "superlative"
    superlatives: List[Tuple[int, str]] = field(default_factory=list)  # [(idx, word)]
    comparatives: List[Tuple[int, str]] = field(default_factory=list)
    negations: List[int] = field(default_factory=list)  # Token indices with negation
    conjunctions: List[Tuple[int, str]] = field(default_factory=list)  # [(idx, "and"/"or")]
    quoted_values: List[str] = field(default_factory=list)
    numeric_values: List[Tuple[int, float]] = field(default_factory=list)  # [(idx, value)]
    date_values: List[Tuple[int, str]] = field(default_factory=list)  # [(idx, iso_date)]
    ordinal_values: List[Tuple[int, int]] = field(default_factory=list)  # [(idx, ordinal_num)]
    noun_chunks: List[Tuple[str, int, int]] = field(default_factory=list)  # [(text, start, end)]

    @classmethod
    def from_question(cls, question: str, nlp=None) -> 'QuestionAnalysis':
        """
        Parse question using spaCy.

        Args:
            question: Question text
            nlp: spaCy model (if None, will load en_core_web_sm)
        """
        analysis = cls(original_text=question)

        # Load spaCy model if needed
        if nlp is None:
            nlp = spacy.load("en_core_web_sm")

        # Parse with spaCy (does all the heavy lifting)
        doc = nlp(question)

        # Extract tokens (spaCy does tokenization, lemmatization, POS, dependencies)
        for idx, token in enumerate(doc):
            # Use better lemmatization
            lemma = get_better_lemma(token)

            # Get singular form for nouns (better for schema linking)
            singular = None
            if token.pos_ == "NOUN":
                singular = get_singular_form(token.text.lower())

            # Parse number words ("twenty" -> 20)
            num_value = None
            if token.like_num:
                try:
                    num_value = float(token.text.replace(",", ""))
                except ValueError:
                    pass
            else:
                # Try to parse as number word
                num_value = parse_number_word(token.text.lower())

            t = Token(
                text=token.text,
                lemma=lemma,
                pos=token.pos_,
                tag=token.tag_,
                dep=token.dep_,
                head_idx=token.head.i,
                idx=idx,
                is_stop=token.is_stop,
                singular_form=singular,
                numeric_value=num_value
            )
            analysis.tokens.append(t)

            # Collect content words (non-stopwords)
            if not token.is_stop and token.pos_ not in ["PUNCT", "SYM", "SPACE"]:
                analysis.content_words.append(lemma)

            # Detect superlatives (spaCy POS tagger does this)
            if token.tag_ in ["JJS", "RBS"]:  # Superlative adjective/adverb
                analysis.superlatives.append((idx, token.text.lower()))

            # Detect comparatives
            if token.tag_ in ["JJR", "RBR"]:  # Comparative adjective/adverb
                analysis.comparatives.append((idx, token.text.lower()))

            # Detect negations (spaCy dependency parser does this)
            if token.dep_ == "neg":
                analysis.negations.append(idx)

            # Detect conjunctions
            if token.pos_ == "CCONJ":
                analysis.conjunctions.append((idx, token.text.lower()))

            # Collect numeric values (including word numbers)
            if num_value is not None:
                analysis.numeric_values.append((idx, num_value))

            # Parse ordinal values ("first" -> 1, "third" -> 3)
            ordinal_value = parse_ordinal(token.text)
            if ordinal_value is not None:
                analysis.ordinal_values.append((idx, ordinal_value))

        # Extract entities (spaCy NER does this)
        for ent in doc.ents:
            analysis.entities.append(Entity(
                text=ent.text,
                label=ent.label_,
                start=ent.start,
                end=ent.end
            ))

            # Parse DATE entities using dateparser
            if ent.label_ == "DATE":
                date_iso = parse_date_expression(ent.text)
                if date_iso:
                    analysis.date_values.append((ent.start, date_iso))

        # Extract noun chunks (spaCy does this - better than n-grams)
        for chunk in doc.noun_chunks:
            analysis.noun_chunks.append((
                chunk.text.lower(),
                chunk.start,
                chunk.end
            ))

        # Extract quoted strings (regex)
        quoted = re.findall(r'"([^"]*)"', question)
        quoted += re.findall(r"'([^']*)'", question)
        analysis.quoted_values = quoted

        # Detect SQL signals using expanded indicators
        question_lower = question.lower()
        question_lemmas = set(t.lemma for t in analysis.tokens)

        for signal, indicators in SQL_INDICATORS.items():
            for indicator in indicators:
                # Check multi-word phrases in original text
                if " " in indicator and indicator in question_lower:
                    analysis.sql_signals.add(signal)
                    break
                # Check single words in lemmas
                elif " " not in indicator and indicator in question_lemmas:
                    analysis.sql_signals.add(signal)
                    break

        # Determine question type
        first_word = analysis.tokens[0].lemma if analysis.tokens else ""

        if first_word in ["how"] and len(analysis.tokens) > 1:
            if analysis.tokens[1].lemma in ["many", "much"]:
                analysis.question_type = "count"
                analysis.sql_signals.add("COUNT")

        if first_word in ["is", "are", "do", "does", "did", "has", "have", "can", "will"]:
            analysis.question_type = "yes_no"

        if analysis.superlatives:
            analysis.question_type = "superlative"
            # Superlatives usually mean ORDER BY + LIMIT
            analysis.sql_signals.add("ORDER_DESC" if "most" in question_lower or "biggest" in question_lower else "ORDER_ASC")
            analysis.sql_signals.add("LIMIT")

        return analysis

    def has_aggregation(self) -> bool:
        """Check if question requires aggregation."""
        agg_signals = {"COUNT", "MAX", "MIN", "AVG", "SUM"}
        return bool(agg_signals & self.sql_signals)

    def has_comparison(self) -> bool:
        """Check if question has comparison operators."""
        comp_signals = {"WHERE_GT", "WHERE_LT", "WHERE_EQ", "WHERE_NEQ"}
        return bool(comp_signals & self.sql_signals)


def analyze_question(question: str, nlp=None) -> QuestionAnalysis:
    """Convenience function to analyze a question."""
    return QuestionAnalysis.from_question(question, nlp)


if __name__ == "__main__":
    # Test
    print("SQL Indicator expansion example:")
    print("MAX indicators:", sorted(SQL_INDICATORS["MAX"])[:20])
    print()

    test_questions = [
        "How many students are there?",
        "What is the name of the student with the highest GPA?",
        "List all courses that have more than 100 students.",
        "Which department has the most professors?",
    ]

    print("Testing question analysis:")
    for q in test_questions:
        print(f"\nQuestion: {q}")
        analysis = analyze_question(q)
        print(f"  Type: {analysis.question_type}")
        print(f"  Signals: {analysis.sql_signals}")
        print(f"  Content words: {analysis.content_words}")
        print(f"  Entities: {[(e.text, e.label) for e in analysis.entities]}")
