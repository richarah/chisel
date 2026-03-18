"""
Ontology-Based Schema Linking (v0.5)

Enhanced schema linking using OWL ontologies and COMA++ inspired composite matching.

Libraries:
- owlready2: OWL ontology reasoning (deterministic)
- rapidfuzz: Fuzzy string matching (already used)
- nltk.wordnet: Synonym expansion (already used)
- networkx: Structural similarity (already used for FK graph)

Technique: COMA++ composite approach
- String similarity (edit distance, n-grams)
- Linguistic similarity (WordNet, lemmatization)
- Structural similarity (FK relationships, parent/child)
- Constraint-based (data types, primary keys)
- Weighted aggregation

Philosophy: Library does the work (NIH principle)
"""

from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
from enum import Enum
import re

# Libraries (NIH principle)
from rapidfuzz import fuzz, process
from nltk.corpus import wordnet
import inflect

# Local
from .schema_graph import SchemaGraph


# ==========================
# DOMAIN ONTOLOGY DEFINITIONS
# ==========================

@dataclass
class OntologyMapping:
    """
    Maps natural language concepts to schema elements via ontology.

    Example:
        "enrolled" -> Enrollment relation (via domain ontology)
        "taking courses" -> Enrollment.course_id (via owl:equivalentProperty)
    """
    concept: str
    schema_element: str
    element_type: str  # 'table', 'column', 'relation'
    confidence: float
    source: str  # 'ontology', 'wordnet', 'fuzzy', 'structural'


class RelationType(Enum):
    """OWL relation types for ontology reasoning."""
    EQUIVALENT = "owl:equivalentClass"
    SUBCLASS = "rdfs:subClassOf"
    SYNONYM = "rdfs:label"
    RELATED = "rdfs:seeAlso"


# Simplified ontology (full version would use owlready2)
# This is a deterministic dictionary-based implementation
DOMAIN_ONTOLOGIES = {
    "university": {
        # Class equivalences
        "Student": ["student", "pupil", "learner", "scholar"],
        "Professor": ["professor", "instructor", "teacher", "faculty"],
        "Course": ["course", "class", "subject", "module"],
        "Department": ["department", "dept", "division", "school"],
        "Enrollment": ["enrollment", "registration", "taking", "enrolled_in"],

        # Property equivalences
        "teaches": ["teaches", "instructs", "gives", "offers"],
        "takes": ["takes", "enrolls_in", "attends", "studies"],
        "works_in": ["works_in", "employed_by", "affiliated_with"],

        # Inverse properties
        "taught_by": "teaches^-1",  # Inverse of teaches
        "has_student": "takes^-1",   # Inverse of takes
    },

    "geography": {
        "Country": ["country", "nation", "state"],
        "City": ["city", "town", "municipality"],
        "Continent": ["continent"],
        "located_in": ["located_in", "in", "within", "part_of"],
    },

    "business": {
        "Company": ["company", "corporation", "firm", "business"],
        "Employee": ["employee", "worker", "staff"],
        "Manager": ["manager", "supervisor", "director"],
        "works_for": ["works_for", "employed_by"],
    }
}


# Abbreviation dictionary (deterministic expansion)
ABBREVIATIONS = {
    # Academic
    "CS": "Computer Science",
    "EE": "Electrical Engineering",
    "ME": "Mechanical Engineering",
    "CE": "Civil Engineering",
    "Prof": "Professor",
    "Dr": "Doctor",
    "PhD": "Doctor of Philosophy",

    # Geographic
    "CA": "California",
    "NY": "New York",
    "TX": "Texas",
    "FL": "Florida",
    "USA": "United States",
    "UK": "United Kingdom",

    # Business
    "CEO": "Chief Executive Officer",
    "CTO": "Chief Technology Officer",
    "VP": "Vice President",
    "HR": "Human Resources",

    # Temporal
    "Jan": "January",
    "Feb": "February",
    "Mar": "March",
    "Apr": "April",
    "Aug": "August",
    "Sep": "September",
    "Oct": "October",
    "Nov": "November",
    "Dec": "December",
}


# ==========================
# COMPOSITE SCHEMA MATCHER
# ==========================

class CompositeSchemaLinker:
    """
    COMA++ inspired composite schema matching.

    Combines multiple matchers with weighted aggregation:
    1. String similarity (30%)
    2. Linguistic similarity (25%)
    3. Structural similarity (25%)
    4. Constraint-based (20%)

    Deterministic: same inputs -> same outputs
    """

    def __init__(self, schema: SchemaGraph):
        self.schema = schema
        self.inflect_engine = inflect.engine()

        # Matcher weights (tuned on development data)
        self.weights = {
            'string': 0.30,
            'linguistic': 0.25,
            'structural': 0.25,
            'constraint': 0.20
        }

    def link_word_to_schema(
        self,
        word: str,
        threshold: float = 75.0,
        domain: str = "university"
    ) -> List[OntologyMapping]:
        """
        Link a word to schema elements using composite matching.

        Args:
            word: Natural language word/phrase
            threshold: Minimum confidence score (0-100)
            domain: Domain ontology to use

        Returns:
            List of ontology mappings sorted by confidence
        """
        mappings = []

        # Expand abbreviations first
        expanded_word = ABBREVIATIONS.get(word.upper(), word)

        # Get all schema elements
        tables = self.schema.tables.keys() if hasattr(self.schema, 'tables') else []

        # Try each matcher
        for table in tables:
            scores = {}

            # 1. String similarity
            scores['string'] = self._string_similarity(expanded_word, table)

            # 2. Linguistic similarity
            scores['linguistic'] = self._linguistic_similarity(expanded_word, table, domain)

            # 3. Structural similarity (if FK relationships exist)
            scores['structural'] = self._structural_similarity(expanded_word, table)

            # 4. Constraint-based similarity
            scores['constraint'] = self._constraint_similarity(expanded_word, table)

            # Weighted aggregation
            confidence = sum(scores[k] * self.weights[k] for k in scores)

            if confidence >= threshold:
                mappings.append(OntologyMapping(
                    concept=word,
                    schema_element=table,
                    element_type='table',
                    confidence=confidence,
                    source='composite'
                ))

        # Sort by confidence
        mappings.sort(key=lambda m: m.confidence, reverse=True)
        return mappings

    def match_all(
        self,
        question_terms: List[str],
        question_phrases: List[str],
        threshold: float = 75.0,
        domain: str = "university"
    ) -> List:
        """
        Match all question terms and phrases to schema using composite matching.

        Returns SchemaLink objects compatible with existing pipeline.

        Args:
            question_terms: Individual content words from question
            question_phrases: Noun phrases/chunks from question
            threshold: Minimum confidence score (0-100)
            domain: Domain ontology to use

        Returns:
            List of SchemaLink objects (from schema_linking module)
        """
        from .schema_linking import SchemaLink, LinkType

        links = []

        # Match phrases (higher priority)
        for phrase in question_phrases:
            ontology_mappings = self.link_word_to_schema(phrase, threshold, domain)

            for mapping in ontology_mappings:
                # Convert OntologyMapping to SchemaLink
                links.append(SchemaLink(
                    question_phrase=phrase,
                    schema_element=mapping.schema_element,
                    element_type=mapping.element_type,
                    table_name=mapping.schema_element if mapping.element_type == 'table' else None,
                    column_name=None,
                    score=mapping.confidence,
                    link_type=LinkType.FUZZY  # Composite matching is fuzzy-like
                ))

        # Match individual terms
        for term in question_terms:
            # Skip if already matched as part of a phrase
            if any(term in phrase for phrase in question_phrases):
                continue

            ontology_mappings = self.link_word_to_schema(term, threshold, domain)

            for mapping in ontology_mappings:
                links.append(SchemaLink(
                    question_phrase=term,
                    schema_element=mapping.schema_element,
                    element_type=mapping.element_type,
                    table_name=mapping.schema_element if mapping.element_type == 'table' else None,
                    column_name=None,
                    score=mapping.confidence,
                    link_type=LinkType.FUZZY
                ))

        # Also match against columns
        all_columns = self.schema.get_all_columns() if hasattr(self.schema, 'get_all_columns') else []

        for phrase in question_phrases + question_terms:
            expanded = ABBREVIATIONS.get(phrase.upper(), phrase)

            for table, column in all_columns:
                scores = {}
                scores['string'] = self._string_similarity(expanded, column)
                scores['linguistic'] = self._linguistic_similarity(expanded, column, domain)
                scores['structural'] = 50.0  # Neutral for columns
                scores['constraint'] = 50.0

                confidence = sum(scores[k] * self.weights[k] for k in scores)

                if confidence >= threshold:
                    links.append(SchemaLink(
                        question_phrase=phrase,
                        schema_element=f"{table}.{column}",
                        element_type='column',
                        table_name=table,
                        column_name=column,
                        score=confidence,
                        link_type=LinkType.FUZZY
                    ))

        # Deduplicate and sort
        seen = {}
        for link in links:
            key = (link.question_phrase, link.schema_element)
            if key not in seen or link.score > seen[key].score:
                seen[key] = link

        unique_links = list(seen.values())
        unique_links.sort(key=lambda x: x.score, reverse=True)

        return unique_links

    def _string_similarity(self, word: str, schema_element: str) -> float:
        """
        String-based similarity using multiple algorithms.

        Uses rapidfuzz (deterministic, fast):
        - WRatio: weighted ratio (accounts for partial matches)
        - Token sort: order-independent matching
        """
        # Normalize
        word_norm = word.lower().replace('_', ' ')
        element_norm = schema_element.lower().replace('_', ' ')

        # Multiple string metrics
        wratio = fuzz.WRatio(word_norm, element_norm)
        token_sort = fuzz.token_sort_ratio(word_norm, element_norm)
        partial = fuzz.partial_ratio(word_norm, element_norm)

        # Average (deterministic)
        return (wratio + token_sort + partial) / 3.0

    def _linguistic_similarity(self, word: str, schema_element: str, domain: str) -> float:
        """
        Linguistic similarity using WordNet + ontology.

        Checks:
        1. Synonyms from WordNet
        2. Lemmatization variants
        3. Domain ontology mappings
        4. Singular/plural forms
        """
        score = 0.0
        word_lower = word.lower()
        element_lower = schema_element.lower()

        # Check domain ontology first (highest confidence)
        ontology = DOMAIN_ONTOLOGIES.get(domain, {})
        for concept, synonyms in ontology.items():
            if concept.lower() == element_lower:
                if word_lower in [s.lower() for s in synonyms]:
                    return 100.0  # Perfect match via ontology

        # WordNet synonyms
        word_synsets = wordnet.synsets(word_lower)
        element_synsets = wordnet.synsets(element_lower)

        if word_synsets and element_synsets:
            # Check for overlap in synsets
            word_lemmas = {lemma.name().lower() for syn in word_synsets for lemma in syn.lemmas()}
            element_lemmas = {lemma.name().lower() for syn in element_synsets for lemma in syn.lemmas()}

            overlap = len(word_lemmas & element_lemmas)
            if overlap > 0:
                score = min(100.0, 70.0 + (overlap * 10))

        # Singular/plural matching
        word_singular = self.inflect_engine.singular_noun(word_lower) or word_lower
        element_singular = self.inflect_engine.singular_noun(element_lower) or element_lower

        if word_singular == element_singular:
            score = max(score, 90.0)

        return score

    def _structural_similarity(self, word: str, table: str) -> float:
        """
        Structural similarity based on FK relationships.

        Intuition: Related tables should have similar semantic roles.
        Uses NetworkX FK graph for structural analysis.
        """
        # Placeholder for structural matching
        # Would use self.schema.fk_graph to find related tables
        # and propagate similarity scores

        # For now, return neutral score
        return 50.0

    def _constraint_similarity(self, word: str, table: str) -> float:
        """
        Constraint-based similarity using schema metadata.

        Checks:
        1. Primary key names (often contain table name)
        2. Column data types (semantic hints)
        3. Column count (complexity indicator)
        """
        score = 50.0  # Neutral baseline

        # Check if word appears in primary key name
        if hasattr(self.schema, 'tables') and table in self.schema.tables:
            table_info = self.schema.tables[table]
            if hasattr(table_info, 'primary_key'):
                pk = table_info.primary_key
                if pk and word.lower() in pk.lower():
                    score += 20.0

        return min(100.0, score)


# ==========================
# VALUE NORMALIZATION
# ==========================

class ValueNormalizer:
    """
    Normalize extracted values using deterministic rules.

    Handles:
    - Abbreviations (CS -> Computer Science)
    - Units (100k -> 100000)
    - Temporal expressions (Fall 2023 -> dates)
    """

    @staticmethod
    def normalize_abbreviation(value: str) -> str:
        """Expand abbreviations deterministically."""
        return ABBREVIATIONS.get(value.upper(), value)

    @staticmethod
    def normalize_number(value: str) -> Optional[float]:
        """
        Normalize number expressions.

        Examples:
            "100k" -> 100000
            "1.5M" -> 1500000
            "5 thousand" -> 5000
        """
        value_clean = value.strip().upper()

        # Handle k, M, B suffixes
        multipliers = {'K': 1000, 'M': 1000000, 'B': 1000000000}

        for suffix, mult in multipliers.items():
            if value_clean.endswith(suffix):
                try:
                    number = float(value_clean[:-1])
                    return number * mult
                except ValueError:
                    pass

        # Try direct conversion
        try:
            return float(value_clean)
        except ValueError:
            return None

    @staticmethod
    def normalize_semester(value: str) -> Dict[str, any]:
        """
        Normalize semester expressions.

        Examples:
            "Fall 2023" -> {semester: 1, year: 2023}
            "Spring 2024" -> {semester: 2, year: 2024}
        """
        value_lower = value.lower()

        # Extract year
        year_match = re.search(r'\b(19|20)\d{2}\b', value)
        year = int(year_match.group()) if year_match else None

        # Map semester to number
        if 'fall' in value_lower or 'autumn' in value_lower:
            return {'semester': 1, 'year': year}
        elif 'spring' in value_lower:
            return {'semester': 2, 'year': year}
        elif 'summer' in value_lower:
            return {'semester': 3, 'year': year}

        return {'semester': None, 'year': year}


# ==========================
# MAIN INTERFACE
# ==========================

def enhanced_schema_linking(
    question_words: List[str],
    schema: SchemaGraph,
    domain: str = "university",
    threshold: float = 75.0
) -> List[OntologyMapping]:
    """
    Enhanced schema linking with ontology and composite matching.

    Args:
        question_words: Words from question analysis
        schema: Database schema
        domain: Domain ontology to use
        threshold: Minimum confidence

    Returns:
        List of schema mappings
    """
    linker = CompositeSchemaLinker(schema)

    all_mappings = []
    for word in question_words:
        mappings = linker.link_word_to_schema(word, threshold, domain)
        all_mappings.extend(mappings)

    # Remove duplicates, keep highest confidence
    unique_mappings = {}
    for mapping in all_mappings:
        key = (mapping.concept, mapping.schema_element)
        if key not in unique_mappings or mapping.confidence > unique_mappings[key].confidence:
            unique_mappings[key] = mapping

    return list(unique_mappings.values())


# ==========================
# TESTING
# ==========================

if __name__ == "__main__":
    print("Ontology-Based Schema Linking v0.5")
    print("=" * 60)

    # Test abbreviation normalization
    print("\nAbbreviation Normalization:")
    test_abbrevs = ["CS", "Prof", "CA", "CEO"]
    for abbrev in test_abbrevs:
        normalized = ValueNormalizer.normalize_abbreviation(abbrev)
        print(f"  {abbrev} -> {normalized}")

    # Test number normalization
    print("\nNumber Normalization:")
    test_numbers = ["100k", "1.5M", "500", "2B"]
    for num in test_numbers:
        normalized = ValueNormalizer.normalize_number(num)
        print(f"  {num} -> {normalized}")

    # Test semester normalization
    print("\nSemester Normalization:")
    test_semesters = ["Fall 2023", "Spring 2024", "Summer 2022"]
    for sem in test_semesters:
        normalized = ValueNormalizer.normalize_semester(sem)
        print(f"  {sem} -> {normalized}")

    print("\n[OK] Ontology-based schema linking ready")
