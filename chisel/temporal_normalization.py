"""
Temporal Normalization (v0.1)

Deterministic date/time expression normalization using TIMEX3 standard.

Libraries:
- dateparser: Parse natural language dates (deterministic rules)
- datetime: Python standard library
- re: Pattern matching for temporal expressions

Standards:
- TIMEX3: ISO standard for temporal annotation
- ISO 8601: Date/time format standard

Philosophy: Dates are structured data, not fuzzy concepts
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from enum import Enum
import re

# Library for date parsing (deterministic rule-based)
try:
    import dateparser
    DATEPARSER_AVAILABLE = True
except ImportError:
    DATEPARSER_AVAILABLE = False
    print("[WARN] dateparser not installed. Install: pip install dateparser")


# ==========================
# TEMPORAL ENTITY TYPES
# ==========================

class TemporalType(Enum):
    """TIMEX3 temporal types."""
    DATE = "DATE"  # Specific date (2023-09-15)
    TIME = "TIME"  # Specific time (14:30:00)
    DURATION = "DURATION"  # Time span (3 years, 2 weeks)
    SET = "SET"  # Recurring times (every Monday, annually)


class Granularity(Enum):
    """Temporal granularity levels."""
    YEAR = "year"
    MONTH = "month"
    WEEK = "week"
    DAY = "day"
    HOUR = "hour"
    MINUTE = "minute"
    SECOND = "second"


@dataclass
class TemporalExpression:
    """
    Normalized temporal expression following TIMEX3 standard.

    Examples:
        "Fall 2023" -> TemporalExpression(
            original_text="Fall 2023",
            normalized_value="2023-09",
            temporal_type=TemporalType.DATE,
            granularity=Granularity.MONTH
        )

        "last year" -> TemporalExpression(
            original_text="last year",
            normalized_value="2023",  # if current year is 2024
            temporal_type=TemporalType.DATE,
            granularity=Granularity.YEAR
        )
    """
    original_text: str
    normalized_value: str  # ISO 8601 format
    temporal_type: TemporalType
    granularity: Granularity
    confidence: float
    metadata: Dict[str, Any]


# ==========================
# SEMESTER/QUARTER MAPPING
# ==========================

# Academic semester mapping (deterministic)
SEMESTER_MAPPING = {
    "spring": {"month": 1, "semester_num": 2},
    "summer": {"month": 5, "semester_num": 3},
    "fall": {"month": 9, "semester_num": 1},
    "autumn": {"month": 9, "semester_num": 1},
    "winter": {"month": 1, "semester_num": 2},  # Winter = Spring in most systems
}

# Business quarter mapping
QUARTER_MAPPING = {
    "q1": {"month": 1, "quarter": 1},
    "q2": {"month": 4, "quarter": 2},
    "q3": {"month": 7, "quarter": 3},
    "q4": {"month": 10, "quarter": 4},
    "first quarter": {"month": 1, "quarter": 1},
    "second quarter": {"month": 4, "quarter": 2},
    "third quarter": {"month": 7, "quarter": 3},
    "fourth quarter": {"month": 10, "quarter": 4},
}

# Month abbreviations (already in standard library, but explicit for clarity)
MONTH_NAMES = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


# ==========================
# TEMPORAL NORMALIZER
# ==========================

class TemporalNormalizer:
    """
    Normalize temporal expressions to ISO 8601 format.

    All normalization is deterministic and rule-based.
    """

    def __init__(self, reference_date: Optional[datetime] = None):
        """
        Initialize temporal normalizer.

        Args:
            reference_date: Reference date for relative expressions (default: now)
        """
        self.reference_date = reference_date or datetime.now()

        # Configure dateparser for deterministic behavior
        self.dateparser_settings = {
            'PREFER_DATES_FROM': 'past',  # Deterministic preference
            'RELATIVE_BASE': self.reference_date,
            'STRICT_PARSING': True,
            'RETURN_AS_TIMEZONE_AWARE': False,
        }

    def normalize(self, text: str) -> List[TemporalExpression]:
        """
        Normalize all temporal expressions in text.

        Args:
            text: Text containing temporal expressions

        Returns:
            List of normalized temporal expressions
        """
        expressions = []

        # Try each normalization strategy
        expressions.extend(self._normalize_semester(text))
        expressions.extend(self._normalize_quarter(text))
        expressions.extend(self._normalize_year(text))
        expressions.extend(self._normalize_month_year(text))
        expressions.extend(self._normalize_relative_date(text))
        expressions.extend(self._normalize_duration(text))

        # If no specific patterns matched, try dateparser
        if not expressions and DATEPARSER_AVAILABLE:
            expr = self._normalize_with_dateparser(text)
            if expr:
                expressions.append(expr)

        return expressions

    def _normalize_semester(self, text: str) -> List[TemporalExpression]:
        """
        Normalize academic semester expressions.

        Examples:
            "Fall 2023" -> "2023-09"
            "Spring 2024" -> "2024-01"
        """
        expressions = []
        text_lower = text.lower()

        # Pattern: (Fall|Spring|Summer) YYYY
        pattern = r'\b(spring|summer|fall|autumn|winter)\s+(19|20)\d{2}\b'
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)

        for match in matches:
            semester_name = match.group(1).lower()
            year = int(match.group(2) + match.group(0).split()[-1][-2:])

            semester_info = SEMESTER_MAPPING.get(semester_name)
            if semester_info:
                month = semester_info["month"]
                normalized = f"{year}-{month:02d}"

                expressions.append(TemporalExpression(
                    original_text=match.group(0),
                    normalized_value=normalized,
                    temporal_type=TemporalType.DATE,
                    granularity=Granularity.MONTH,
                    confidence=0.95,
                    metadata={
                        "semester": semester_info["semester_num"],
                        "year": year
                    }
                ))

        return expressions

    def _normalize_quarter(self, text: str) -> List[TemporalExpression]:
        """
        Normalize business quarter expressions.

        Examples:
            "Q1 2023" -> "2023-01"
            "first quarter 2024" -> "2024-01"
        """
        expressions = []
        text_lower = text.lower()

        # Pattern: Q[1-4] YYYY
        pattern = r'\bq([1-4])\s+(19|20)\d{2}\b'
        matches = re.finditer(pattern, text_lower)

        for match in matches:
            quarter_num = int(match.group(1))
            year = int(match.group(2) + match.group(0).split()[-1][-2:])

            quarter_key = f"q{quarter_num}"
            quarter_info = QUARTER_MAPPING[quarter_key]

            month = quarter_info["month"]
            normalized = f"{year}-{month:02d}"

            expressions.append(TemporalExpression(
                original_text=match.group(0),
                normalized_value=normalized,
                temporal_type=TemporalType.DATE,
                granularity=Granularity.MONTH,
                confidence=0.95,
                metadata={
                    "quarter": quarter_num,
                    "year": year
                }
            ))

        return expressions

    def _normalize_year(self, text: str) -> List[TemporalExpression]:
        """
        Normalize standalone year expressions.

        Examples:
            "2023" -> "2023"
            "in 2020" -> "2020"
        """
        expressions = []

        # Pattern: 4-digit year
        pattern = r'\b(19|20)\d{2}\b'
        matches = re.finditer(pattern, text)

        for match in matches:
            year = match.group(0)

            expressions.append(TemporalExpression(
                original_text=year,
                normalized_value=year,
                temporal_type=TemporalType.DATE,
                granularity=Granularity.YEAR,
                confidence=0.9,
                metadata={"year": int(year)}
            ))

        return expressions

    def _normalize_month_year(self, text: str) -> List[TemporalExpression]:
        """
        Normalize month-year expressions using dateparser.

        Examples:
            "January 2023" -> "2023-01"
            "Dec 2024" -> "2024-12"
        """
        expressions = []

        if not DATEPARSER_AVAILABLE:
            return expressions

        # Pattern: (Month name/abbr) YYYY
        pattern = r'\b([A-Za-z]+)\s+(19|20)\d{2}\b'
        matches = re.finditer(pattern, text, re.IGNORECASE)

        for match in matches:
            try:
                # Use dateparser for robust month parsing
                parsed = dateparser.parse(match.group(0), settings=self.dateparser_settings)
                if parsed:
                    normalized = parsed.strftime("%Y-%m")

                    expressions.append(TemporalExpression(
                        original_text=match.group(0),
                        normalized_value=normalized,
                        temporal_type=TemporalType.DATE,
                        granularity=Granularity.MONTH,
                        confidence=0.95,
                        metadata={
                            "year": parsed.year,
                            "month": parsed.month,
                            "source": "dateparser"
                        }
                    ))
            except Exception:
                continue  # Skip invalid matches

        return expressions

    def _normalize_relative_date(self, text: str) -> List[TemporalExpression]:
        """
        Normalize relative date expressions using dateparser.

        Examples:
            "last year" -> "2023" (if current year is 2024)
            "next month" -> "2024-02" (if current month is 2024-01)
            "2 days ago" -> "2024-03-16"
        """
        expressions = []

        if not DATEPARSER_AVAILABLE:
            return expressions

        # Common relative patterns
        relative_patterns = [
            r'\blast\s+year\b',
            r'\bprevious\s+year\b',
            r'\bthis\s+year\b',
            r'\bcurrent\s+year\b',
            r'\bnext\s+year\b',
            r'\blast\s+month\b',
            r'\bthis\s+month\b',
            r'\bnext\s+month\b',
            r'\b\d+\s+(days?|weeks?|months?|years?)\s+ago\b',
        ]

        for pattern in relative_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    parsed = dateparser.parse(match.group(0), settings=self.dateparser_settings)
                    if parsed:
                        # Infer granularity
                        granularity = self._infer_granularity(match.group(0))

                        if granularity == Granularity.YEAR:
                            normalized = parsed.strftime("%Y")
                        elif granularity == Granularity.MONTH:
                            normalized = parsed.strftime("%Y-%m")
                        else:
                            normalized = parsed.strftime("%Y-%m-%d")

                        expressions.append(TemporalExpression(
                            original_text=match.group(0),
                            normalized_value=normalized,
                            temporal_type=TemporalType.DATE,
                            granularity=granularity,
                            confidence=0.9,
                            metadata={"relative": True, "source": "dateparser"}
                        ))
                except Exception:
                    continue

        return expressions

    def _normalize_duration(self, text: str) -> List[TemporalExpression]:
        """
        Normalize duration expressions.

        Examples:
            "3 years" -> "P3Y" (ISO 8601 duration)
            "2 weeks" -> "P2W"
            "5 days" -> "P5D"
        """
        expressions = []
        text_lower = text.lower()

        # Pattern: NUMBER (years|months|weeks|days|hours|minutes)
        duration_pattern = r'\b(\d+)\s+(years?|months?|weeks?|days?|hours?|minutes?|seconds?)\b'
        matches = re.finditer(duration_pattern, text_lower)

        for match in matches:
            amount = int(match.group(1))
            unit = match.group(2).rstrip('s')  # Remove plural 's'

            # Convert to ISO 8601 duration format
            unit_map = {
                'year': 'Y',
                'month': 'M',
                'week': 'W',
                'day': 'D',
                'hour': 'H',
                'minute': 'M',
                'second': 'S'
            }

            iso_unit = unit_map.get(unit, 'D')

            # ISO 8601: P for period, T for time
            if iso_unit in ['H', 'M', 'S']:
                normalized = f"PT{amount}{iso_unit}"
            else:
                normalized = f"P{amount}{iso_unit}"

            expressions.append(TemporalExpression(
                original_text=match.group(0),
                normalized_value=normalized,
                temporal_type=TemporalType.DURATION,
                granularity=Granularity[unit.upper()],
                confidence=0.9,
                metadata={
                    "amount": amount,
                    "unit": unit
                }
            ))

        return expressions

    def _normalize_with_dateparser(self, text: str) -> Optional[TemporalExpression]:
        """
        Use dateparser library as fallback.

        Only used if specific patterns don't match.
        """
        if not DATEPARSER_AVAILABLE:
            return None

        try:
            parsed_date = dateparser.parse(text, settings=self.dateparser_settings)

            if parsed_date:
                # Determine granularity from text
                granularity = self._infer_granularity(text)

                # Format according to granularity
                if granularity == Granularity.YEAR:
                    normalized = parsed_date.strftime("%Y")
                elif granularity == Granularity.MONTH:
                    normalized = parsed_date.strftime("%Y-%m")
                else:
                    normalized = parsed_date.strftime("%Y-%m-%d")

                return TemporalExpression(
                    original_text=text,
                    normalized_value=normalized,
                    temporal_type=TemporalType.DATE,
                    granularity=granularity,
                    confidence=0.8,  # Lower confidence for fallback
                    metadata={"source": "dateparser"}
                )

        except Exception as e:
            print(f"[WARN] dateparser failed: {e}")

        return None

    def _infer_granularity(self, text: str) -> Granularity:
        """Infer temporal granularity from text."""
        text_lower = text.lower()

        if any(word in text_lower for word in ['year', 'annual', 'yearly']):
            return Granularity.YEAR
        elif any(word in text_lower for word in ['month', 'monthly']):
            return Granularity.MONTH
        elif any(word in text_lower for word in ['week', 'weekly']):
            return Granularity.WEEK
        elif any(word in text_lower for word in ['day', 'daily']):
            return Granularity.DAY
        elif any(word in text_lower for word in ['hour', 'hourly']):
            return Granularity.HOUR
        else:
            return Granularity.DAY  # Default


# ==========================
# TEMPORAL VALUE MATCHER
# ==========================

class TemporalValueMatcher:
    """
    Match normalized temporal values to database schema.

    Handles different database representations:
    - Integer year columns (2023)
    - Date columns (2023-09-15)
    - Semester/quarter columns (1, 2, 3)
    - Text columns ("Fall 2023")
    """

    def __init__(self, schema):
        self.schema = schema

    def match_to_schema(self, temporal_expr: TemporalExpression,
                       column_name: str, column_type: str) -> Any:
        """
        Convert normalized temporal value to schema-appropriate format.

        Args:
            temporal_expr: Normalized temporal expression
            column_name: Target database column
            column_type: Database column type

        Returns:
            Value in appropriate format for schema
        """
        normalized = temporal_expr.normalized_value

        # Integer year column
        if column_type.lower() in ['int', 'integer', 'number']:
            if temporal_expr.granularity == Granularity.YEAR:
                return int(normalized)
            elif 'semester' in temporal_expr.metadata:
                return temporal_expr.metadata['semester']
            elif 'quarter' in temporal_expr.metadata:
                return temporal_expr.metadata['quarter']

        # Date column
        elif column_type.lower() in ['date', 'datetime', 'timestamp']:
            return normalized

        # Text column
        elif column_type.lower() in ['text', 'varchar', 'char', 'string']:
            return temporal_expr.original_text

        # Default: return normalized value
        return normalized


# ==========================
# TESTING
# ==========================

if __name__ == "__main__":
    print("Temporal Normalization v0.1")
    print("=" * 60)

    normalizer = TemporalNormalizer()

    # Test cases
    test_cases = [
        "Fall 2023",
        "Spring 2024",
        "Q1 2023",
        "January 2023",
        "last year",
        "3 years",
        "2 weeks",
        "2023",
    ]

    print("\nTemporal Normalization Tests:")
    for test in test_cases:
        print(f"\n  Input: {test}")
        expressions = normalizer.normalize(test)

        if expressions:
            for expr in expressions:
                print(f"    Normalized: {expr.normalized_value}")
                print(f"    Type: {expr.temporal_type.value}")
                print(f"    Granularity: {expr.granularity.value}")
                print(f"    Confidence: {expr.confidence:.2f}")
                if expr.metadata:
                    print(f"    Metadata: {expr.metadata}")
        else:
            print("    No temporal expression found")

    print("\n[OK] Temporal normalization ready")
    if not DATEPARSER_AVAILABLE:
        print("[INFO] Install dateparser for enhanced date parsing: pip install dateparser")
