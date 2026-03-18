"""
Shared matching utilities - DRY principle.

Consolidates fuzzy matching, schema linking, and grounding logic
to avoid duplication across modules.
"""

from typing import List, Tuple, Optional
from rapidfuzz import fuzz


def fuzzy_match(query: str, target: str, threshold: float = 75.0) -> float:
    """
    Single source of truth for fuzzy string matching.

    Args:
        query: Query string
        target: Target string to match against
        threshold: Minimum score to consider a match

    Returns:
        Fuzzy match score (0-100)
    """
    return fuzz.WRatio(query.lower(), target.lower())


def find_best_match(
    query: str,
    candidates: List[str],
    threshold: float = 75.0
) -> Optional[Tuple[str, float]]:
    """
    Find best matching candidate for query.

    Args:
        query: Query string
        candidates: List of candidate strings
        threshold: Minimum score threshold

    Returns:
        (best_match, score) or None if no match above threshold
    """
    best_match = None
    best_score = 0.0

    for candidate in candidates:
        score = fuzzy_match(query, candidate, threshold)
        if score > best_score:
            best_score = score
            best_match = candidate

    if best_score >= threshold:
        return (best_match, best_score)
    return None


def normalize_name(name: str) -> str:
    """
    Normalize schema/column names for matching.

    Args:
        name: Raw name

    Returns:
        Normalized name (lowercase, underscores)
    """
    return name.lower().replace(' ', '_').replace('-', '_')
