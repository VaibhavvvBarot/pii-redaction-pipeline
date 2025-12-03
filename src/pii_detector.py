"""PII Detection - exact match first, then fuzzy."""
import re
import logging
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass

from .lexicon import (
    DAYS, MONTHS, COLORS, STATES, CITIES_MULTI, CITIES_SINGLE,
    CATEGORY_LABELS, get_sorted_terms_by_length
)
from .config import PIIMatch, WordTimestamp

logger = logging.getLogger(__name__)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def normalize_word(word: str) -> str:
    """Normalize a word for matching. Handles possessives and plurals."""
    if not word:
        return ""
    word = word.lower()
    word = re.sub(r"['']s$", "", word)
    word = word.rstrip(".,!?;:\"'")
    word = word.lstrip("\"'")
    
    # Handle plurals like "Mondays" -> "monday"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        singular = word[:-1]
        all_pii_lower = set(DAYS + MONTHS + COLORS + STATES + CITIES_SINGLE)
        if singular in all_pii_lower:
            word = singular
    return word


def normalize_phrase(phrase: str) -> str:
    """Normalize a multi-word phrase."""
    words = phrase.split()
    return " ".join(normalize_word(w) for w in words)


# Context patterns for "may" as month vs modal verb
MAY_MONTH_PATTERNS = [
    r'\b(in|during|last|next|this|of|since|before|after|until|by)\s+may\b',
    r'\bmay\s+\d{1,2}(st|nd|rd|th)?\b',
    r'\bmay\s+of\s+\d{4}\b',
    r'^may\s+\d',
]


def is_may_month(text: str, match_start: int, match_end: int) -> bool:
    """Check if 'may' is the month (True) or modal verb (False)."""
    context_start = max(0, match_start - 20)
    context_end = min(len(text), match_end + 20)
    context = text[context_start:context_end].lower()
    
    for pattern in MAY_MONTH_PATTERNS:
        if re.search(pattern, context, re.IGNORECASE):
            return True
    return False
