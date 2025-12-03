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
    """Normalize a word for matching."""
    if not word:
        return ""
    word = word.lower()
    # Remove possessives
    word = re.sub(r"['']s$", "", word)
    # Remove punctuation
    word = word.rstrip(".,!?;:\"'")
    word = word.lstrip("\"'")
    return word


def normalize_phrase(phrase: str) -> str:
    """Normalize a multi-word phrase."""
    words = phrase.split()
    return " ".join(normalize_word(w) for w in words)
