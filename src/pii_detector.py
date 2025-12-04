"""
PII Detection with 2-layer detection: exact match first, then fuzzy.
Detects days, months, colors, cities, and states.
Uses longest-first matching for multi-word entities like "New York City".
"""
import re
import logging
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass

from .lexicon import (
    DAYS, MONTHS, COLORS, STATES, CITIES_MULTI, CITIES_SINGLE,
    CATEGORY_LABELS, get_sorted_terms_by_length
)
from .config import (
    FUZZY_MAX_DISTANCE, FUZZY_MIN_CONFIDENCE,
    PIIMatch, WordTimestamp
)
from .transcriber import TranscriptionResult

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
    """
    Normalize a word for matching.
    Handles case, possessives (Monday's), punctuation, and simple plurals.
    """
    if not word:
        return ""

    word = word.lower()

    # Remove possessives ('s, 's)
    word = re.sub(r"['']s$", "", word)

    # Remove trailing punctuation
    word = word.rstrip(".,!?;:\"'")

    # Remove leading punctuation
    word = word.lstrip("\"'")

    # Handle simple plurals for day/month names
    # "Mondays" → "Monday", "Tuesdays" → "Tuesday"
    # But NOT "dress" → "dres" (ends in ss)
    # And NOT "atlas" → "atla" (not a known plural pattern)
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        singular = word[:-1]
        # Only apply if the singular form is a known PII term
        all_pii_lower = set(DAYS + MONTHS + COLORS + STATES + CITIES_SINGLE)
        if singular in all_pii_lower:
            word = singular

    return word


def normalize_phrase(phrase: str) -> str:
    """Normalize a multi-word phrase."""
    words = phrase.split()
    return " ".join(normalize_word(w) for w in words)


# Context patterns for "may" as a month (not modal verb)
MAY_MONTH_PATTERNS = [
    r'\b(in|during|last|next|this|of|since|before|after|until|by)\s+may\b',
    r'\bmay\s+\d{1,2}(st|nd|rd|th)?\b',  # May 15, May 1st
    r'\bmay\s+of\s+\d{4}\b',              # May of 2024
    r'^may\s+\d',                          # May at start followed by number
]


def is_may_month(text: str, match_start: int, match_end: int) -> bool:
    """
    Check if "may" is the month (True) or modal verb (False).
    Looks at surrounding context for patterns like "in May" or "May 15th".
    """
    # Get surrounding context
    context_start = max(0, match_start - 20)
    context_end = min(len(text), match_end + 20)
    context = text[context_start:context_end].lower()

    # Check if any month pattern matches
    for pattern in MAY_MONTH_PATTERNS:
        if re.search(pattern, context, re.IGNORECASE):
            return True

    return False


class PIIDetector:
    """
    Detects PII in transcripts using 2-layer detection.
    Layer 1: Exact match (longest-first)
    Layer 2: Fuzzy match (Levenshtein <= 2)

    Cities are matched before colors to prevent "Brownsville" -> "[COLOR]sville".
    """

    def __init__(self):
        """Initialize the detector with sorted PII terms."""
        # Get terms sorted by length (longest first)
        # Cities come before colors in the sorted list
        self.sorted_terms = get_sorted_terms_by_length()

        # Build lookup sets for fast checking
        self.pii_sets: Dict[str, Set[str]] = {
            "day": set(d.lower() for d in DAYS),
            "month": set(m.lower() for m in MONTHS),
            "color": set(c.lower() for c in COLORS),
            "state": set(s.lower() for s in STATES),
            "city": set(c.lower() for c in CITIES_MULTI + CITIES_SINGLE)
        }

        # All PII terms for fuzzy matching
        self.all_pii_terms = set()
        for terms in self.pii_sets.values():
            self.all_pii_terms.update(terms)

        # Term to category mapping
        self.term_to_category: Dict[str, str] = {}
        for term, category in self.sorted_terms:
            if term.lower() not in self.term_to_category:
                self.term_to_category[term.lower()] = category

    def detect(self, transcript: TranscriptionResult) -> List[PIIMatch]:
        """Detect all PII in a transcript. Returns list of PIIMatch objects."""
        all_words = transcript.get_all_words()
        if not all_words:
            return []

        matches: List[PIIMatch] = []
        matched_indices: Set[int] = set()

        # Layer 1: Exact matching (longest-first)
        exact_matches = self._exact_match(all_words, matched_indices)
        matches.extend(exact_matches)

        # Layer 2: Fuzzy matching for unmatched words
        fuzzy_matches = self._fuzzy_match(all_words, matched_indices)
        matches.extend(fuzzy_matches)

        # Sort by start time
        matches.sort(key=lambda m: m.start_time)

        logger.info(
            f"Detected {len(matches)} PII: "
            f"{len(exact_matches)} exact, {len(fuzzy_matches)} fuzzy"
        )

        return matches

    def _exact_match(
        self,
        words: List[WordTimestamp],
        matched_indices: Set[int]
    ) -> List[PIIMatch]:
        """Layer 1: Exact matching, longest phrases first. Cities before colors."""
        matches: List[PIIMatch] = []
        n_words = len(words)

        # Try to match phrases of decreasing length
        # Max phrase length we'll try (most multi-word cities are 3 words)
        max_phrase_len = 4

        i = 0
        while i < n_words:
            if i in matched_indices:
                i += 1
                continue

            matched = False

            # Try longest phrases first
            for phrase_len in range(min(max_phrase_len, n_words - i), 0, -1):
                if matched:
                    break

                # Build normalized phrase from words
                word_slice = words[i:i + phrase_len]
                phrase_words = [normalize_word(w.word) for w in word_slice]
                phrase = " ".join(phrase_words)

                # Check if this phrase is in our lexicon
                for term, category in self.sorted_terms:
                    term_lower = term.lower()
                    term_words = term_lower.split()

                    if len(term_words) != phrase_len:
                        continue

                    if phrase == term_lower:
                        # Special handling for "may"
                        if term_lower == "may":
                            # Get full text context
                            full_text = " ".join(w.word for w in words)
                            word_pos = sum(len(w.word) + 1 for w in words[:i])
                            if not is_may_month(full_text, word_pos, word_pos + 3):
                                continue

                        # Found exact match
                        indices = list(range(i, i + phrase_len))
                        matched_indices.update(indices)

                        match = PIIMatch(
                            text=" ".join(w.word for w in word_slice),
                            category=category,
                            start_time=word_slice[0].start,
                            end_time=word_slice[-1].end,
                            confidence=1.0,
                            word_indices=indices,
                            is_fuzzy=False
                        )
                        matches.append(match)
                        matched = True
                        i += phrase_len - 1  # -1 because we'll increment at end
                        break

            i += 1

        return matches

    def _fuzzy_match(
        self,
        words: List[WordTimestamp],
        matched_indices: Set[int]
    ) -> List[PIIMatch]:
        """
        Layer 2: Fuzzy matching to catch Whisper transcription errors.
        Only single words, Levenshtein <= 2, min 5 chars to avoid false positives.
        """
        matches: List[PIIMatch] = []

        # Common words that should NEVER be fuzzy matched
        # (too short, common, or known false positives)
        FUZZY_BLACKLIST = {
            # Short words - too risky for fuzzy matching
            "like", "back", "lack", "lick", "lock", "luck",  # Not colors
            "read", "lead", "bead", "dead", "head",  # Not "red"
            "plan", "clan", "scan",  # Not "tan"
            "lime", "time", "dime", "mime",  # These are distinct words
            "goal", "coal", "foal",  # Not "gold"
            "pin", "tin", "bin", "fin", "win", "sin",  # Not "pink"
            "pint", "pine", "ping",  # Not "pink"
            "tank", "sank", "rank", "bank",  # Not "tan"
            "beat", "heat", "meat", "neat", "seat",  # Not "teal"
            "tale", "tall",  # Not "teal"
            # Longer words that are common and not PII
            "remember", "november", "september", "december",  # Common words/months in context
            "member", "ember",  # Parts of month names
            "around", "round", "sound", "found", "bound",  # Common words
            "texture", "mixture", "fixture",  # Not "texas"
            "salon", "gallon", "talon",  # Not "salmon"
        }

        for i, word_ts in enumerate(words):
            if i in matched_indices:
                continue

            word = normalize_word(word_ts.word)

            # Skip words in blacklist
            if word in FUZZY_BLACKLIST:
                continue

            # Require minimum length of 5 for fuzzy matching
            # (4-letter words like "back", "like" are too risky)
            if len(word) < 5:
                continue

            # Find best fuzzy match
            best_match: Optional[Tuple[str, str, int]] = None  # (term, category, distance)

            for term, category in self.sorted_terms:
                term_lower = term.lower()

                # Only fuzzy match single words
                if " " in term_lower:
                    continue

                # Target term must also be long enough
                if len(term_lower) < 5:
                    continue

                distance = levenshtein_distance(word, term_lower)

                if distance == 0:
                    # This should have been caught by exact match
                    continue

                # For distance=2, require longer words (≥7 chars)
                if distance == 2 and len(word) < 7:
                    continue

                if distance <= FUZZY_MAX_DISTANCE:
                    # Check relative distance (don't match if too much of word is different)
                    relative_distance = distance / max(len(word), len(term_lower))
                    if relative_distance > 0.25:  # Stricter threshold
                        continue

                    if best_match is None or distance < best_match[2]:
                        best_match = (term_lower, category, distance)

            if best_match:
                term, category, distance = best_match
                confidence = 1.0 - (distance / max(len(word), len(term)))

                if confidence >= FUZZY_MIN_CONFIDENCE:
                    # Special handling for "may" fuzzy matches
                    if term == "may":
                        full_text = " ".join(w.word for w in words)
                        word_pos = sum(len(w.word) + 1 for w in words[:i])
                        if not is_may_month(full_text, word_pos, word_pos + len(word)):
                            continue

                    matched_indices.add(i)
                    match = PIIMatch(
                        text=word_ts.word,
                        category=category,
                        start_time=word_ts.start,
                        end_time=word_ts.end,
                        confidence=confidence,
                        word_indices=[i],
                        is_fuzzy=True
                    )
                    matches.append(match)

                    logger.debug(
                        f"Fuzzy match: '{word_ts.word}' -> '{term}' "
                        f"(distance={distance}, confidence={confidence:.2f})"
                    )

        return matches

    def detect_in_text(self, text: str) -> List[Dict]:
        """Detect PII in plain text (used for verification)."""
        matches = []
        text_lower = text.lower()

        # Track matched character positions
        matched_positions: Set[int] = set()

        # Match phrases (longest first)
        for term, category in self.sorted_terms:
            term_lower = term.lower()

            # Find all occurrences
            start = 0
            while True:
                # Use word boundary matching
                pattern = r'\b' + re.escape(term_lower) + r'\b'
                match = re.search(pattern, text_lower[start:], re.IGNORECASE)

                if not match:
                    break

                abs_start = start + match.start()
                abs_end = start + match.end()

                # Check if already matched
                if any(pos in matched_positions for pos in range(abs_start, abs_end)):
                    start = abs_start + 1
                    continue

                # Special handling for "may"
                if term_lower == "may":
                    if not is_may_month(text, abs_start, abs_end):
                        start = abs_end
                        continue

                # Mark positions as matched
                for pos in range(abs_start, abs_end):
                    matched_positions.add(pos)

                matches.append({
                    "text": text[abs_start:abs_end],
                    "category": category,
                    "start": abs_start,
                    "end": abs_end
                })

                start = abs_end

        # Sort by position
        matches.sort(key=lambda m: m["start"])
        return matches


def detect_pii(transcript: TranscriptionResult) -> List[PIIMatch]:
    """Convenience function to detect PII in a transcript."""
    detector = PIIDetector()
    return detector.detect(transcript)
