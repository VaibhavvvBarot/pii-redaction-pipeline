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


class PIIDetector:
    """Detects PII using 2-layer detection: exact match then fuzzy."""
    
    def __init__(self):
        self.sorted_terms = get_sorted_terms_by_length()
        
        # Build lookup sets
        self.pii_sets: Dict[str, Set[str]] = {
            "day": set(d.lower() for d in DAYS),
            "month": set(m.lower() for m in MONTHS),
            "color": set(c.lower() for c in COLORS),
            "state": set(s.lower() for s in STATES),
            "city": set(c.lower() for c in CITIES_MULTI + CITIES_SINGLE)
        }
        
        self.all_pii_terms = set()
        for terms in self.pii_sets.values():
            self.all_pii_terms.update(terms)
        
        # Term to category mapping
        self.term_to_category: Dict[str, str] = {}
        for term, category in self.sorted_terms:
            if term.lower() not in self.term_to_category:
                self.term_to_category[term.lower()] = category

    def detect(self, transcript) -> List[PIIMatch]:
        """Detect all PII in a transcript."""
        all_words = transcript.get_all_words()
        if not all_words:
            return []
        
        matches: List[PIIMatch] = []
        matched_indices: Set[int] = set()
        
        # Layer 1: Exact matching
        exact_matches = self._exact_match(all_words, matched_indices)
        matches.extend(exact_matches)
        
        # Layer 2: Fuzzy matching
        fuzzy_matches = self._fuzzy_match(all_words, matched_indices)
        matches.extend(fuzzy_matches)
        
        matches.sort(key=lambda m: m.start_time)
        logger.info(f"Detected {len(matches)} PII: {len(exact_matches)} exact, {len(fuzzy_matches)} fuzzy")
        return matches

    def _exact_match(self, words: List[WordTimestamp], matched_indices: Set[int]) -> List[PIIMatch]:
        """Layer 1: Exact matching, longest phrases first."""
        matches = []
        n_words = len(words)
        max_phrase_len = 4
        
        i = 0
        while i < n_words:
            if i in matched_indices:
                i += 1
                continue
            
            matched = False
            for phrase_len in range(min(max_phrase_len, n_words - i), 0, -1):
                if matched:
                    break
                
                word_slice = words[i:i + phrase_len]
                phrase_words = [normalize_word(w.word) for w in word_slice]
                phrase = " ".join(phrase_words)
                
                for term, category in self.sorted_terms:
                    term_lower = term.lower()
                    term_words = term_lower.split()
                    
                    if len(term_words) != phrase_len:
                        continue
                    
                    if phrase == term_lower:
                        # Special handling for "may"
                        if term_lower == "may":
                            full_text = " ".join(w.word for w in words)
                            word_pos = sum(len(w.word) + 1 for w in words[:i])
                            if not is_may_month(full_text, word_pos, word_pos + 3):
                                continue
                        
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
                        i += phrase_len - 1
                        break
            i += 1
        return matches

    def _fuzzy_match(self, words: List[WordTimestamp], matched_indices: Set[int]) -> List[PIIMatch]:
        """Layer 2: Fuzzy matching to catch ASR errors."""
        matches = []
        
        # Words that should never be fuzzy matched
        FUZZY_BLACKLIST = {
            "like", "back", "lack", "read", "lead", "plan", "lime",
            "goal", "coal", "pin", "tin", "tank", "beat", "heat",
            "remember", "member", "around", "texture", "salon"
        }
        
        from .config import FUZZY_MAX_DISTANCE, FUZZY_MIN_CONFIDENCE
        
        for i, word_ts in enumerate(words):
            if i in matched_indices:
                continue
            
            word = normalize_word(word_ts.word)
            
            if word in FUZZY_BLACKLIST:
                continue
            
            # Min 5 chars for fuzzy matching
            if len(word) < 5:
                continue
            
            best_match = None
            
            for term, category in self.sorted_terms:
                term_lower = term.lower()
                
                if " " in term_lower:  # Only single words
                    continue
                if len(term_lower) < 5:
                    continue
                
                distance = levenshtein_distance(word, term_lower)
                
                if distance == 0:
                    continue
                
                # Stricter for distance=2
                if distance == 2 and len(word) < 7:
                    continue
                
                if distance <= FUZZY_MAX_DISTANCE:
                    relative_distance = distance / max(len(word), len(term_lower))
                    if relative_distance > 0.25:
                        continue
                    
                    if best_match is None or distance < best_match[2]:
                        best_match = (term_lower, category, distance)
            
            if best_match:
                term, category, distance = best_match
                confidence = 1.0 - (distance / max(len(word), len(term)))
                
                if confidence >= FUZZY_MIN_CONFIDENCE:
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
                    logger.debug(f"Fuzzy: '{word_ts.word}' -> '{term}' (dist={distance})")
        
        return matches

    def detect_in_text(self, text: str) -> List[Dict]:
        """Detect PII in plain text (for verification)."""
        matches = []
        text_lower = text.lower()
        matched_positions: Set[int] = set()
        
        for term, category in self.sorted_terms:
            term_lower = term.lower()
            start = 0
            
            while True:
                pattern = r'\b' + re.escape(term_lower) + r'\b'
                match = re.search(pattern, text_lower[start:], re.IGNORECASE)
                
                if not match:
                    break
                
                abs_start = start + match.start()
                abs_end = start + match.end()
                
                if any(pos in matched_positions for pos in range(abs_start, abs_end)):
                    start = abs_start + 1
                    continue
                
                if term_lower == "may":
                    if not is_may_month(text, abs_start, abs_end):
                        start = abs_end
                        continue
                
                for pos in range(abs_start, abs_end):
                    matched_positions.add(pos)
                
                matches.append({
                    "text": text[abs_start:abs_end],
                    "category": category,
                    "start": abs_start,
                    "end": abs_end
                })
                start = abs_end
        
        matches.sort(key=lambda m: m["start"])
        return matches


def detect_pii(transcript) -> List[PIIMatch]:
    """Convenience function to detect PII."""
    detector = PIIDetector()
    return detector.detect(transcript)
