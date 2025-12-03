"""Text redaction - replaces PII with category labels."""
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from .config import PIIMatch
from .lexicon import CATEGORY_LABELS

logger = logging.getLogger(__name__)


@dataclass
class RedactedTranscript:
    """A redacted transcript."""
    conversation_id: str
    original_text: str
    redacted_text: str
    pii_count: int
    redactions: List[Dict] = field(default_factory=list)
    
    def to_dict(self):
        return {
            "conversation_id": self.conversation_id,
            "original_text": self.original_text,
            "redacted_text": self.redacted_text,
            "pii_count": self.pii_count,
            "redactions": self.redactions
        }


class TextRedactor:
    """Redacts PII from transcript text."""
    
    def redact(self, transcript, pii_matches: List[PIIMatch]) -> RedactedTranscript:
        """Redact PII from transcript."""
        original_text = " ".join(
            seg["text"] for seg in transcript.segments
        )
        
        if not pii_matches:
            return RedactedTranscript(
                conversation_id="",
                original_text=original_text,
                redacted_text=original_text,
                pii_count=0,
                redactions=[]
            )
        
        # Get all words
        all_words = transcript.get_all_words()
        redacted_words = [w.word for w in all_words]
        redactions = []
        
        # Apply redactions
        for match in pii_matches:
            label = CATEGORY_LABELS.get(match.category, f"[{match.category.upper()}]")
            
            # Replace words at matched indices
            for idx in match.word_indices:
                if idx < len(redacted_words):
                    redacted_words[idx] = label if idx == match.word_indices[0] else ""
            
            redactions.append({
                "original": match.text,
                "replacement": label,
                "category": match.category,
                "start_time": match.start_time,
                "end_time": match.end_time
            })
        
        # Build redacted text
        redacted_text = " ".join(w for w in redacted_words if w)
        
        return RedactedTranscript(
            conversation_id="",
            original_text=original_text,
            redacted_text=redacted_text,
            pii_count=len(pii_matches),
            redactions=redactions
        )
