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
