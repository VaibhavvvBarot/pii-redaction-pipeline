"""
Text redaction - replaces PII in transcripts with labels like [CITY], [STATE], [DAY].
"""
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

from .config import PIIMatch, WordTimestamp
from .lexicon import CATEGORY_LABELS
from .transcriber import TranscriptionResult, TranscriptionSegment

logger = logging.getLogger(__name__)


@dataclass
class RedactionLog:
    """Log of a single redaction."""
    original_text: str
    replacement: str
    category: str
    start_time: float
    end_time: float
    confidence: float
    is_fuzzy: bool


@dataclass
class RedactedTranscript:
    """A redacted transcript with logs."""
    conversation_id: str
    original_text: str
    redacted_text: str
    redacted_segments: List[TranscriptionSegment]
    redaction_logs: List[RedactionLog]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "conversation_id": self.conversation_id,
            "original_text": self.original_text,
            "redacted_text": self.redacted_text,
            "segments": [
                {
                    "text": seg.text,
                    "start": seg.start,
                    "end": seg.end,
                    "words": [
                        {"word": w.word, "start": w.start, "end": w.end}
                        for w in seg.words
                    ]
                }
                for seg in self.redacted_segments
            ],
            "redactions": [
                {
                    "original": log.original_text,
                    "replacement": log.replacement,
                    "category": log.category,
                    "start_time": log.start_time,
                    "end_time": log.end_time,
                    "confidence": log.confidence,
                    "is_fuzzy": log.is_fuzzy
                }
                for log in self.redaction_logs
            ],
            "pii_count": len(self.redaction_logs)
        }


class TextRedactor:
    """Redacts PII from transcripts."""

    def __init__(self):
        """Initialize the text redactor."""
        self.labels = CATEGORY_LABELS

    def redact(
        self,
        transcript: TranscriptionResult,
        pii_matches: List[PIIMatch]
    ) -> RedactedTranscript:
        """
        Redact PII from a transcript.

        Args:
            transcript: TranscriptionResult with word timestamps
            pii_matches: List of PII matches to redact

        Returns:
            RedactedTranscript with original and redacted text
        """
        # Build set of word indices to redact
        word_to_match: Dict[int, PIIMatch] = {}
        for match in pii_matches:
            for idx in match.word_indices:
                word_to_match[idx] = match

        # Get all words
        all_words = transcript.get_all_words()

        # Process segments
        redacted_segments = []
        redaction_logs = []

        global_word_idx = 0

        for segment in transcript.segments:
            redacted_words = []
            redacted_word_timestamps = []

            for word_ts in segment.words:
                if global_word_idx in word_to_match:
                    match = word_to_match[global_word_idx]

                    # Only replace on first word of multi-word match
                    if global_word_idx == match.word_indices[0]:
                        label = self.labels.get(match.category, f"[{match.category.upper()}]")
                        redacted_words.append(label)

                        # Create a word timestamp for the label
                        redacted_word_timestamps.append(WordTimestamp(
                            word=label,
                            start=match.start_time,
                            end=match.end_time,
                            confidence=match.confidence
                        ))

                        # Log the redaction
                        redaction_logs.append(RedactionLog(
                            original_text=match.text,
                            replacement=label,
                            category=match.category,
                            start_time=match.start_time,
                            end_time=match.end_time,
                            confidence=match.confidence,
                            is_fuzzy=match.is_fuzzy
                        ))
                    # Skip subsequent words in multi-word match
                else:
                    redacted_words.append(word_ts.word)
                    redacted_word_timestamps.append(word_ts)

                global_word_idx += 1

            # Create redacted segment
            redacted_text = " ".join(redacted_words)
            redacted_segments.append(TranscriptionSegment(
                text=redacted_text,
                start=segment.start,
                end=segment.end,
                words=redacted_word_timestamps
            ))

        # Build full texts
        original_text = transcript.get_full_text()
        redacted_full_text = " ".join(seg.text for seg in redacted_segments)

        result = RedactedTranscript(
            conversation_id=transcript.conversation_id,
            original_text=original_text,
            redacted_text=redacted_full_text,
            redacted_segments=redacted_segments,
            redaction_logs=redaction_logs
        )

        logger.info(
            f"Redacted transcript: {len(redaction_logs)} PII instances replaced"
        )

        return result


def redact_text(
    transcript: TranscriptionResult,
    pii_matches: List[PIIMatch]
) -> RedactedTranscript:
    """
    Convenience function to redact PII from a transcript.

    Args:
        transcript: TranscriptionResult
        pii_matches: List of PII matches

    Returns:
        RedactedTranscript
    """
    redactor = TextRedactor()
    return redactor.redact(transcript, pii_matches)
