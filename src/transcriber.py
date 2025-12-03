"""Transcription using faster-whisper."""
import logging
from typing import List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass 
class TranscriptionResult:
    """Result of transcribing an audio file."""
    segments: List[dict] = field(default_factory=list)
    audio_duration: float = 0.0
    language: str = "en"

    def get_all_words(self):
        """Get all words with timestamps."""
        from .config import WordTimestamp
        words = []
        for seg in self.segments:
            for w in seg.get("words", []):
                words.append(WordTimestamp(
                    word=w["word"],
                    start=w["start"],
                    end=w["end"]
                ))
        return words

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "segments": self.segments,
            "audio_duration": self.audio_duration,
            "language": self.language
        }
