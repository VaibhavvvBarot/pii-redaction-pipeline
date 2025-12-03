"""Config for the PII pipeline."""
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

OUTPUT_DIR = Path("output")
WHISPER_MODEL = "base"
SAMPLE_RATE = 16000

# Audio redaction
MIN_BLEEP_DURATION_MS = 400
BLEEP_FREQUENCY_HZ = 1000
BLEEP_AMPLITUDE = 0.5
PADDING_BEFORE_MS = 150
PADDING_AFTER_MS = 150
OUTPUT_AUDIO_FORMAT = "flac"

@dataclass
class WordTimestamp:
    """A word with its timestamp."""
    word: str
    start: float
    end: float

@dataclass
class PIIMatch:
    """A detected PII match."""
    text: str
    category: str
    start_time: float
    end_time: float
    confidence: float
    word_indices: List[int]
    is_fuzzy: bool = False

@dataclass
class ProcessingResult:
    """Result of processing a conversation."""
    conversation_id: str
    success: bool
    error: Optional[str] = None
