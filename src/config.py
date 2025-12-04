"""
Pipeline configuration.
"""
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data_exploration"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Audio settings
SAMPLE_RATE = 16000  # Hz
AUDIO_CHANNELS = 1   # Mono

# Transcription settings (faster-whisper)
# Model options: tiny, base, small, medium, large-v3
# Tested WER on conversational data:
#   base:   6.65% WER (fastest)
#   small:  5.30% WER
#   medium: 5.79% WER
#   large-v3: ~3-4% WER (best accuracy)
WHISPER_MODEL = "large-v3"  # Best accuracy
WHISPER_DEVICE = "auto"     # Will use MPS/CUDA if available, else CPU
WHISPER_COMPUTE_TYPE = "float16"  # Use float32 for CPU
WHISPER_BEAM_SIZE = 5       # Balance between speed and accuracy
WHISPER_LANGUAGE = "en"     # Force English

# PII Detection settings
FUZZY_MAX_DISTANCE = 2      # Maximum Levenshtein distance for fuzzy matching
FUZZY_MIN_CONFIDENCE = 0.7  # Minimum confidence for fuzzy matches

# Audio redaction settings
MIN_BLEEP_DURATION_MS = 400     # Minimum bleep duration
BLEEP_FREQUENCY_HZ = 1000       # Bleep tone frequency (1kHz)
BLEEP_AMPLITUDE = 0.5           # Bleep volume (50%)
PADDING_BEFORE_MS = 150         # Padding before PII word (±100ms accuracy + 50ms safety)
PADDING_AFTER_MS = 150          # Padding after PII word

# Verification thresholds
VERIFY_PASS_THRESHOLD = 0       # 0 PII found = PASS
VERIFY_REVIEW_THRESHOLD = 2    # ≤2 low-confidence = PASS_WITH_NOTE
VERIFY_FAIL_THRESHOLD = 2      # >2 = FAIL

# Output format
OUTPUT_AUDIO_FORMAT = "flac"    # Lossless compression


@dataclass
class ProcessingResult:
    """Result of processing a single conversation."""
    conversation_id: str
    success: bool
    stage: Optional[str] = None  # transcription, detection, redaction, verification
    error: Optional[str] = None
    pii_count: int = 0
    verification_status: Optional[str] = None  # PASS, PASS_WITH_NOTE, REVIEW, FAIL
    audio_duration_s: float = 0.0
    processing_time_s: float = 0.0


@dataclass
class PIIMatch:
    """A detected PII instance."""
    text: str                    # Original text matched
    category: str               # day, month, color, state, city
    start_time: float           # Start time in seconds
    end_time: float             # End time in seconds
    confidence: float           # 1.0 for exact, <1.0 for fuzzy
    word_indices: list          # Indices of words in transcript
    is_fuzzy: bool = False      # Whether this was a fuzzy match


@dataclass
class WordTimestamp:
    """A word with its timestamp."""
    word: str
    start: float  # seconds
    end: float    # seconds
    confidence: float = 1.0
