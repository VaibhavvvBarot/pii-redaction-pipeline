"""Main pipeline - runs all steps end to end."""
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .config import ProcessingResult, OUTPUT_DIR, WHISPER_MODEL, OUTPUT_AUDIO_FORMAT
from .transcriber import Transcriber, TranscriptionResult
from .pii_detector import PIIDetector, PIIMatch
from .text_redactor import TextRedactor, RedactedTranscript
from .audio_redactor import AudioRedactor, BleepRegion
from .verifier import Verifier, VerificationResult, VerificationStatus

logger = logging.getLogger(__name__)


@dataclass
class ConversationOutput:
    """Output for a processed conversation."""
    conversation_id: str
    success: bool
    error: Optional[str] = None
    stage: Optional[str] = None
    transcript_raw: Optional[TranscriptionResult] = None
    transcript_redacted: Optional[RedactedTranscript] = None
    pii_matches: List[PIIMatch] = field(default_factory=list)
    bleep_regions: List[BleepRegion] = field(default_factory=list)
    redacted_audio_path: Optional[str] = None
    verification: Optional[VerificationResult] = None
    processing_time_s: float = 0.0
