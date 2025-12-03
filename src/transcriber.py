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
