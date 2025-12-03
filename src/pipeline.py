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


class Pipeline:
    """Main PII de-identification pipeline."""
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        whisper_model: str = WHISPER_MODEL,
        verify_audio: bool = True,
        save_outputs: bool = True
    ):
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.whisper_model = whisper_model
        self.verify_audio = verify_audio
        self.save_outputs = save_outputs
        
        # Init components
        self.transcriber = Transcriber(model_size=whisper_model)
        self.detector = PIIDetector()
        self.text_redactor = TextRedactor()
        self.audio_redactor = AudioRedactor()
        self.verifier = Verifier()
        
        if save_outputs:
            self._create_output_dirs()
    
    def _create_output_dirs(self):
        dirs = [
            self.output_dir / "audio" / "train",
            self.output_dir / "transcripts_raw" / "train",
            self.output_dir / "transcripts_deid" / "train",
            self.output_dir / "metadata",
            self.output_dir / "qa"
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
