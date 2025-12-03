"""Verification - re-transcribe redacted audio and check for PII leakage."""
import logging
from typing import List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    PASS = "PASS"
    PASS_WITH_NOTE = "PASS_WITH_NOTE"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    FAIL = "FAIL"


@dataclass
class VerificationResult:
    """Result of verifying a redacted transcript."""
    overall_status: VerificationStatus
    transcript_status: VerificationStatus
    audio_status: Optional[VerificationStatus] = None
    leaked_pii: List[dict] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


class Verifier:
    """Verifies that PII was properly redacted."""
    
    def __init__(self):
        from .pii_detector import PIIDetector
        self.detector = PIIDetector()
    
    def verify(
        self,
        redacted_transcript,
        redacted_audio_path: Optional[str] = None,
        verify_audio: bool = True
    ) -> VerificationResult:
        """Verify that redacted content has no PII leakage."""
        leaked_pii = []
        notes = []
        
        # Check transcript
        text_pii = self.detector.detect_in_text(redacted_transcript.redacted_text)
        
        # Filter out labels like [CITY]
        real_leaks = [p for p in text_pii if not p["text"].startswith("[")]
        
        if real_leaks:
            leaked_pii.extend(real_leaks)
            notes.append(f"Found {len(real_leaks)} PII in redacted transcript")
        
        transcript_status = VerificationStatus.PASS if not real_leaks else VerificationStatus.FAIL
        
        # Check audio if requested
        audio_status = None
        if verify_audio and redacted_audio_path:
            audio_status = self._verify_audio(redacted_audio_path, leaked_pii, notes)
        
        # Determine overall status
        if transcript_status == VerificationStatus.FAIL:
            overall = VerificationStatus.FAIL
        elif audio_status == VerificationStatus.FAIL:
            overall = VerificationStatus.FAIL
        elif audio_status == VerificationStatus.REVIEW_REQUIRED:
            overall = VerificationStatus.REVIEW_REQUIRED
        elif notes:
            overall = VerificationStatus.PASS_WITH_NOTE
        else:
            overall = VerificationStatus.PASS
        
        return VerificationResult(
            overall_status=overall,
            transcript_status=transcript_status,
            audio_status=audio_status,
            leaked_pii=leaked_pii,
            notes=notes
        )

    def _verify_audio(self, audio_path: str, leaked_pii: list, notes: list) -> VerificationStatus:
        """Re-transcribe redacted audio and check for PII."""
        try:
            from .transcriber import Transcriber
            
            transcriber = Transcriber(model_size="base")
            transcript = transcriber.transcribe(audio_path)
            
            text = " ".join(seg["text"] for seg in transcript.segments)
            audio_pii = self.detector.detect_in_text(text)
            
            # Filter labels
            real_leaks = [p for p in audio_pii if not p["text"].startswith("[")]
            
            # Count by confidence
            high_conf = [p for p in real_leaks if not any(
                label in p["text"].lower() for label in ["bleep", "beep", "tone"]
            )]
            
            if len(high_conf) >= 3:
                notes.append(f"Audio verification: {len(high_conf)} potential PII leaks")
                return VerificationStatus.FAIL
            elif len(high_conf) >= 1:
                notes.append(f"Audio verification: {len(high_conf)} potential leaks (review)")
                return VerificationStatus.REVIEW_REQUIRED
            else:
                return VerificationStatus.PASS
                
        except Exception as e:
            notes.append(f"Audio verification failed: {e}")
            return VerificationStatus.REVIEW_REQUIRED
