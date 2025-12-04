"""
Verification - checks that PII was actually redacted.
Text: re-scan redacted transcript for remaining PII
Audio: re-transcribe redacted audio and look for leaks
"""
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

from .config import (
    VERIFY_PASS_THRESHOLD,
    VERIFY_REVIEW_THRESHOLD,
    VERIFY_FAIL_THRESHOLD,
    FUZZY_MIN_CONFIDENCE
)
from .pii_detector import PIIDetector
from .transcriber import Transcriber, TranscriptionResult
from .text_redactor import RedactedTranscript

logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    """Verification status levels."""
    PASS = "PASS"                    # No PII found
    PASS_WITH_NOTE = "PASS_WITH_NOTE"  # Low-confidence matches only (likely noise)
    REVIEW_REQUIRED = "REVIEW_REQUIRED"  # Some high-confidence matches
    FAIL = "FAIL"                    # Multiple PII instances leaked


@dataclass
class VerificationResult:
    """Result of verification."""
    conversation_id: str
    text_status: VerificationStatus
    audio_status: Optional[VerificationStatus]
    text_pii_found: List[Dict]
    audio_pii_found: List[Dict]
    notes: List[str]

    @property
    def overall_status(self) -> VerificationStatus:
        """Get overall verification status."""
        if self.audio_status is None:
            return self.text_status

        # Return worst status
        statuses = [self.text_status, self.audio_status]
        if VerificationStatus.FAIL in statuses:
            return VerificationStatus.FAIL
        if VerificationStatus.REVIEW_REQUIRED in statuses:
            return VerificationStatus.REVIEW_REQUIRED
        if VerificationStatus.PASS_WITH_NOTE in statuses:
            return VerificationStatus.PASS_WITH_NOTE
        return VerificationStatus.PASS

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "conversation_id": self.conversation_id,
            "overall_status": self.overall_status.value,
            "text_status": self.text_status.value,
            "audio_status": self.audio_status.value if self.audio_status else None,
            "text_pii_found": self.text_pii_found,
            "audio_pii_found": self.audio_pii_found,
            "notes": self.notes
        }


class Verifier:
    """Verifies PII redaction in text and audio."""

    def __init__(self, transcriber: Optional[Transcriber] = None):
        """
        Initialize the verifier.

        Args:
            transcriber: Transcriber instance for audio verification
        """
        self.detector = PIIDetector()
        self.transcriber = transcriber

    def _determine_status(
        self,
        pii_found: List[Dict],
        source: str
    ) -> tuple[VerificationStatus, List[str]]:
        """
        Determine verification status based on PII found.

        Args:
            pii_found: List of PII matches found
            source: "text" or "audio"

        Returns:
            Tuple of (status, notes)
        """
        notes = []
        count = len(pii_found)

        if count == 0:
            return VerificationStatus.PASS, []

        # Check if all matches are low confidence (likely ASR noise)
        high_conf_matches = [
            p for p in pii_found
            if p.get("confidence", 1.0) >= FUZZY_MIN_CONFIDENCE
        ]

        if count <= VERIFY_REVIEW_THRESHOLD:
            if len(high_conf_matches) == 0:
                notes.append(
                    f"{source.capitalize()}: {count} low-confidence match(es), "
                    "likely ASR noise or false positive"
                )
                return VerificationStatus.PASS_WITH_NOTE, notes
            else:
                notes.append(
                    f"{source.capitalize()}: {count} PII match(es) found, "
                    "manual review recommended"
                )
                return VerificationStatus.REVIEW_REQUIRED, notes

        # count > VERIFY_FAIL_THRESHOLD
        notes.append(
            f"{source.capitalize()}: {count} PII instances leaked - "
            "check bleep alignment"
        )
        return VerificationStatus.FAIL, notes

    def verify_text(self, redacted_transcript: RedactedTranscript) -> tuple:
        """
        Verify that redacted transcript contains no PII.

        Args:
            redacted_transcript: The redacted transcript to verify

        Returns:
            Tuple of (status, pii_found, notes)
        """
        logger.info(f"Verifying text redaction for {redacted_transcript.conversation_id}")

        # Scan redacted text for remaining PII
        pii_found = self.detector.detect_in_text(redacted_transcript.redacted_text)

        # Filter out our own labels (they contain category names which might match)
        # e.g., "[CITY]" should not match as a city
        pii_found = [
            p for p in pii_found
            if not p["text"].startswith("[") and not p["text"].endswith("]")
        ]

        status, notes = self._determine_status(pii_found, "text")

        logger.info(f"Text verification: {status.value}, {len(pii_found)} PII found")
        return status, pii_found, notes

    def verify_audio(
        self,
        redacted_audio_path: str,
        conversation_id: str
    ) -> tuple:
        """
        Verify that redacted audio contains no audible PII.

        Re-transcribes the redacted audio and scans for PII.

        Args:
            redacted_audio_path: Path to redacted audio file
            conversation_id: Conversation ID for logging

        Returns:
            Tuple of (status, pii_found, notes)
        """
        logger.info(f"Verifying audio redaction for {conversation_id}")

        if self.transcriber is None:
            # Use a small model for verification (faster)
            self.transcriber = Transcriber(model_size="base")

        # Re-transcribe redacted audio
        try:
            transcript = self.transcriber.transcribe(redacted_audio_path)
        except Exception as e:
            logger.error(f"Failed to re-transcribe audio: {e}")
            return (
                VerificationStatus.REVIEW_REQUIRED,
                [],
                [f"Audio verification failed: {e}"]
            )

        # Detect PII in re-transcription
        pii_matches = self.detector.detect(transcript)

        # Convert to dicts with confidence
        pii_found = [
            {
                "text": m.text,
                "category": m.category,
                "confidence": m.confidence,
                "is_fuzzy": m.is_fuzzy,
                "start_time": m.start_time,
                "end_time": m.end_time
            }
            for m in pii_matches
        ]

        status, notes = self._determine_status(pii_found, "audio")

        logger.info(f"Audio verification: {status.value}, {len(pii_found)} PII found")
        return status, pii_found, notes

    def verify(
        self,
        redacted_transcript: RedactedTranscript,
        redacted_audio_path: Optional[str] = None,
        verify_audio: bool = True
    ) -> VerificationResult:
        """
        Perform full verification of redaction.

        Args:
            redacted_transcript: Redacted transcript to verify
            redacted_audio_path: Path to redacted audio (optional)
            verify_audio: Whether to verify audio (set False to skip)

        Returns:
            VerificationResult
        """
        conversation_id = redacted_transcript.conversation_id
        notes = []

        # Verify text
        text_status, text_pii, text_notes = self.verify_text(redacted_transcript)
        notes.extend(text_notes)

        # Verify audio if requested and path provided
        audio_status = None
        audio_pii = []

        if verify_audio and redacted_audio_path:
            audio_status, audio_pii, audio_notes = self.verify_audio(
                redacted_audio_path, conversation_id
            )
            notes.extend(audio_notes)

        return VerificationResult(
            conversation_id=conversation_id,
            text_status=text_status,
            audio_status=audio_status,
            text_pii_found=text_pii,
            audio_pii_found=audio_pii,
            notes=notes
        )


def verify_redaction(
    redacted_transcript: RedactedTranscript,
    redacted_audio_path: Optional[str] = None,
    verify_audio: bool = True
) -> VerificationResult:
    """
    Convenience function to verify redaction.

    Args:
        redacted_transcript: Redacted transcript
        redacted_audio_path: Path to redacted audio
        verify_audio: Whether to verify audio

    Returns:
        VerificationResult
    """
    verifier = Verifier()
    return verifier.verify(redacted_transcript, redacted_audio_path, verify_audio)
