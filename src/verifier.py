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
