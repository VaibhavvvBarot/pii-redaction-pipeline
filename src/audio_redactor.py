"""Audio redaction - replaces PII with bleep tones."""
import math
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from .config import (
    PIIMatch, MIN_BLEEP_DURATION_MS, BLEEP_FREQUENCY_HZ,
    BLEEP_AMPLITUDE, PADDING_BEFORE_MS, PADDING_AFTER_MS,
    SAMPLE_RATE, OUTPUT_AUDIO_FORMAT
)

logger = logging.getLogger(__name__)


@dataclass
class BleepRegion:
    """A region to bleep in audio."""
    start_time: float
    end_time: float
    bleep_duration: float
    pii_matches: List[PIIMatch]
