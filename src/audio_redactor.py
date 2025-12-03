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


def generate_bleep_tone(
    duration_s: float,
    sample_rate: int = SAMPLE_RATE,
    frequency: float = BLEEP_FREQUENCY_HZ,
    amplitude: float = BLEEP_AMPLITUDE
) -> np.ndarray:
    """Generate a sine wave bleep with fade in/out."""
    n_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, n_samples, dtype=np.float32)
    
    bleep = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # Fade in/out to avoid clicks
    fade_samples = int(0.01 * sample_rate)
    if n_samples > 2 * fade_samples:
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        bleep[:fade_samples] *= fade_in
        bleep[-fade_samples:] *= fade_out
    
    return bleep.astype(np.float32)


def merge_overlapping_regions(regions: List[BleepRegion], min_gap_s: float = 0.1) -> List[BleepRegion]:
    """Merge overlapping or adjacent bleep regions."""
    if not regions:
        return []
    
    sorted_regions = sorted(regions, key=lambda r: r.start_time)
    merged = [sorted_regions[0]]
    
    for region in sorted_regions[1:]:
        last = merged[-1]
        if region.start_time <= last.end_time + min_gap_s:
            merged[-1] = BleepRegion(
                start_time=last.start_time,
                end_time=max(last.end_time, region.end_time),
                bleep_duration=0,
                pii_matches=last.pii_matches + region.pii_matches
            )
        else:
            merged.append(region)
    
    # Recalculate durations
    for region in merged:
        duration_ms = (region.end_time - region.start_time) * 1000
        region.bleep_duration = max(MIN_BLEEP_DURATION_MS, duration_ms) / 1000
    
    return merged


class AudioRedactor:
    """Redacts PII from audio with bleep tones."""
    
    def __init__(
        self,
        min_bleep_ms: int = MIN_BLEEP_DURATION_MS,
        bleep_freq: float = BLEEP_FREQUENCY_HZ,
        bleep_amp: float = BLEEP_AMPLITUDE,
        padding_before_ms: int = PADDING_BEFORE_MS,
        padding_after_ms: int = PADDING_AFTER_MS
    ):
        self.min_bleep_ms = min_bleep_ms
        self.bleep_freq = bleep_freq
        self.bleep_amp = bleep_amp
        self.padding_before_s = padding_before_ms / 1000
        self.padding_after_s = padding_after_ms / 1000
    
    def calculate_bleep_regions(
        self,
        pii_matches: List[PIIMatch],
        audio_duration: float
    ) -> List[BleepRegion]:
        """Calculate bleep regions from PII matches."""
        regions = []
        
        for match in pii_matches:
            start_time = max(0, match.start_time - self.padding_before_s)
            end_time = min(audio_duration, match.end_time + self.padding_after_s)
            
            duration_ms = (end_time - start_time) * 1000
            bleep_duration_s = max(self.min_bleep_ms, duration_ms) / 1000
            
            regions.append(BleepRegion(
                start_time=start_time,
                end_time=end_time,
                bleep_duration=bleep_duration_s,
                pii_matches=[match]
            ))
        
        return merge_overlapping_regions(regions)
