"""
Audio redaction - replaces PII segments with bleep tones.
Bleep duration is max(400ms, word_duration + padding) so longer words are fully covered.
"""
import math
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from .config import (
    PIIMatch,
    MIN_BLEEP_DURATION_MS,
    BLEEP_FREQUENCY_HZ,
    BLEEP_AMPLITUDE,
    PADDING_BEFORE_MS,
    PADDING_AFTER_MS,
    SAMPLE_RATE,
    OUTPUT_AUDIO_FORMAT
)

logger = logging.getLogger(__name__)


@dataclass
class BleepRegion:
    """A region to be bleeped in the audio."""
    start_time: float      # Start time in seconds (with padding)
    end_time: float        # End time in seconds (with padding)
    bleep_duration: float  # Duration of bleep in seconds
    pii_matches: List[PIIMatch]  # PII matches in this region


def generate_bleep_tone(
    duration_s: float,
    sample_rate: int = SAMPLE_RATE,
    frequency: float = BLEEP_FREQUENCY_HZ,
    amplitude: float = BLEEP_AMPLITUDE
) -> np.ndarray:
    """Generate a sine wave bleep tone with fade in/out to avoid clicks."""
    n_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, n_samples, dtype=np.float32)

    # Generate sine wave
    bleep = amplitude * np.sin(2 * np.pi * frequency * t)

    # Apply fade in/out to avoid clicks (10ms each)
    fade_samples = int(0.01 * sample_rate)
    if n_samples > 2 * fade_samples:
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        bleep[:fade_samples] *= fade_in
        bleep[-fade_samples:] *= fade_out

    return bleep.astype(np.float32)


def merge_overlapping_regions(
    regions: List[BleepRegion],
    min_gap_s: float = 0.1
) -> List[BleepRegion]:
    """Merge overlapping or adjacent bleep regions into single bleeps."""
    if not regions:
        return []

    # Sort by start time
    sorted_regions = sorted(regions, key=lambda r: r.start_time)

    merged = [sorted_regions[0]]

    for region in sorted_regions[1:]:
        last = merged[-1]

        # Check if regions overlap or are very close
        if region.start_time <= last.end_time + min_gap_s:
            # Merge: extend end time and combine PII matches
            merged[-1] = BleepRegion(
                start_time=last.start_time,
                end_time=max(last.end_time, region.end_time),
                bleep_duration=0,  # Will recalculate
                pii_matches=last.pii_matches + region.pii_matches
            )
        else:
            merged.append(region)

    # Recalculate bleep durations for merged regions
    for region in merged:
        actual_duration_ms = (region.end_time - region.start_time) * 1000
        region.bleep_duration = max(MIN_BLEEP_DURATION_MS, actual_duration_ms) / 1000

    return merged


class AudioRedactor:
    """
    Redacts PII from audio with bleep tones.
    Uses max(400ms, word_duration + padding) so long phrases are fully covered.
    Adjacent PII gets merged into one continuous bleep.
    """

    def __init__(
        self,
        min_bleep_ms: int = MIN_BLEEP_DURATION_MS,
        bleep_freq: float = BLEEP_FREQUENCY_HZ,
        bleep_amp: float = BLEEP_AMPLITUDE,
        padding_before_ms: int = PADDING_BEFORE_MS,
        padding_after_ms: int = PADDING_AFTER_MS
    ):
        """Initialize with configurable bleep settings."""
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
        """Calculate bleep regions from PII matches. Uses max(400ms, actual + padding)."""
        regions = []

        for match in pii_matches:
            # Calculate padded region
            start_time = max(0, match.start_time - self.padding_before_s)
            end_time = min(audio_duration, match.end_time + self.padding_after_s)

            # Calculate actual duration with padding
            actual_duration_ms = (end_time - start_time) * 1000

            # Bleep duration: max(min_bleep, actual_duration)
            bleep_duration_ms = max(self.min_bleep_ms, actual_duration_ms)
            bleep_duration_s = bleep_duration_ms / 1000

            regions.append(BleepRegion(
                start_time=start_time,
                end_time=end_time,
                bleep_duration=bleep_duration_s,
                pii_matches=[match]
            ))

        # Merge overlapping regions
        return merge_overlapping_regions(regions)

    def redact(
        self,
        audio_path: str,
        pii_matches: List[PIIMatch],
        output_path: Optional[str] = None
    ) -> Tuple[str, List[BleepRegion]]:
        """Redact PII from audio file. Returns (output_path, bleep_regions)."""
        import soundfile as sf

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Read audio
        audio_data, sample_rate = sf.read(str(audio_path), dtype='float32')

        # Handle stereo -> mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        audio_duration = len(audio_data) / sample_rate
        logger.info(f"Loaded audio: {audio_duration:.1f}s, {sample_rate}Hz")

        # Calculate bleep regions
        regions = self.calculate_bleep_regions(pii_matches, audio_duration)
        logger.info(f"Calculated {len(regions)} bleep regions from {len(pii_matches)} PII matches")

        # Apply bleeps
        redacted_audio = audio_data.copy()

        for region in regions:
            # Calculate sample positions
            start_sample = int(region.start_time * sample_rate)
            end_sample = int(region.end_time * sample_rate)

            # Generate bleep tone
            bleep = generate_bleep_tone(
                region.bleep_duration,
                sample_rate,
                self.bleep_freq,
                self.bleep_amp
            )

            # Replace audio segment with bleep
            # If bleep is longer than segment, we extend; if shorter, we truncate
            segment_length = end_sample - start_sample
            bleep_length = len(bleep)

            if bleep_length >= segment_length:
                # Bleep covers the segment (possibly longer)
                redacted_audio[start_sample:end_sample] = bleep[:segment_length]
            else:
                # Bleep is shorter (shouldn't happen with our formula, but handle it)
                redacted_audio[start_sample:start_sample + bleep_length] = bleep
                # Silence the rest
                redacted_audio[start_sample + bleep_length:end_sample] = 0

            logger.debug(
                f"Applied bleep at {region.start_time:.2f}s-{region.end_time:.2f}s "
                f"(duration: {region.bleep_duration:.3f}s)"
            )

        # Determine output path
        if output_path is None:
            output_path = audio_path.parent / f"{audio_path.stem}_redacted.{OUTPUT_AUDIO_FORMAT}"
        else:
            output_path = Path(output_path)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write output
        sf.write(str(output_path), redacted_audio, sample_rate)
        logger.info(f"Saved redacted audio to: {output_path}")

        return str(output_path), regions


def redact_audio(
    audio_path: str,
    pii_matches: List[PIIMatch],
    output_path: Optional[str] = None
) -> Tuple[str, List[BleepRegion]]:
    """Convenience function to redact PII from audio."""
    redactor = AudioRedactor()
    return redactor.redact(audio_path, pii_matches, output_path)
