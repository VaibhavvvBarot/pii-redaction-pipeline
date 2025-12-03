"""
Tests for audio redaction.
Checks the bleep duration formula, multi-word handling, region merging,
and tone generation.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_redactor import (
    AudioRedactor,
    generate_bleep_tone,
    merge_overlapping_regions,
    BleepRegion
)
from src.config import PIIMatch, MIN_BLEEP_DURATION_MS, PADDING_BEFORE_MS, PADDING_AFTER_MS


class TestBleepDuration:
    """Test that bleep duration uses max(400ms, duration + padding)."""

    @pytest.fixture
    def redactor(self):
        return AudioRedactor()

    def test_short_word_uses_minimum(self, redactor):
        """Short word (200ms) should use minimum 400ms bleep."""
        pii = [PIIMatch(
            text="red",
            category="color",
            start_time=1.0,
            end_time=1.2,  # 200ms word
            confidence=1.0,
            word_indices=[0]
        )]

        regions = redactor.calculate_bleep_regions(pii, audio_duration=10.0)
        assert len(regions) == 1

        # With 150ms padding on each side: 200 + 300 = 500ms
        # max(400, 500) = 500ms
        expected_min = MIN_BLEEP_DURATION_MS / 1000
        assert regions[0].bleep_duration >= expected_min

    def test_long_word_uses_actual_duration(self, redactor):
        """Long word (800ms) should use actual duration + padding, not just 400ms."""
        pii = [PIIMatch(
            text="San Francisco",
            category="city",
            start_time=1.0,
            end_time=1.8,  # 800ms phrase
            confidence=1.0,
            word_indices=[0, 1]
        )]

        regions = redactor.calculate_bleep_regions(pii, audio_duration=10.0)
        assert len(regions) == 1

        # 800ms + 300ms padding = 1100ms
        # Should be > 400ms minimum
        assert regions[0].bleep_duration > 0.4
        # Should be approximately 1.1s (with padding)
        actual_with_padding = 0.8 + (PADDING_BEFORE_MS + PADDING_AFTER_MS) / 1000
        assert regions[0].bleep_duration >= actual_with_padding * 0.9  # Allow 10% tolerance


class TestMultiWordBleep:
    """Test continuous bleep for multi-word PII."""

    @pytest.fixture
    def redactor(self):
        return AudioRedactor()

    def test_multi_word_continuous_region(self, redactor):
        """Multi-word PII should create one continuous bleep region."""
        pii = [PIIMatch(
            text="New York City",
            category="city",
            start_time=1.0,
            end_time=1.8,
            confidence=1.0,
            word_indices=[0, 1, 2]
        )]

        regions = redactor.calculate_bleep_regions(pii, audio_duration=10.0)

        # Should be exactly 1 region, not 3
        assert len(regions) == 1
        # Region should span from first word start to last word end (with padding)
        assert regions[0].start_time < 1.0  # Has padding before
        assert regions[0].end_time > 1.8    # Has padding after


class TestRegionMerging:
    """Test that adjacent PII regions are merged."""

    def test_overlapping_regions_merged(self):
        """Overlapping regions should merge into one."""
        regions = [
            BleepRegion(start_time=1.0, end_time=1.5, bleep_duration=0.5, pii_matches=[]),
            BleepRegion(start_time=1.4, end_time=2.0, bleep_duration=0.6, pii_matches=[])
        ]

        merged = merge_overlapping_regions(regions)
        assert len(merged) == 1
        assert merged[0].start_time == 1.0
        assert merged[0].end_time == 2.0

    def test_adjacent_regions_merged(self):
        """Regions within 100ms of each other should merge."""
        regions = [
            BleepRegion(start_time=1.0, end_time=1.5, bleep_duration=0.5, pii_matches=[]),
            BleepRegion(start_time=1.55, end_time=2.0, bleep_duration=0.45, pii_matches=[])
        ]

        merged = merge_overlapping_regions(regions, min_gap_s=0.1)
        assert len(merged) == 1

    def test_distant_regions_not_merged(self):
        """Regions with large gap should stay separate."""
        regions = [
            BleepRegion(start_time=1.0, end_time=1.5, bleep_duration=0.5, pii_matches=[]),
            BleepRegion(start_time=3.0, end_time=3.5, bleep_duration=0.5, pii_matches=[])
        ]

        merged = merge_overlapping_regions(regions, min_gap_s=0.1)
        assert len(merged) == 2


class TestBleepToneGeneration:
    """Test bleep tone audio generation."""

    def test_bleep_duration(self):
        """Generated bleep should have correct duration."""
        duration = 0.4  # 400ms
        sample_rate = 16000
        bleep = generate_bleep_tone(duration, sample_rate)

        expected_samples = int(duration * sample_rate)
        assert len(bleep) == expected_samples

    def test_bleep_amplitude(self):
        """Bleep should respect amplitude setting."""
        bleep = generate_bleep_tone(0.4, 16000, amplitude=0.5)

        # Max amplitude should be approximately 0.5
        assert np.max(np.abs(bleep)) <= 0.55  # Allow small tolerance
        assert np.max(np.abs(bleep)) >= 0.45

    def test_bleep_is_float32(self):
        """Bleep should be float32 for audio processing."""
        bleep = generate_bleep_tone(0.4, 16000)
        assert bleep.dtype == np.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
