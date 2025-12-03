"""Config for the PII pipeline."""
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

OUTPUT_DIR = Path("output")
WHISPER_MODEL = "base"
SAMPLE_RATE = 16000

# Audio redaction
MIN_BLEEP_DURATION_MS = 400
BLEEP_FREQUENCY_HZ = 1000
BLEEP_AMPLITUDE = 0.5
PADDING_BEFORE_MS = 150
PADDING_AFTER_MS = 150
OUTPUT_AUDIO_FORMAT = "flac"
