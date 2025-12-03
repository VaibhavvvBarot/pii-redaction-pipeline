"""Basic config for the pipeline."""
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

# Paths
OUTPUT_DIR = Path("output")

# Whisper settings
WHISPER_MODEL = "base"
