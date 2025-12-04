# PII De-Identification Pipeline

Pipeline for detecting and redacting PII from audio conversations and their transcripts.

## Features

- Automatic transcription using faster-whisper with word-level timestamps
- PII detection for days, months, colors, US cities, US states
- Audio redaction with 1kHz bleep tones
- Text redaction with category labels ([CITY], [DAY], etc.)
- Verification by re-transcribing redacted audio
- Quality reports and metrics

## Quick Start

### Installation

```bash
# Clone and install
cd pii_pipeline
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

### Basic Usage

```bash
# Process a single file
python main.py --file path/to/audio.wav --output output/

# Process a directory
python main.py --input audio_dir/ --output output/ --model base

# Run on test data
python main.py --test
```

### Docker

```bash
# Build
docker build -t pii-pipeline .

# Run
docker run -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output \
  pii-pipeline --input /app/data --output /app/output --model base
```

## Output Structure

```
output/
├── audio/train/
│   └── {conversation_id}.flac    # Redacted audio (FLAC format)
├── transcripts_raw/train/
│   └── {conversation_id}.json    # Original transcription
├── transcripts_deid/train/
│   └── {conversation_id}.json    # Redacted transcription
├── metadata/
│   └── manifest.json             # Dataset manifest
└── qa/
    └── processing_report.json    # Quality report
```

## Redacted Transcript Format

```json
{
  "conversation_id": "F2M2_USA_USA_010",
  "original_text": "I visited Houston on Monday",
  "redacted_text": "I visited [CITY] on [DAY]",
  "pii_count": 2,
  "redactions": [
    {
      "original": "Houston",
      "replacement": "[CITY]",
      "category": "city",
      "start_time": 1.20,
      "end_time": 1.55
    }
  ]
}
```

## Configuration

Key settings in `src/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `WHISPER_MODEL` | `large-v3` | Whisper model size |
| `MIN_BLEEP_DURATION_MS` | `400` | Minimum bleep duration |
| `PADDING_BEFORE_MS` | `150` | Padding before PII word |
| `PADDING_AFTER_MS` | `150` | Padding after PII word |
| `BLEEP_FREQUENCY_HZ` | `1000` | Bleep tone frequency |

## Architecture

```
Audio → Transcribe → Detect PII → Redact Text → Redact Audio → Verify
         (faster-     (2-layer:     (replace      (bleep       (re-transcribe
          whisper)     exact +       with tags)    tones)        and check)
                       fuzzy)
```

See [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) for detailed architecture documentation.

## PII Categories

| Category | Examples | Replacement |
|----------|----------|-------------|
| Days | Monday, Tuesday | [DAY] |
| Months | January, February | [MONTH] |
| Colors | red, blue, green | [COLOR] |
| Cities | Houston, New York City | [CITY] |
| States | Texas, California | [STATE] |

## Quality Metrics

- Verification Status: PASS, PASS_WITH_NOTE, REVIEW_REQUIRED, FAIL
- WER: Word Error Rate vs human transcripts
- PII Detection Rate: % of audio files with detected PII

## Visualization

```bash
# View processing report
python scripts/visualize_results.py

# View specific transcript
python scripts/visualize_results.py --transcript output/transcripts_deid/train/FILE.json
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_pii_detector.py -v

# Run with coverage
pytest tests/ --cov=src
```

## Requirements

- Python 3.9+
- faster-whisper
- soundfile
- numpy
- torch

## License

Internal use only.
