# PII De-Identification Pipeline - System Design

## Overview

This pipeline takes audio conversations, transcribes them, finds and redacts PII (Personally Identifiable Information), and outputs a clean de-identified dataset.

**PII Categories** (per the assignment):
- Days of the week (Monday, Tuesday, etc.)
- Months (January, February, etc.)
- Colors (red, blue, green, etc.)
- US Cities (Houston, New York, etc.)
- US States (Texas, California, etc.)

## Architecture
<img width="4988" height="4080" alt="image" src="https://github.com/user-attachments/assets/1be1957c-1e65-4495-ace5-1931a9a1e549" />


## Key Design Decisions

### 1. Transcription: faster-whisper over WhisperX

| Criteria | faster-whisper | WhisperX |
|----------|---------------|----------|
| Timestamp accuracy | ±50-100ms | ±30ms |
| MPS (Apple Silicon) | Works | Limited |
| CPU fallback | Good | Very slow |
| Setup complexity | Low | Higher |

I went with faster-whisper. The 150ms padding compensates for Whisper's general timestamp uncertainty (±50-100ms).

### 2. Bleep Duration Formula

```python
bleep_duration = max(400ms, word_end - word_start + 2 * padding)
```

A constant 400ms bleep wouldn't work because "San Francisco" takes about 800ms to say - you'd still hear "...cisco". On the other hand, a 50ms word would have an inaudible bleep. 400ms minimum is long enough to be clearly intentional.

### 3. Bleep vs Silence

| Method | Pros | Cons |
|--------|------|------|
| Silence | Natural | Sounds like audio dropout, length reveals word length |
| Bleep | Clearly intentional | Slightly jarring |

I went with a 1kHz bleep at 50% volume with 10ms fade in/out.

### 4. City/Color Collision

"Brownsville" contains "brown", "Greenville" contains "green" - if you match colors first, you get "[COLOR]sville" which is wrong.

Fix: Match cities before colors in the detection order.

### 5. "may" Context Rules

"may" can be a modal verb or a month name:

```
"You may proceed" - NOT a month (modal verb)
"In May we travel" - IS a month (preceded by preposition)
```

I look for month-like context patterns:
- Preceded by: in, during, last, next, this, of
- Followed by: date pattern (May 15th)

## Data Flow
<img width="2135" height="2100" alt="image" src="https://github.com/user-attachments/assets/146b0018-5555-4ed9-a3fb-a8e60160b560" />


## Error Handling

Each file is processed independently so one failure doesn't crash the whole batch:

```python
for audio_file in audio_files:
    try:
        result = process_conversation(audio_file)
        results.append(result)
    except Exception as e:
        results.append(Failure(file=audio_file, error=str(e)))
        continue  # Process next file
```

## Performance Characteristics

| Stage | Time (5 min audio) | Notes |
|-------|-------------------|-------|
| Transcription | ~60s (CPU) | ~6x real-time on base model |
| PII Detection | <1s | O(words × lexicon size) |
| Audio Redaction | <1s | NumPy vectorized operations |
| Verification | ~45s (CPU) | Re-transcription is costly |
| **Total** | **~2 min** | Per 5-minute audio file |

## Scalability

### Current (40 files)
- Sequential processing: ~80 minutes total
- Memory: ~2GB peak (Whisper model)

### For Production
- Add multiprocessing for parallel file processing
- GPU (CUDA) would give ~10x speedup
- Checkpoint progress for large batches
- Deploy on AWS Batch or GCP Cloud Run for scale

## Quality Metrics

1. **Transcription Accuracy**: WER (Word Error Rate) vs human transcripts
2. **Detection Recall**: % of actual PII detected
3. **Detection Precision**: % of detections that are actual PII
4. **Redaction Effectiveness**: % of audio PII that's inaudible after redaction
5. **Verification Rate**: % of files passing audio verification

## Output Specification

### Audio Format
- Format: FLAC (lossless compression)
- Sample rate: 16kHz (matches input)
- Channels: Mono
- Compression ratio: ~50% vs WAV

### Transcript Format
```json
{
  "conversation_id": "F2M2_USA_USA_010",
  "redacted_text": "I visited [CITY] on [DAY]",
  "segments": [...],
  "redactions": [
    {"original": "Houston", "replacement": "[CITY]", "start_time": 1.20}
  ],
  "pii_count": 51
}
```

## Security Notes

1. **No PII in logs** - only redacted text gets logged
2. **Original audio** - can be deleted after processing
3. **Output isolation** - redacted files go in separate directory
4. **Verification** - automated check before delivery

## Testing

1. **Unit Tests** - 43 tests covering edge cases
2. **Integration Tests** - end-to-end on sample files
3. **Verification Tests** - re-transcribe redacted audio and check for leaks

## Deployment

```bash
# Docker
docker build -t pii-pipeline .
docker run -v ./data:/data pii-pipeline --input /data/audio --output /data/output

# Local
pip install -r requirements.txt
python main.py --input audio/ --output output/ --model base
```
