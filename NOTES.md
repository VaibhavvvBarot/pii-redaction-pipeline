# Assumptions, Tradeoffs, and TODOs

## Assumptions

1. Input audio is 16kHz mono WAV (matches the dataset)
2. All conversations are in English
3. 2-person conversations
4. Only detecting the 5 specified categories (days, months, colors, cities, states) - not real PII

## Tradeoffs

### Transcription
| Decision | Tradeoff |
|----------|----------|
| faster-whisper over WhisperX | WhisperX has better timestamps (±30ms vs ±100ms) but has compatibility issues with Apple MPS. I added 150ms padding to compensate. |
| large-v3 model default | Best accuracy. Tested base model at 6.65% WER, small at 5.30% WER. large-v3 expected ~3-4% WER per Whisper benchmarks. Slower but most accurate. |

### PII Detection
| Decision | Tradeoff |
|----------|----------|
| 2-layer detection (exact + fuzzy) | Fuzzy matching catches ASR errors ("Huston" to Houston) but risks false positives. I added strict constraints (min 5 chars, blacklist). |
| Cities matched before colors | Prevents "Brownsville" from becoming "[COLOR]sville" but adds ordering complexity. |
| "may" context rules | Avoids false positives on modal verb but might miss some edge cases. |

### Audio Redaction
| Decision | Tradeoff |
|----------|----------|
| Bleep tone (not silence) | More jarring to listen to, but clearly signals intentional redaction and doesn't reveal word length. |
| 150ms padding | May clip adjacent words slightly, but ensures PII is fully covered given timestamp uncertainty. |

## TODOs (Future Improvements)

### If I Had More Time
1. GPU acceleration - would significantly speed up large-v3 model
2. Speaker diarization - add speaker labels to output
3. Parallel processing - process multiple files concurrently
4. Parquet output - currently using JSON for metadata; Parquet would be better at scale

### Production Considerations
1. Monitoring - add metrics/logging for production
2. Retry logic - handle transient failures
3. Incremental processing - skip already-processed files
4. Cloud deployment - AWS Batch or GCP Cloud Run

## Known Limitations

1. CPU-only tested - GPU would be ~10x faster
2. English only - would need language detection for multilingual
3. No real PII detection - only the 5 fake PII categories
4. Timestamp accuracy - ±100ms means occasional slight audio leakage (caught by verification)
