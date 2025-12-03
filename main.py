#!/usr/bin/env python3
"""
PII De-Identification Pipeline

Usage:
    python main.py --input <audio_dir> --output <output_dir> [options]
    python main.py --test  # Run on sample files
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Suppress duplicate library warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.pipeline import Pipeline, run_pipeline
from src.config import OUTPUT_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def find_audio_files(input_dir: str) -> list:
    """Find all audio files in a directory."""
    input_path = Path(input_dir)
    audio_extensions = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

    files = []
    for ext in audio_extensions:
        files.extend(input_path.glob(f"*{ext}"))
        files.extend(input_path.glob(f"**/*{ext}"))

    # Remove duplicates and sort
    files = sorted(set(files))
    return [str(f) for f in files]


def main():
    parser = argparse.ArgumentParser(
        description="PII De-Identification Pipeline for Audio/Transcript Data"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input directory containing audio files"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory for processed files"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip audio verification (faster but less thorough)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test on sample files in data_exploration/audio"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Process a single audio file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine input files
    if args.test:
        # Use sample files
        test_dir = Path(__file__).parent / "data_exploration" / "audio"
        if not test_dir.exists():
            logger.error(f"Test directory not found: {test_dir}")
            sys.exit(1)
        audio_files = find_audio_files(str(test_dir))
        if not audio_files:
            logger.error("No audio files found in test directory")
            sys.exit(1)
        logger.info(f"Test mode: found {len(audio_files)} audio files")

    elif args.file:
        # Single file
        if not Path(args.file).exists():
            logger.error(f"File not found: {args.file}")
            sys.exit(1)
        audio_files = [args.file]

    elif args.input:
        # Directory of files
        if not Path(args.input).exists():
            logger.error(f"Input directory not found: {args.input}")
            sys.exit(1)
        audio_files = find_audio_files(args.input)
        if not audio_files:
            logger.error("No audio files found in input directory")
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)

    logger.info(f"Processing {len(audio_files)} audio file(s)")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Whisper model: {args.model}")
    logger.info(f"Audio verification: {not args.no_verify}")

    # Run pipeline
    results = run_pipeline(
        audio_paths=audio_files,
        output_dir=args.output,
        whisper_model=args.model,
        verify_audio=not args.no_verify
    )

    # Summary
    successes = sum(1 for r in results if r.success)
    failures = len(results) - successes

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Processed: {len(results)} files")
    print(f"Successful: {successes}")
    print(f"Failed: {failures}")

    if failures > 0:
        print("\nFailed files:")
        for r in results:
            if not r.success:
                print(f"  - {r.conversation_id}: {r.error}")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
