#!/usr/bin/env python3
"""
Download audio files from the HuggingFace dataset.
"""
import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "Appenlimited/1000h-us-english-smartphone-conversation"
OUTPUT_DIR = Path("data_exploration/audio")


def download_all_audio_files(max_files: int = 50):
    """Download all audio files from the dataset."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Listing files in {REPO_ID}...")
    try:
        all_files = list_repo_files(REPO_ID, repo_type="dataset")
    except Exception as e:
        print(f"Error listing files: {e}")
        return

    # Filter for audio files
    audio_files = [f for f in all_files if f.endswith('.wav')]
    print(f"Found {len(audio_files)} audio files")

    # Limit to max_files
    audio_files = audio_files[:max_files]
    print(f"Downloading {len(audio_files)} files...")

    downloaded = 0
    failed = []

    for i, file_path in enumerate(audio_files, 1):
        filename = Path(file_path).name
        local_path = OUTPUT_DIR / filename

        if local_path.exists():
            print(f"[{i}/{len(audio_files)}] Skipping {filename} (already exists)")
            downloaded += 1
            continue

        print(f"[{i}/{len(audio_files)}] Downloading {filename}...")
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=file_path,
                repo_type="dataset",
                local_dir=OUTPUT_DIR.parent,
                local_dir_use_symlinks=False
            )
            downloaded += 1
        except Exception as e:
            print(f"  Error: {e}")
            failed.append(filename)

    print(f"\nDownload complete: {downloaded}/{len(audio_files)} files")
    if failed:
        print(f"Failed: {failed}")


def download_transcripts():
    """Download human transcripts for WER comparison."""
    transcript_dir = Path("data_exploration/transcripts_human")
    transcript_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading human transcripts...")
    try:
        all_files = list_repo_files(REPO_ID, repo_type="dataset")
        transcript_files = [f for f in all_files if "TRANSCRIPTION" in f and f.endswith('.txt')]

        for i, file_path in enumerate(transcript_files[:40], 1):
            print(f"[{i}] Downloading {Path(file_path).name}...")
            try:
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=file_path,
                    repo_type="dataset",
                    local_dir=Path("data_exploration"),
                    local_dir_use_symlinks=False
                )
            except Exception as e:
                print(f"  Error: {e}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download dataset files")
    parser.add_argument("--max-files", type=int, default=40, help="Maximum files to download")
    parser.add_argument("--transcripts", action="store_true", help="Also download transcripts")

    args = parser.parse_args()

    download_all_audio_files(args.max_files)

    if args.transcripts:
        download_transcripts()
