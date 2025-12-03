#!/usr/bin/env python3
"""
Calculate WER (Word Error Rate) by comparing ASR output to human transcripts.
"""
import re
import json
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.wer_calculator import calculate_wer, print_wer_report, calculate_batch_wer


def clean_human_transcript(text: str) -> str:
    """Clean human transcript for WER calculation."""
    # Remove timestamps like [0.000]
    text = re.sub(r'\[\d+\.\d+\]', '', text)
    # Remove speaker tags
    text = re.sub(r'<Speaker_\d+>', '', text)
    # Remove noise tags like <no-speech>, <breath>, <cough>, etc.
    text = re.sub(r'<[^>]+>', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normalize whitespace and lowercase
    text = ' '.join(text.lower().split())
    return text


def clean_asr_transcript(data: dict) -> str:
    """Clean ASR transcript JSON for WER calculation."""
    if 'segments' in data:
        text = ' '.join(seg['text'] for seg in data['segments'])
    elif 'redacted_text' in data:
        text = data['redacted_text']
    else:
        text = data.get('text', '')

    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normalize whitespace and lowercase
    text = ' '.join(text.lower().split())
    return text


def find_matching_files(asr_dir: Path, human_dir: Path):
    """Find matching ASR and human transcript pairs."""
    pairs = []

    for asr_file in asr_dir.glob("*.json"):
        conv_id = asr_file.stem

        # Search for matching human transcript
        for human_file in human_dir.rglob(f"{conv_id}.txt"):
            pairs.append((asr_file, human_file))
            break

    return pairs


def calculate_single_wer(asr_path: str, human_path: str) -> dict:
    """Calculate WER for a single file pair."""
    # Load human transcript
    with open(human_path) as f:
        human_raw = f.read()
    human_clean = clean_human_transcript(human_raw)

    # Load ASR transcript
    with open(asr_path) as f:
        asr_data = json.load(f)
    asr_clean = clean_asr_transcript(asr_data)

    # Calculate WER
    result = calculate_wer(human_clean, asr_clean)

    return {
        "conversation_id": Path(asr_path).stem,
        "wer": result.wer,
        "substitutions": result.substitutions,
        "insertions": result.insertions,
        "deletions": result.deletions,
        "reference_words": result.reference_words,
        "hypothesis_words": result.hypothesis_words
    }


def main():
    parser = argparse.ArgumentParser(description="Calculate WER between ASR and human transcripts")
    parser.add_argument("--asr", type=str, help="Path to single ASR transcript JSON")
    parser.add_argument("--human", type=str, help="Path to matching human transcript TXT")
    parser.add_argument("--asr-dir", type=str, default="output/transcripts_raw/train",
                       help="Directory containing ASR transcripts")
    parser.add_argument("--human-dir", type=str,
                       default="data_exploration/USE_ASR003_Sample/TRANSCRIPTION_SEGMENTED_TO_SENTENCES",
                       help="Directory containing human transcripts")
    parser.add_argument("--output", type=str, help="Output JSON file for results")

    args = parser.parse_args()

    if args.asr and args.human:
        # Single file comparison
        result = calculate_single_wer(args.asr, args.human)
        print(f"\n{'='*60}")
        print(f"WER ANALYSIS: {result['conversation_id']}")
        print(f"{'='*60}")
        print(f"Word Error Rate: {result['wer']:.2%}")
        print(f"  Substitutions: {result['substitutions']}")
        print(f"  Insertions: {result['insertions']}")
        print(f"  Deletions: {result['deletions']}")
        print(f"  Reference words: {result['reference_words']}")
        print(f"  Hypothesis words: {result['hypothesis_words']}")

    else:
        # Batch comparison
        asr_dir = Path(args.asr_dir)
        human_dir = Path(args.human_dir)

        if not asr_dir.exists():
            print(f"ASR directory not found: {asr_dir}")
            return

        pairs = find_matching_files(asr_dir, human_dir)
        print(f"Found {len(pairs)} matching file pairs")

        if not pairs:
            print("No matching files found")
            return

        results = []
        for asr_path, human_path in pairs:
            print(f"Processing {asr_path.stem}...")
            result = calculate_single_wer(str(asr_path), str(human_path))
            results.append(result)
            print(f"  WER: {result['wer']:.2%}")

        # Calculate aggregate stats
        total_subs = sum(r['substitutions'] for r in results)
        total_ins = sum(r['insertions'] for r in results)
        total_dels = sum(r['deletions'] for r in results)
        total_ref = sum(r['reference_words'] for r in results)

        aggregate_wer = (total_subs + total_ins + total_dels) / total_ref if total_ref > 0 else 0
        mean_wer = sum(r['wer'] for r in results) / len(results)

        print(f"\n{'='*60}")
        print("AGGREGATE WER STATISTICS")
        print(f"{'='*60}")
        print(f"Files analyzed: {len(results)}")
        print(f"Aggregate WER: {aggregate_wer:.2%}")
        print(f"Mean WER: {mean_wer:.2%}")
        print(f"Total reference words: {total_ref}")
        print(f"Total errors: {total_subs + total_ins + total_dels}")

        # Save results
        if args.output:
            output_data = {
                "aggregate_wer": aggregate_wer,
                "mean_wer": mean_wer,
                "files_analyzed": len(results),
                "total_reference_words": total_ref,
                "individual_results": results
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
