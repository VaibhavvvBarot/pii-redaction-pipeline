#!/usr/bin/env python3
"""
Simple visualization for pipeline results.
Shows PII counts, before/after comparison, and processing times.
Uses ASCII charts to avoid matplotlib dependency.
"""
import json
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List

# Simple ASCII visualization (no matplotlib dependency)


def create_bar_chart(data: Dict[str, int], title: str, max_width: int = 50) -> str:
    """Create an ASCII bar chart."""
    if not data:
        return f"{title}\n  (no data)"

    lines = [title, "=" * len(title)]
    max_value = max(data.values()) if data.values() else 1

    for label, value in sorted(data.items(), key=lambda x: -x[1]):
        bar_length = int((value / max_value) * max_width) if max_value > 0 else 0
        bar = "█" * bar_length
        lines.append(f"  {label:12} │ {bar} {value}")

    return "\n".join(lines)


def create_comparison_table(before: Dict, after: Dict, title: str) -> str:
    """Create a before/after comparison table."""
    lines = [title, "=" * len(title)]
    lines.append(f"  {'Metric':<25} │ {'Before':>10} │ {'After':>10} │ {'Δ':>8}")
    lines.append("  " + "-" * 60)

    all_keys = set(before.keys()) | set(after.keys())
    for key in sorted(all_keys):
        b = before.get(key, 0)
        a = after.get(key, 0)
        delta = a - b
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        lines.append(f"  {key:<25} │ {b:>10} │ {a:>10} │ {delta_str:>8}")

    return "\n".join(lines)


def visualize_processing_report(report_path: str) -> str:
    """Visualize the processing report."""
    with open(report_path) as f:
        report = json.load(f)

    output = []

    # Header
    output.append("\n" + "=" * 70)
    output.append("       PII DE-IDENTIFICATION PIPELINE - RESULTS SUMMARY")
    output.append("=" * 70 + "\n")

    # Summary stats
    summary = report.get("summary", {})
    output.append("PROCESSING SUMMARY")
    output.append("-" * 40)
    output.append(f"  Total files processed: {summary.get('total_processed', 0)}")
    output.append(f"  Successful:            {summary.get('successful', 0)}")
    output.append(f"  Failed:                {summary.get('failed', 0)}")
    output.append(f"  Total PII redacted:    {summary.get('total_pii_redacted', 0)}")
    output.append("")

    # Verification status
    status = report.get("verification_status", {})
    output.append(create_bar_chart(status, "VERIFICATION STATUS"))
    output.append("")

    # Processing times
    times = report.get("processing_times", {})
    if times:
        output.append("PROCESSING TIMES")
        output.append("-" * 40)
        total_time = sum(times.values())
        for conv_id, time_s in sorted(times.items()):
            bar_len = int((time_s / max(times.values())) * 30)
            bar = "▓" * bar_len
            output.append(f"  {conv_id[:30]:<30} │ {bar} {time_s:.1f}s")
        output.append(f"\n  Total processing time: {total_time:.1f}s")
        output.append(f"  Average per file:      {total_time/len(times):.1f}s")
        output.append("")

    # Failures
    failures = report.get("failures", [])
    if failures:
        output.append("FAILURES")
        output.append("-" * 40)
        for fail in failures:
            output.append(f"  ✗ {fail['conversation_id']}")
            output.append(f"    Stage: {fail['stage']}")
            output.append(f"    Error: {fail['error'][:60]}...")
        output.append("")

    return "\n".join(output)


def visualize_transcript_deid(transcript_path: str) -> str:
    """Visualize a single de-identified transcript."""
    with open(transcript_path) as f:
        data = json.load(f)

    output = []

    conv_id = data.get("conversation_id", "Unknown")
    pii_count = data.get("pii_count", 0)

    output.append(f"\nCONVERSATION: {conv_id}")
    output.append("=" * 50)

    # PII by category
    redactions = data.get("redactions", [])
    by_category = Counter(r["category"] for r in redactions)

    output.append(create_bar_chart(dict(by_category), f"PII DETECTED ({pii_count} total)"))
    output.append("")

    # Sample redactions
    output.append("SAMPLE REDACTIONS")
    output.append("-" * 40)
    for r in redactions[:10]:
        output.append(f"  '{r['original']:<15}' → {r['replacement']}")
    if len(redactions) > 10:
        output.append(f"  ... and {len(redactions) - 10} more")
    output.append("")

    # Text comparison
    original = data.get("original_text", "")[:200]
    redacted = data.get("redacted_text", "")[:200]

    output.append("TEXT COMPARISON (first 200 chars)")
    output.append("-" * 40)
    output.append(f"  Original: {original}...")
    output.append(f"  Redacted: {redacted}...")

    return "\n".join(output)


def main():
    """Main visualization entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize PII pipeline results")
    parser.add_argument(
        "--report",
        type=str,
        default="output/qa/processing_report.json",
        help="Path to processing report JSON"
    )
    parser.add_argument(
        "--transcript",
        type=str,
        help="Path to a specific de-identified transcript to visualize"
    )
    parser.add_argument(
        "--all-transcripts",
        action="store_true",
        help="Visualize all transcripts in output/transcripts_deid/"
    )

    args = parser.parse_args()

    # Visualize processing report
    if Path(args.report).exists():
        print(visualize_processing_report(args.report))

    # Visualize specific transcript
    if args.transcript and Path(args.transcript).exists():
        print(visualize_transcript_deid(args.transcript))

    # Visualize all transcripts
    if args.all_transcripts:
        deid_dir = Path("output/transcripts_deid/train")
        if deid_dir.exists():
            for transcript in sorted(deid_dir.glob("*.json")):
                print(visualize_transcript_deid(str(transcript)))
                print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
