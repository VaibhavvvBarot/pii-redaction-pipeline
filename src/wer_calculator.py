"""
WER (Word Error Rate) calculator.
Compares ASR output against human transcripts to measure accuracy.
WER = (Substitutions + Insertions + Deletions) / Reference Words
"""
import re
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class WERResult:
    """WER calculation result."""
    wer: float                  # Word Error Rate (0.0 - 1.0+)
    substitutions: int
    insertions: int
    deletions: int
    reference_words: int
    hypothesis_words: int
    aligned_pairs: List[Tuple[str, str]]  # (ref_word, hyp_word) pairs


def normalize_text(text: str) -> List[str]:
    """
    Normalize text for WER calculation.

    - Lowercase
    - Remove punctuation
    - Split into words
    """
    text = text.lower()
    # Remove punctuation except apostrophes in contractions
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    words = text.strip().split()
    return words


def levenshtein_alignment(ref: List[str], hyp: List[str]) -> Tuple[int, int, int, List]:
    """
    Compute Levenshtein distance with alignment tracking.

    Returns:
        (substitutions, insertions, deletions, alignment)
    """
    m, n = len(ref), len(hyp)

    # DP table: dp[i][j] = (distance, operation)
    # operation: 'M' (match), 'S' (sub), 'I' (insert), 'D' (delete)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    ops = [[None] * (n + 1) for _ in range(m + 1)]

    # Initialize
    for i in range(m + 1):
        dp[i][0] = i
        ops[i][0] = 'D'
    for j in range(n + 1):
        dp[0][j] = j
        ops[0][j] = 'I'
    ops[0][0] = None

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
                ops[i][j] = 'M'
            else:
                sub = dp[i-1][j-1] + 1
                ins = dp[i][j-1] + 1
                delete = dp[i-1][j] + 1

                if sub <= ins and sub <= delete:
                    dp[i][j] = sub
                    ops[i][j] = 'S'
                elif ins <= delete:
                    dp[i][j] = ins
                    ops[i][j] = 'I'
                else:
                    dp[i][j] = delete
                    ops[i][j] = 'D'

    # Backtrace to get alignment
    alignment = []
    i, j = m, n
    subs, ins, dels = 0, 0, 0

    while i > 0 or j > 0:
        op = ops[i][j]
        if op == 'M':
            alignment.append((ref[i-1], hyp[j-1]))
            i -= 1
            j -= 1
        elif op == 'S':
            alignment.append((ref[i-1], hyp[j-1]))
            subs += 1
            i -= 1
            j -= 1
        elif op == 'I':
            alignment.append(('', hyp[j-1]))
            ins += 1
            j -= 1
        elif op == 'D':
            alignment.append((ref[i-1], ''))
            dels += 1
            i -= 1

    alignment.reverse()
    return subs, ins, dels, alignment


def calculate_wer(reference: str, hypothesis: str) -> WERResult:
    """
    Calculate Word Error Rate between reference and hypothesis.

    Args:
        reference: Human transcription (ground truth)
        hypothesis: ASR transcription (model output)

    Returns:
        WERResult with detailed metrics
    """
    ref_words = normalize_text(reference)
    hyp_words = normalize_text(hypothesis)

    if len(ref_words) == 0:
        # Edge case: empty reference
        if len(hyp_words) == 0:
            return WERResult(
                wer=0.0,
                substitutions=0,
                insertions=0,
                deletions=0,
                reference_words=0,
                hypothesis_words=0,
                aligned_pairs=[]
            )
        else:
            return WERResult(
                wer=float('inf'),
                substitutions=0,
                insertions=len(hyp_words),
                deletions=0,
                reference_words=0,
                hypothesis_words=len(hyp_words),
                aligned_pairs=[('', w) for w in hyp_words]
            )

    subs, ins, dels, alignment = levenshtein_alignment(ref_words, hyp_words)

    wer = (subs + ins + dels) / len(ref_words)

    return WERResult(
        wer=wer,
        substitutions=subs,
        insertions=ins,
        deletions=dels,
        reference_words=len(ref_words),
        hypothesis_words=len(hyp_words),
        aligned_pairs=alignment
    )


def calculate_batch_wer(pairs: List[Tuple[str, str]]) -> Dict:
    """
    Calculate aggregate WER for multiple reference-hypothesis pairs.

    Args:
        pairs: List of (reference, hypothesis) tuples

    Returns:
        Dictionary with aggregate statistics
    """
    total_subs = 0
    total_ins = 0
    total_dels = 0
    total_ref_words = 0
    results = []

    for ref, hyp in pairs:
        result = calculate_wer(ref, hyp)
        results.append(result)
        total_subs += result.substitutions
        total_ins += result.insertions
        total_dels += result.deletions
        total_ref_words += result.reference_words

    if total_ref_words == 0:
        aggregate_wer = 0.0
    else:
        aggregate_wer = (total_subs + total_ins + total_dels) / total_ref_words

    individual_wers = [r.wer for r in results if r.reference_words > 0]

    return {
        "aggregate_wer": aggregate_wer,
        "mean_wer": sum(individual_wers) / len(individual_wers) if individual_wers else 0.0,
        "total_substitutions": total_subs,
        "total_insertions": total_ins,
        "total_deletions": total_dels,
        "total_reference_words": total_ref_words,
        "num_samples": len(pairs),
        "individual_results": results
    }


def print_wer_report(result: WERResult, max_errors: int = 20):
    """Print a human-readable WER report."""
    print(f"Word Error Rate: {result.wer:.2%}")
    print(f"  Substitutions: {result.substitutions}")
    print(f"  Insertions: {result.insertions}")
    print(f"  Deletions: {result.deletions}")
    print(f"  Reference words: {result.reference_words}")
    print(f"  Hypothesis words: {result.hypothesis_words}")

    # Show errors
    errors = [(r, h) for r, h in result.aligned_pairs if r != h][:max_errors]
    if errors:
        print(f"\nFirst {len(errors)} errors:")
        for ref, hyp in errors:
            if ref and hyp:
                print(f"  SUB: '{ref}' → '{hyp}'")
            elif ref:
                print(f"  DEL: '{ref}'")
            else:
                print(f"  INS: '{hyp}'")


if __name__ == "__main__":
    # Example usage
    ref = "I visited Houston Texas on Monday"
    hyp = "I visited Huston Texas on Munday"

    result = calculate_wer(ref, hyp)
    print_wer_report(result)
