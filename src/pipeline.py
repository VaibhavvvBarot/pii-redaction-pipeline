"""
Main pipeline that runs all the steps:
1. Transcription (faster-whisper)
2. PII Detection (exact + fuzzy)
3. Text Redaction
4. Audio Redaction
5. Verification

Each file is processed independently so one failure doesn't stop the batch.
"""
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .config import (
    ProcessingResult,
    OUTPUT_DIR,
    WHISPER_MODEL,
    OUTPUT_AUDIO_FORMAT
)
from .transcriber import Transcriber, TranscriptionResult
from .pii_detector import PIIDetector, PIIMatch
from .text_redactor import TextRedactor, RedactedTranscript
from .audio_redactor import AudioRedactor, BleepRegion
from .verifier import Verifier, VerificationResult, VerificationStatus

logger = logging.getLogger(__name__)


@dataclass
class ConversationOutput:
    """Complete output for a processed conversation."""
    conversation_id: str
    success: bool
    error: Optional[str] = None
    stage: Optional[str] = None
    transcript_raw: Optional[TranscriptionResult] = None
    transcript_redacted: Optional[RedactedTranscript] = None
    pii_matches: List[PIIMatch] = field(default_factory=list)
    bleep_regions: List[BleepRegion] = field(default_factory=list)
    redacted_audio_path: Optional[str] = None
    verification: Optional[VerificationResult] = None
    processing_time_s: float = 0.0


class Pipeline:
    """
    Main PII de-identification pipeline.

    Processes audio files through:
    1. Transcription with word timestamps
    2. PII detection (days, months, colors, cities, states)
    3. Text redaction
    4. Audio redaction (bleep tones)
    5. Verification
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        whisper_model: str = WHISPER_MODEL,
        verify_audio: bool = True,
        save_outputs: bool = True
    ):
        """
        Initialize the pipeline.

        Args:
            output_dir: Directory for output files
            whisper_model: Whisper model size
            verify_audio: Whether to re-transcribe and verify audio
            save_outputs: Whether to save files to disk
        """
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.whisper_model = whisper_model
        self.verify_audio = verify_audio
        self.save_outputs = save_outputs

        # Initialize components
        self.transcriber = Transcriber(model_size=whisper_model)
        self.detector = PIIDetector()
        self.text_redactor = TextRedactor()
        self.audio_redactor = AudioRedactor()
        self.verifier = Verifier()

        # Create output directories
        if save_outputs:
            self._create_output_dirs()

    def _create_output_dirs(self):
        """Create output directory structure."""
        dirs = [
            self.output_dir / "audio" / "train",
            self.output_dir / "transcripts_raw" / "train",
            self.output_dir / "transcripts_deid" / "train",
            self.output_dir / "metadata",
            self.output_dir / "qa"
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def process_conversation(self, audio_path: str) -> ConversationOutput:
        """
        Process a single conversation through the full pipeline.

        Args:
            audio_path: Path to audio file

        Returns:
            ConversationOutput with all results
        """
        start_time = time.time()
        audio_path = Path(audio_path)
        conversation_id = audio_path.stem

        logger.info(f"Processing conversation: {conversation_id}")

        output = ConversationOutput(
            conversation_id=conversation_id,
            success=False
        )

        try:
            # Step 1: Transcription
            logger.info(f"[1/5] Transcribing {conversation_id}...")
            output.stage = "transcription"
            transcript = self.transcriber.transcribe(str(audio_path))
            output.transcript_raw = transcript

            # Step 2: PII Detection
            logger.info(f"[2/5] Detecting PII in {conversation_id}...")
            output.stage = "detection"
            pii_matches = self.detector.detect(transcript)
            output.pii_matches = pii_matches

            logger.info(f"Found {len(pii_matches)} PII instances")

            # Step 3: Text Redaction
            logger.info(f"[3/5] Redacting transcript for {conversation_id}...")
            output.stage = "text_redaction"
            redacted_transcript = self.text_redactor.redact(transcript, pii_matches)
            output.transcript_redacted = redacted_transcript

            # Step 4: Audio Redaction
            logger.info(f"[4/5] Redacting audio for {conversation_id}...")
            output.stage = "audio_redaction"

            if self.save_outputs:
                audio_output_path = (
                    self.output_dir / "audio" / "train" /
                    f"{conversation_id}.{OUTPUT_AUDIO_FORMAT}"
                )
            else:
                audio_output_path = None

            redacted_audio_path, bleep_regions = self.audio_redactor.redact(
                str(audio_path),
                pii_matches,
                str(audio_output_path) if audio_output_path else None
            )
            output.redacted_audio_path = redacted_audio_path
            output.bleep_regions = bleep_regions

            # Step 5: Verification
            logger.info(f"[5/5] Verifying redaction for {conversation_id}...")
            output.stage = "verification"
            verification = self.verifier.verify(
                redacted_transcript,
                redacted_audio_path if self.verify_audio else None,
                verify_audio=self.verify_audio
            )
            output.verification = verification

            # Save outputs if requested
            if self.save_outputs:
                self._save_outputs(output)

            # Mark success
            output.success = True
            output.stage = None
            output.error = None

            logger.info(
                f"Completed {conversation_id}: "
                f"{len(pii_matches)} PII redacted, "
                f"verification={verification.overall_status.value}"
            )

        except FileNotFoundError as e:
            output.error = f"File not found: {e}"
            logger.error(f"Failed {conversation_id} at {output.stage}: {output.error}")

        except Exception as e:
            output.error = str(e)
            logger.error(f"Failed {conversation_id} at {output.stage}: {output.error}")

        finally:
            output.processing_time_s = time.time() - start_time

        return output

    def _save_outputs(self, output: ConversationOutput):
        """Save outputs to disk."""
        conv_id = output.conversation_id

        # Save raw transcript
        if output.transcript_raw:
            raw_path = self.output_dir / "transcripts_raw" / "train" / f"{conv_id}.json"
            with open(raw_path, "w") as f:
                json.dump(output.transcript_raw.to_dict(), f, indent=2)

        # Save redacted transcript
        if output.transcript_redacted:
            deid_path = self.output_dir / "transcripts_deid" / "train" / f"{conv_id}.json"
            with open(deid_path, "w") as f:
                json.dump(output.transcript_redacted.to_dict(), f, indent=2)

        logger.debug(f"Saved outputs for {conv_id}")

    def process_batch(
        self,
        audio_paths: List[str],
        continue_on_error: bool = True
    ) -> List[ConversationOutput]:
        """
        Process multiple conversations.

        Args:
            audio_paths: List of audio file paths
            continue_on_error: Continue processing if one file fails

        Returns:
            List of ConversationOutput objects
        """
        results = []
        total = len(audio_paths)

        logger.info(f"Processing batch of {total} conversations")

        for i, audio_path in enumerate(audio_paths, 1):
            logger.info(f"[{i}/{total}] Processing {Path(audio_path).stem}")

            try:
                result = self.process_conversation(audio_path)
                results.append(result)

            except Exception as e:
                if continue_on_error:
                    logger.error(f"Failed to process {audio_path}: {e}")
                    results.append(ConversationOutput(
                        conversation_id=Path(audio_path).stem,
                        success=False,
                        error=str(e),
                        stage="unknown"
                    ))
                else:
                    raise

        # Generate summary report and metadata
        self._generate_report(results)
        self._generate_metadata_manifest(results)

        return results

    def _generate_metadata_manifest(self, results: List[ConversationOutput]):
        """Generate metadata manifest in the format requested by customer."""
        if not self.save_outputs:
            return

        manifest_rows = []
        for r in results:
            if not r.success:
                continue

            # Build metadata row matching the requested schema
            row = {
                "conversation_id": r.conversation_id,
                "duration_sec": r.transcript_raw.audio_duration if r.transcript_raw else 0,
                "num_speakers": 2,  # Dataset is 2-person conversations
                "sample_rate": 16000,
                "has_pii": len(r.pii_matches) > 0,
                "pii_count": len(r.pii_matches),
                "deid_version": datetime.now().strftime("%Y-%m-%d_v1"),
                "qa_status": r.verification.overall_status.value if r.verification else "pending"
            }
            manifest_rows.append(row)

        # Save as JSON (parquet requires pyarrow, keep it simple)
        manifest_path = self.output_dir / "metadata" / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest_rows, f, indent=2)

        logger.info(f"Saved metadata manifest to {manifest_path}")

    def _generate_report(self, results: List[ConversationOutput]):
        """Generate a summary report of processing."""
        successes = [r for r in results if r.success]
        failures = [r for r in results if not r.success]

        # Count verification statuses
        status_counts = {s.value: 0 for s in VerificationStatus}
        for r in successes:
            if r.verification:
                status_counts[r.verification.overall_status.value] += 1

        # Total PII counts and duration
        total_pii = sum(len(r.pii_matches) for r in successes)
        total_duration = sum(
            r.transcript_raw.audio_duration if r.transcript_raw else 0
            for r in successes
        )

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_conversations": len(results),
                "successful": len(successes),
                "failed": len(failures),
                "total_duration_sec": round(total_duration, 1),
                "total_duration_min": round(total_duration / 60, 1),
                "total_pii_redacted": total_pii
            },
            "verification_status": status_counts,
            "failures": [
                {
                    "conversation_id": r.conversation_id,
                    "stage": r.stage,
                    "error": r.error
                }
                for r in failures
            ],
            "processing_times": {
                r.conversation_id: r.processing_time_s
                for r in results
            }
        }

        # Save report
        if self.save_outputs:
            report_path = self.output_dir / "qa" / "processing_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

            logger.info(f"Saved processing report to {report_path}")

        # Log summary
        logger.info(
            f"\n{'='*60}\n"
            f"PROCESSING COMPLETE\n"
            f"{'='*60}\n"
            f"Total: {len(results)}\n"
            f"Success: {len(successes)}\n"
            f"Failed: {len(failures)}\n"
            f"PII Redacted: {total_pii}\n"
            f"Verification: {status_counts}\n"
            f"{'='*60}"
        )


def run_pipeline(
    audio_paths: List[str],
    output_dir: Optional[str] = None,
    whisper_model: str = "base",
    verify_audio: bool = True
) -> List[ConversationOutput]:
    """
    Convenience function to run the pipeline.

    Args:
        audio_paths: List of audio file paths
        output_dir: Output directory
        whisper_model: Whisper model size
        verify_audio: Whether to verify audio redaction

    Returns:
        List of ConversationOutput objects
    """
    pipeline = Pipeline(
        output_dir=output_dir,
        whisper_model=whisper_model,
        verify_audio=verify_audio
    )
    return pipeline.process_batch(audio_paths)
