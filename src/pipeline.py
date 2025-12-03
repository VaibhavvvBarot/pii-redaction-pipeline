"""Main pipeline - runs all steps end to end."""
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .config import ProcessingResult, OUTPUT_DIR, WHISPER_MODEL, OUTPUT_AUDIO_FORMAT
from .transcriber import Transcriber, TranscriptionResult
from .pii_detector import PIIDetector, PIIMatch
from .text_redactor import TextRedactor, RedactedTranscript
from .audio_redactor import AudioRedactor, BleepRegion
from .verifier import Verifier, VerificationResult, VerificationStatus

logger = logging.getLogger(__name__)


@dataclass
class ConversationOutput:
    """Output for a processed conversation."""
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
    """Main PII de-identification pipeline."""
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        whisper_model: str = WHISPER_MODEL,
        verify_audio: bool = True,
        save_outputs: bool = True
    ):
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.whisper_model = whisper_model
        self.verify_audio = verify_audio
        self.save_outputs = save_outputs
        
        # Init components
        self.transcriber = Transcriber(model_size=whisper_model)
        self.detector = PIIDetector()
        self.text_redactor = TextRedactor()
        self.audio_redactor = AudioRedactor()
        self.verifier = Verifier()
        
        if save_outputs:
            self._create_output_dirs()
    
    def _create_output_dirs(self):
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
        """Process a single conversation through the full pipeline."""
        start_time = time.time()
        audio_path = Path(audio_path)
        conversation_id = audio_path.stem
        
        logger.info(f"Processing: {conversation_id}")
        output = ConversationOutput(conversation_id=conversation_id, success=False)
        
        try:
            # Step 1: Transcribe
            logger.info(f"[1/5] Transcribing...")
            output.stage = "transcription"
            transcript = self.transcriber.transcribe(str(audio_path))
            output.transcript_raw = transcript
            
            # Step 2: Detect PII
            logger.info(f"[2/5] Detecting PII...")
            output.stage = "detection"
            pii_matches = self.detector.detect(transcript)
            output.pii_matches = pii_matches
            logger.info(f"Found {len(pii_matches)} PII instances")
            
            # Step 3: Redact text
            logger.info(f"[3/5] Redacting transcript...")
            output.stage = "text_redaction"
            redacted = self.text_redactor.redact(transcript, pii_matches)
            output.transcript_redacted = redacted
            
            # Step 4: Redact audio
            logger.info(f"[4/5] Redacting audio...")
            output.stage = "audio_redaction"
            audio_out = self.output_dir / "audio" / "train" / f"{conversation_id}.{OUTPUT_AUDIO_FORMAT}" if self.save_outputs else None
            redacted_path, regions = self.audio_redactor.redact(str(audio_path), pii_matches, str(audio_out) if audio_out else None)
            output.redacted_audio_path = redacted_path
            output.bleep_regions = regions
            
            # Step 5: Verify
            logger.info(f"[5/5] Verifying...")
            output.stage = "verification"
            verification = self.verifier.verify(redacted, redacted_path if self.verify_audio else None, verify_audio=self.verify_audio)
            output.verification = verification
            
            if self.save_outputs:
                self._save_outputs(output)
            
            output.success = True
            output.stage = None
            output.error = None
            logger.info(f"Done: {len(pii_matches)} PII, status={verification.overall_status.value}")
            
        except Exception as e:
            output.error = str(e)
            logger.error(f"Failed at {output.stage}: {e}")
        finally:
            output.processing_time_s = time.time() - start_time
        
        return output

    def _save_outputs(self, output: ConversationOutput):
        """Save outputs to disk."""
        conv_id = output.conversation_id
        
        if output.transcript_raw:
            path = self.output_dir / "transcripts_raw" / "train" / f"{conv_id}.json"
            with open(path, "w") as f:
                json.dump(output.transcript_raw.to_dict(), f, indent=2)
        
        if output.transcript_redacted:
            path = self.output_dir / "transcripts_deid" / "train" / f"{conv_id}.json"
            with open(path, "w") as f:
                json.dump(output.transcript_redacted.to_dict(), f, indent=2)

    def process_batch(self, audio_paths: List[str], continue_on_error: bool = True) -> List[ConversationOutput]:
        """Process multiple conversations."""
        results = []
        total = len(audio_paths)
        
        logger.info(f"Processing batch of {total} conversations")
        
        for i, audio_path in enumerate(audio_paths, 1):
            logger.info(f"[{i}/{total}] {Path(audio_path).stem}")
            
            try:
                result = self.process_conversation(audio_path)
                results.append(result)
            except Exception as e:
                if continue_on_error:
                    logger.error(f"Failed: {e}")
                    results.append(ConversationOutput(
                        conversation_id=Path(audio_path).stem,
                        success=False, error=str(e), stage="unknown"
                    ))
                else:
                    raise
        
        self._generate_report(results)
        self._generate_metadata_manifest(results)
        return results
