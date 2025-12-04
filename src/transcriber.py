"""
Transcription using faster-whisper.
Chose faster-whisper over WhisperX for better Apple Silicon support.
Provides word-level timestamps needed for audio redaction.
"""
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# Suppress duplicate library warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from faster_whisper import WhisperModel

from .config import (
    WHISPER_MODEL,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_BEAM_SIZE,
    WHISPER_LANGUAGE,
    WordTimestamp
)

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    """A segment of transcription with word timestamps."""
    text: str
    start: float
    end: float
    words: List[WordTimestamp]


@dataclass
class TranscriptionResult:
    """Complete transcription result for an audio file."""
    conversation_id: str
    audio_path: str
    audio_duration: float
    segments: List[TranscriptionSegment]
    language: str
    language_probability: float

    def get_all_words(self) -> List[WordTimestamp]:
        """Get all words from all segments."""
        all_words = []
        for segment in self.segments:
            all_words.extend(segment.words)
        return all_words

    def get_full_text(self) -> str:
        """Get the full transcript text."""
        return " ".join(seg.text.strip() for seg in self.segments)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "conversation_id": self.conversation_id,
            "audio_path": self.audio_path,
            "audio_duration": self.audio_duration,
            "language": self.language,
            "language_probability": self.language_probability,
            "segments": [
                {
                    "text": seg.text,
                    "start": seg.start,
                    "end": seg.end,
                    "words": [
                        {"word": w.word, "start": w.start, "end": w.end, "confidence": w.confidence}
                        for w in seg.words
                    ]
                }
                for seg in self.segments
            ]
        }


class Transcriber:
    """Transcribes audio files using faster-whisper with word timestamps."""

    def __init__(
        self,
        model_size: str = WHISPER_MODEL,
        device: str = WHISPER_DEVICE,
        compute_type: str = WHISPER_COMPUTE_TYPE
    ):
        """
        Initialize the transcriber.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3)
            device: Device to use (auto, cpu, cuda, mps)
            compute_type: Compute type (float16, float32, int8)
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model: Optional[WhisperModel] = None

    def _get_model(self) -> WhisperModel:
        """Lazy-load the Whisper model."""
        if self._model is None:
            logger.info(f"Loading Whisper model: {self.model_size}")

            # Determine device and compute type
            device = self.device
            compute_type = self.compute_type

            if device == "auto":
                # Check for available devices
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                    compute_type = "float16"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    # MPS available but faster-whisper works better with CPU
                    device = "cpu"
                    compute_type = "float32"
                else:
                    device = "cpu"
                    compute_type = "float32"

            logger.info(f"Using device: {device}, compute_type: {compute_type}")

            self._model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=compute_type
            )

        return self._model

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to the audio file (WAV, 16kHz, mono)

        Returns:
            TranscriptionResult with segments and word timestamps

        Raises:
            FileNotFoundError: If audio file doesn't exist
            Exception: For transcription errors
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        conversation_id = audio_path.stem
        logger.info(f"Transcribing: {conversation_id}")

        model = self._get_model()

        # Transcribe with word timestamps
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=WHISPER_LANGUAGE,
            beam_size=WHISPER_BEAM_SIZE,
            word_timestamps=True,
            vad_filter=True,  # Filter out non-speech
            vad_parameters=dict(
                min_silence_duration_ms=500,  # Minimum silence between speech
            )
        )

        # Convert to our data structures
        segments = []
        for segment in segments_iter:
            words = []
            if segment.words:
                for word_info in segment.words:
                    words.append(WordTimestamp(
                        word=word_info.word.strip(),
                        start=word_info.start,
                        end=word_info.end,
                        confidence=word_info.probability if hasattr(word_info, 'probability') else 1.0
                    ))

            segments.append(TranscriptionSegment(
                text=segment.text.strip(),
                start=segment.start,
                end=segment.end,
                words=words
            ))

        result = TranscriptionResult(
            conversation_id=conversation_id,
            audio_path=str(audio_path),
            audio_duration=info.duration,
            segments=segments,
            language=info.language,
            language_probability=info.language_probability
        )

        word_count = len(result.get_all_words())
        logger.info(
            f"Transcribed {conversation_id}: "
            f"{len(segments)} segments, {word_count} words, "
            f"{info.duration:.1f}s audio"
        )

        return result


def transcribe_audio(audio_path: str, model_size: str = "base") -> TranscriptionResult:
    """
    Convenience function to transcribe a single audio file.

    Args:
        audio_path: Path to audio file
        model_size: Whisper model size

    Returns:
        TranscriptionResult
    """
    transcriber = Transcriber(model_size=model_size)
    return transcriber.transcribe(audio_path)
