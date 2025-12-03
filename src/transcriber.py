"""Transcription using faster-whisper."""
import logging
from typing import List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass 
class TranscriptionResult:
    """Result of transcribing an audio file."""
    segments: List[dict] = field(default_factory=list)
    audio_duration: float = 0.0
    language: str = "en"

    def get_all_words(self):
        """Get all words with timestamps."""
        from .config import WordTimestamp
        words = []
        for seg in self.segments:
            for w in seg.get("words", []):
                words.append(WordTimestamp(
                    word=w["word"],
                    start=w["start"],
                    end=w["end"]
                ))
        return words

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "segments": self.segments,
            "audio_duration": self.audio_duration,
            "language": self.language
        }


class Transcriber:
    """Transcriber using faster-whisper."""
    
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.model = None
    
    def _load_model(self):
        """Lazy load the model."""
        if self.model is None:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(self.model_size, compute_type="int8")
        return self.model

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe an audio file."""
        model = self._load_model()
        
        segments_data, info = model.transcribe(
            audio_path,
            word_timestamps=True,
            language="en"
        )
        
        segments = []
        for seg in segments_data:
            words = [{"word": w.word, "start": w.start, "end": w.end} 
                     for w in seg.words] if seg.words else []
            segments.append({
                "text": seg.text,
                "start": seg.start,
                "end": seg.end,
                "words": words
            })
        
        return TranscriptionResult(
            segments=segments,
            audio_duration=info.duration,
            language=info.language
        )
