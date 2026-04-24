from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class SupportedLanguage(str, Enum):
    AUTO      = "unknown"
    KANNADA   = "kn-IN"
    HINDI     = "hi-IN"
    TAMIL     = "ta-IN"
    TELUGU    = "te-IN"
    MALAYALAM = "ml-IN"
    ENGLISH   = "en-IN"


class SpeakerSegment(BaseModel):
    speaker: str
    speaker_label: str
    start: float
    end: float
    start_formatted: str = ""   # HH:MM:SS format
    end_formatted: str = ""     # HH:MM:SS format
    transcript: str = ""
    language_detected: Optional[str] = None


class ProcessAudioResponse(BaseModel):
    success: bool
    total_speakers: int
    duration_seconds: float
    segments: list[SpeakerSegment]
    full_transcript: str
    processing_time_seconds: float


class ErrorResponse(BaseModel):
    success: bool = False
    error_code: str
    message: str
    detail: Optional[str] = None


