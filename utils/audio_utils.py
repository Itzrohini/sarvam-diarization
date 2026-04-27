import io
import subprocess
import tempfile
import os
import numpy as np
import soundfile as sf
from core.logger import get_logger
from core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()

SUPPORTED_MIME_TYPES = {
    "audio/wav", "audio/wave", "audio/x-wav",
    "audio/mpeg", "audio/mp3",
    "audio/ogg", "audio/webm",
    "audio/flac", "audio/x-flac",
    "audio/mp4", "audio/m4a",
    "audio/x-m4a",
    "video/mp4",
    "video/mpeg",
    "video/webm",
    "video/quicktime",
}

# Hard cap: 500 MB — supports up to ~30 min WAV / ~3 hr MP3
# Your settings.max_audio_size_bytes overrides this if it is larger.
# If your .env still has the old 25 MB value, this fallback kicks in.
_FALLBACK_MAX_BYTES = 500 * 1024 * 1024   # 500 MB

# Sarvam requires at least 0.1 s of audio — we pad anything shorter.
MIN_SEGMENT_SAMPLES_AT_16K = 1600   # 0.1 s × 16000 Hz


def validate_audio_file(content_type: str, file_size: int) -> None:
    if content_type not in SUPPORTED_MIME_TYPES:
        raise ValueError(
            f"Unsupported format: '{content_type}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_MIME_TYPES))}"
        )

    # Use whichever limit is larger so a stale .env never blocks 30-min files.
    effective_max = max(
        getattr(settings, "max_audio_size_bytes", 0),
        _FALLBACK_MAX_BYTES,
    )
    if file_size > effective_max:
        max_mb = effective_max / (1024 * 1024)
        actual_mb = file_size / (1024 * 1024)
        raise ValueError(
            f"File too large: {actual_mb:.1f} MB. Max allowed: {max_mb:.0f} MB."
        )


def convert_to_mono_wav(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Convert any audio format to mono 16 kHz WAV using ffmpeg.
    Handles M4A, MP3, OGG, FLAC, WEBM — everything.
    """
    logger.debug("Converting audio using ffmpeg...")

    with tempfile.NamedTemporaryFile(suffix=".input_audio", delete=False) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in_path = tmp_in.name

    tmp_out_path = tmp_in_path + "_converted.wav"

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", tmp_in_path,
                "-ar", "16000",
                "-ac", "1",
                "-f", "wav",
                tmp_out_path,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error("ffmpeg error: %s", result.stderr)
            raise ValueError(
                f"ffmpeg failed to convert audio: {result.stderr[-200:]}"
            )

        audio_array, sample_rate = sf.read(tmp_out_path, dtype="float32")

        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        logger.info(
            "Audio converted: %.2f seconds at %d Hz",
            len(audio_array) / sample_rate,
            sample_rate,
        )
        return audio_array, sample_rate

    finally:
        if os.path.exists(tmp_in_path):
            os.unlink(tmp_in_path)
        if os.path.exists(tmp_out_path):
            os.unlink(tmp_out_path)


def slice_audio_segment(
    audio_array: np.ndarray,
    sample_rate: int,
    start_seconds: float,
    end_seconds: float,
) -> bytes:
    """
    Slice a segment from the full audio array and return WAV bytes.

    FIX: Sarvam rejects audio shorter than ~0.1 s with a 400 error.
    We pad anything below MIN_SEGMENT_SAMPLES_AT_16K with silence so
    the API never sees an illegally short clip. This prevents the
    cascade of '[transcription failed]' you saw after the 2-minute mark.
    """
    start_sample = max(0, int(start_seconds * sample_rate))
    end_sample = min(len(audio_array), int(end_seconds * sample_rate))
    segment = audio_array[start_sample:end_sample]

    # ── Pad if the slice is too short for Sarvam ─────────────────────
    min_samples = max(
        MIN_SEGMENT_SAMPLES_AT_16K,
        int(0.1 * sample_rate),   # 100 ms regardless of sample rate
    )
    if len(segment) < min_samples:
        padding = np.zeros(min_samples - len(segment), dtype=np.float32)
        segment = np.concatenate([segment, padding])
        logger.debug(
            "Padded short segment [%.3fs–%.3fs] to %.3fs",
            start_seconds, end_seconds, min_samples / sample_rate,
        )

    buffer = io.BytesIO()
    sf.write(buffer, segment, sample_rate, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return buffer.read()


def get_audio_duration(audio_array: np.ndarray, sample_rate: int) -> float:
    return len(audio_array) / sample_rate