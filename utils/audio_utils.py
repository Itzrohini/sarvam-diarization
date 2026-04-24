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


def validate_audio_file(content_type: str, file_size: int) -> None:
    if content_type not in SUPPORTED_MIME_TYPES:
        raise ValueError(
            f"Unsupported format: '{content_type}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_MIME_TYPES))}"
        )
    if file_size > settings.max_audio_size_bytes:
        max_mb = settings.max_audio_size_bytes / (1024 * 1024)
        actual_mb = file_size / (1024 * 1024)
        raise ValueError(f"File too large: {actual_mb:.1f} MB. Max: {max_mb:.0f} MB.")


def convert_to_mono_wav(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Convert any audio format to mono 16kHz WAV using ffmpeg directly.
    This bypasses librosa and speechbrain completely — much more reliable.
    """
    logger.debug("Converting audio using ffmpeg...")

    # Write input bytes to a temp file
    with tempfile.NamedTemporaryFile(suffix=".input_audio", delete=False) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in_path = tmp_in.name

    # Output will be a temp WAV file
    tmp_out_path = tmp_in_path + "_converted.wav"

    try:
        # Use ffmpeg directly to convert to 16kHz mono WAV
        # This handles M4A, MP3, OGG, FLAC, WEBM — everything
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",               # overwrite output if exists
                "-i", tmp_in_path,  # input file
                "-ar", "16000",     # sample rate 16000 Hz
                "-ac", "1",         # mono (1 channel)
                "-f", "wav",        # output format WAV
                tmp_out_path,       # output file
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error("ffmpeg error: %s", result.stderr)
            raise ValueError(f"ffmpeg failed to convert audio: {result.stderr[-200:]}")

        # Read the converted WAV file with soundfile
        audio_array, sample_rate = sf.read(tmp_out_path, dtype="float32")

        # If somehow still stereo, mix down to mono
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        logger.info(
            "Audio converted: %.2f seconds at %d Hz",
            len(audio_array) / sample_rate,
            sample_rate,
        )
        return audio_array, sample_rate

    finally:
        # Always clean up temp files
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
    start_sample = max(0, int(start_seconds * sample_rate))
    end_sample = min(len(audio_array), int(end_seconds * sample_rate))
    segment = audio_array[start_sample:end_sample]

    buffer = io.BytesIO()
    sf.write(buffer, segment, sample_rate, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return buffer.read()


def get_audio_duration(audio_array: np.ndarray, sample_rate: int) -> float:
    return len(audio_array) / sample_rate