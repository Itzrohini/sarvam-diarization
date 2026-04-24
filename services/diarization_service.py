import asyncio
import numpy as np
from typing import Optional
from pyannote.audio import Pipeline
import torch

from core.config import get_settings
from core.logger import get_logger
from utils.audio_utils import convert_to_mono_wav, get_audio_duration

logger = get_logger(__name__)
settings = get_settings()


class DiarizationService:

    def __init__(self):
        self._pipeline: Optional[Pipeline] = None
        self._device = self._select_device()

    def _select_device(self) -> torch.device:
        if torch.cuda.is_available():
            logger.info("GPU found — using CUDA for Pyannote.")
            return torch.device("cuda")
        logger.info("No GPU — using CPU for Pyannote.")
        return torch.device("cpu")

    def initialize(self) -> None:
        if self._pipeline is not None:
            return

        logger.info("Loading Pyannote model from HuggingFace...")
        try:
            # Login to HuggingFace first using the global login
            from huggingface_hub import login
            login(token=settings.huggingface_token, add_to_git_credential=False)
            logger.info("HuggingFace login successful.")

            import torch
            torch.set_num_threads(8)
            logger.info("Pytorch using%d CPU threads", torch.get_num_threads())


            # Now load WITHOUT passing token — login() handles auth globally
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1"
            )
            self._pipeline = self._pipeline.to(self._device)
            logger.info("Pyannote pipeline ready on %s.", self._device)

        except Exception as e:
            logger.error("Failed to load Pyannote: %s", str(e))
            raise RuntimeError(
                f"Pyannote model failed to load. Error: {str(e)}"
            ) from e

    def _assign_speaker_labels(
        self,
        speaker_ids: list[str],
        custom_labels: list[str] = None
    ) -> dict[str, str]:
        default_labels = [
            "Shop Owner",
            "Customer",
            "Person 3",
            "Person 4",
            "Person 5",
        ]

        labels_to_use = custom_labels if custom_labels else default_labels

        result = {}
        for i, sid in enumerate(sorted(set(speaker_ids))):
            if i < len(labels_to_use):
                result[sid] = labels_to_use[i]
            else:
                result[sid] = f"Speaker {i + 1}"
        return result

    async def diarize(
        self,
        audio_bytes: bytes,
        num_speakers: Optional[int] = None,
        min_speakers: int = 1,
        max_speakers: int = 10,
        custom_labels: list[str] = None,
    ) -> list[dict]:
        if self._pipeline is None:
            raise RuntimeError("DiarizationService not initialized.")

        logger.info("Starting diarization. Audio: %d bytes", len(audio_bytes))

        audio_array, sample_rate = convert_to_mono_wav(audio_bytes)
        duration = get_audio_duration(audio_array, sample_rate)
        logger.info("Audio duration: %.2f seconds", duration)

        waveform_tensor = torch.from_numpy(audio_array).unsqueeze(0)
        pyannote_input = {"waveform": waveform_tensor, "sample_rate": sample_rate}

        pipeline_kwargs = {}
        if num_speakers is not None:
            pipeline_kwargs["num_speakers"] = num_speakers
        else:
            pipeline_kwargs["min_speakers"] = min_speakers
            pipeline_kwargs["max_speakers"] = max_speakers

        loop = asyncio.get_event_loop()
        diarization_result = await loop.run_in_executor(
            None,
            lambda: self._pipeline(pyannote_input, **pipeline_kwargs),
        )

        raw_segments = []
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            raw_segments.append({
                "speaker": speaker,
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
            })

        if not raw_segments:
            logger.warning("No speech detected in audio.")
            return []

        speaker_ids = [s["speaker"] for s in raw_segments]
        label_map = self._assign_speaker_labels(speaker_ids, custom_labels)

        segments = []
        for seg in raw_segments:
            segments.append({
                "speaker": seg["speaker"],
                "speaker_label": label_map[seg["speaker"]],
                "start": seg["start"],
                "end": seg["end"],
            })

        logger.info(
            "Diarization done: %d segments, %d speakers.",
            len(segments),
            len(set(s["speaker"] for s in segments)),
        )
        return segments


_diarization_service: Optional[DiarizationService] = None


def get_diarization_service() -> DiarizationService:
    global _diarization_service
    if _diarization_service is None:
        _diarization_service = DiarizationService()
    return _diarization_service