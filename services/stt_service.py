import asyncio
import httpx
from typing import Optional

from core.config import get_settings
from core.logger import get_logger
from utils.audio_utils import slice_audio_segment, convert_to_mono_wav

logger = get_logger(__name__)
settings = get_settings()

SARVAM_MAX_SEGMENT_SECONDS = 55.0
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 2.0


class STTService:

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    async def initialize(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=15.0),
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
            ),
        )
        logger.info("STT HTTP client ready.")

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            logger.info("STT HTTP client closed.")

    async def _transcribe_audio_bytes(
        self,
        audio_bytes: bytes,
        language_code: str = "unknown",
        filename: str = "segment.wav",
    ) -> dict:
        if self._client is None:
            raise RuntimeError("STTService not initialized.")

        headers = {
            "Authorization": f"Bearer {settings.sarvam_api_key}",
        }
        files = {
            "file": (filename, audio_bytes, "audio/wav"),
        }
        data = {
            "language_code": language_code,
            "model": "saaras:v3",
            "with_timestamps": "false",
        }

        last_error: Optional[Exception] = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.debug(
                    "Sarvam API call attempt %d/%d | lang: %s | size: %d bytes",
                    attempt, MAX_RETRIES, language_code, len(audio_bytes),
                )
                response = await self._client.post(
                    settings.sarvam_api_url,
                    headers=headers,
                    files=files,
                    data=data,
                )
                response.raise_for_status()
                result = response.json()
                logger.debug("Sarvam response: %s", result)
                return result

            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status == 401:
                    raise RuntimeError(
                        "Sarvam auth failed. Check SARVAM_API_KEY in .env"
                    ) from e
                if status == 429:
                    wait = RETRY_DELAY_SECONDS * (2 ** attempt)
                    logger.warning("Rate limited. Waiting %.1fs...", wait)
                    await asyncio.sleep(wait)
                    last_error = e
                    continue
                if status >= 500:
                    logger.warning("Sarvam server error %d, retrying...", status)
                    await asyncio.sleep(RETRY_DELAY_SECONDS * attempt)
                    last_error = e
                    continue
                raise RuntimeError(f"Sarvam error {status}: {e.response.text}") from e

            except httpx.RequestError as e:
                logger.warning("Network error attempt %d: %s", attempt, str(e))
                await asyncio.sleep(RETRY_DELAY_SECONDS * attempt)
                last_error = e

        raise RuntimeError(
            f"Sarvam API failed after {MAX_RETRIES} attempts. Last: {last_error}"
        )

    def _extract_transcript(self, response: dict) -> tuple[str, Optional[str]]:
        transcript = response.get("transcript", "").strip()
        language = response.get("language_code")
        return transcript, language

    async def transcribe_segments(
        self,
        audio_array,
        sample_rate: int,
        segments: list[dict],
        language_code: str = "unknown",
    ) -> list[dict]:
        logger.info("Transcribing %d segments with Sarvam...", len(segments))

        async def transcribe_one(seg: dict, idx: int) -> dict:
            start = seg["start"]
            end = seg["end"]
            duration = end - start

            if duration < 0.3:
                logger.debug("Segment %d too short (%.2fs), skipping.", idx, duration)
                return {**seg, "transcript": "", "language_detected": None}

            # If segment is longer than 55 seconds, split it into chunks
            # and transcribe each chunk separately, then join the results
            if duration > SARVAM_MAX_SEGMENT_SECONDS:
                logger.info(
                    "Segment %d is %.2fs — splitting into chunks.", idx, duration
                )
                all_transcripts = []
                detected_lang = None
                chunk_start = start

                while chunk_start < end:
                    chunk_end = min(chunk_start + SARVAM_MAX_SEGMENT_SECONDS, end)
                    chunk_duration = chunk_end - chunk_start

                    if chunk_duration < 0.3:
                        break

                    try:
                        audio_bytes = slice_audio_segment(
                            audio_array, sample_rate, chunk_start, chunk_end
                        )
                        response = await self._transcribe_audio_bytes(
                            audio_bytes=audio_bytes,
                            language_code=language_code,
                            filename=f"segment_{idx:03d}_chunk.wav",
                        )
                        chunk_text, lang = self._extract_transcript(response)
                        if chunk_text:
                            all_transcripts.append(chunk_text)
                        if lang and not detected_lang:
                            detected_lang = lang
                        logger.info(
                            "Chunk [%.1fs-%.1fs]: %s",
                            chunk_start, chunk_end, chunk_text[:40]
                        )
                    except Exception as e:
                        logger.error(
                            "Chunk [%.1fs-%.1fs] failed: %s",
                            chunk_start, chunk_end, str(e)
                        )
                        all_transcripts.append("[chunk failed]")

                    chunk_start = chunk_end

                return {
                    **seg,
                    "transcript": " ".join(all_transcripts),
                    "language_detected": detected_lang,
                }

            # Normal segment under 55 seconds
            effective_end = min(end, start + SARVAM_MAX_SEGMENT_SECONDS)
            audio_bytes = slice_audio_segment(
                audio_array, sample_rate, start, effective_end
            )

            try:
                response = await self._transcribe_audio_bytes(
                    audio_bytes=audio_bytes,
                    language_code=language_code,
                    filename=f"segment_{idx:03d}.wav",
                )
                transcript, detected_lang = self._extract_transcript(response)

                logger.info(
                    "Segment %d [%s | %.1fs-%.1fs]: %s",
                    idx,
                    seg.get("speaker_label", seg["speaker"]),
                    start, end,
                    transcript[:60],
                )
                return {**seg, "transcript": transcript, "language_detected": detected_lang}

            except Exception as e:
                logger.error("Segment %d transcription failed: %s", idx, str(e))
                return {**seg, "transcript": "[transcription failed]", "language_detected": None}

        tasks = [transcribe_one(seg, i) for i, seg in enumerate(segments)]
        results = await asyncio.gather(*tasks)
        return list(results)


_stt_service: Optional[STTService] = None


def get_stt_service() -> STTService:
    global _stt_service
    if _stt_service is None:
        _stt_service = STTService()
    return _stt_service