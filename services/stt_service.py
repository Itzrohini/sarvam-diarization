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
MIN_SEGMENT_DURATION = 0.5      # Skip segments shorter than this (seconds)
MAX_CONCURRENT_CALLS = 4        # Max parallel Sarvam API calls at once


class STTService:

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
        # Semaphore limits how many Sarvam calls run at the same time.
        # Without this, 116 tasks fire simultaneously → rate-limit → failures.
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)

    async def initialize(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(1800, connect=15.0),
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
                    # Exponential backoff on rate limit
                    wait = RETRY_DELAY_SECONDS * (2 ** attempt)
                    logger.warning(
                        "Rate limited (429). Waiting %.1fs before retry %d/%d...",
                        wait, attempt, MAX_RETRIES
                    )
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

    # ──────────────────────────────────────────────────────────────────
    # NEW: Merge short / same-speaker adjacent segments before STT.
    # This reduces 116 segments → ~30-40, cutting API calls by ~65%.
    # ──────────────────────────────────────────────────────────────────
    def _merge_short_segments(
        self,
        segments: list[dict],
        min_duration: float = 1.5,
        max_gap: float = 0.4,
    ) -> list[dict]:
        """
        Merge adjacent segments from the same speaker when:
          - the segment is shorter than min_duration, OR
          - the gap between them is smaller than max_gap seconds.
        This avoids sending dozens of tiny audio clips to Sarvam.
        """
        if not segments:
            return segments

        merged = [dict(segments[0])]   # copy first segment

        for seg in segments[1:]:
            prev = merged[-1]
            duration = seg["end"] - seg["start"]
            gap = seg["start"] - prev["end"]

            same_speaker = seg["speaker"] == prev["speaker"]
            is_short = duration < min_duration
            is_close = gap < max_gap

            if same_speaker and (is_short or is_close):
                # Extend the previous segment to cover this one
                prev["end"] = seg["end"]
            else:
                merged.append(dict(seg))

        before = len(segments)
        after = len(merged)
        logger.info(
            "Segment merge: %d → %d segments (saved %d Sarvam API calls)",
            before, after, before - after,
        )
        return merged

    async def transcribe_segments(
        self,
        audio_array,
        sample_rate: int,
        segments: list[dict],
        language_code: str = "unknown",
    ) -> list[dict]:

        # ── Step 1: Merge short segments to reduce API calls ──────────
        segments = self._merge_short_segments(segments)
        logger.info(
            "Transcribing %d segments with Sarvam (max %d concurrent)...",
            len(segments), MAX_CONCURRENT_CALLS,
        )

        async def transcribe_one(seg: dict, idx: int) -> dict:
            start = seg["start"]
            end = seg["end"]
            duration = end - start

            # ── Step 2: Skip micro-segments (they always fail on Sarvam) ──
            if duration < MIN_SEGMENT_DURATION:
                logger.debug(
                    "Segment %d too short (%.3fs) — skipping.", idx, duration
                )
                return {**seg, "transcript": "", "language_detected": None}

            # ── Step 3: Semaphore — only MAX_CONCURRENT_CALLS run at once ──
            async with self._semaphore:

                # Long segment (>55s): split into chunks
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

                        if chunk_duration < MIN_SEGMENT_DURATION:
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
                                "  Chunk [%.1fs–%.1fs]: %s",
                                chunk_start, chunk_end, chunk_text[:50],
                            )
                        except Exception as e:
                            logger.error(
                                "  Chunk [%.1fs–%.1fs] failed: %s",
                                chunk_start, chunk_end, str(e),
                            )
                            all_transcripts.append("[chunk failed]")

                        chunk_start = chunk_end

                    return {
                        **seg,
                        "transcript": " ".join(all_transcripts),
                        "language_detected": detected_lang,
                    }

                # ── Normal segment under 55 seconds ──────────────────────
                audio_bytes = slice_audio_segment(
                    audio_array, sample_rate, start, end
                )

                try:
                    response = await self._transcribe_audio_bytes(
                        audio_bytes=audio_bytes,
                        language_code=language_code,
                        filename=f"segment_{idx:03d}.wav",
                    )
                    transcript, detected_lang = self._extract_transcript(response)

                    logger.info(
                        "Segment %d [%s | %.1fs–%.1fs]: %s",
                        idx,
                        seg.get("speaker_label", seg["speaker"]),
                        start, end,
                        transcript[:60],
                    )
                    return {
                        **seg,
                        "transcript": transcript,
                        "language_detected": detected_lang,
                    }

                except Exception as e:
                    logger.error(
                        "Segment %d [%.1fs–%.1fs] failed: %s", idx, start, end, str(e)
                    )
                    return {
                        **seg,
                        "transcript": "[transcription failed]",
                        "language_detected": None,
                    }

        # ── Step 4: Run all tasks — semaphore controls concurrency ────
        # asyncio.gather is fine here because the semaphore inside
        # transcribe_one ensures only MAX_CONCURRENT_CALLS run at once.
        tasks = [transcribe_one(seg, i) for i, seg in enumerate(segments)]
        results = await asyncio.gather(*tasks)
        return list(results)


_stt_service: Optional[STTService] = None


def get_stt_service() -> STTService:
    global _stt_service
    if _stt_service is None:
        _stt_service = STTService()
    return _stt_service