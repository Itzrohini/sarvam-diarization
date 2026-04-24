import time
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, Annotated
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy.orm import Session
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request
from database.database import create_tables, get_db
from database import crud
from typing import Optional



from core.config import get_settings
from core.logger import get_logger, setup_logging
from models.schemas import (
    ProcessAudioResponse,
    SpeakerSegment,
    ErrorResponse,
    SupportedLanguage,
)
from services.diarization_service import get_diarization_service, DiarizationService
from services.stt_service import get_stt_service, STTService
from utils.audio_utils import validate_audio_file, convert_to_mono_wav, get_audio_duration

setup_logging()
logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 50)
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    logger.info("=" * 50)

    diarization_svc = get_diarization_service()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, diarization_svc.initialize)
    logger.info("Diarization service ready.")
    
    create_tables()
    logger.info("Database Ready.")

    stt_svc = get_stt_service()
    await stt_svc.initialize()
    logger.info("STT service ready.")

    logger.info("All services up. Accepting requests.")
    yield

    logger.info("Shutting down...")
    await stt_svc.close()
    logger.info("Shutdown complete.")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Multi-speaker STT for Indian languages.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)
# Serve the frontend folder as static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse("frontend/index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled error on %s: %s", request.url.path, str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_code="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred.",
            detail=str(exc) if settings.debug else None,
        ).model_dump(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error_code=f"HTTP_{exc.status_code}",
            message=exc.detail,
        ).model_dump(),
    )


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
    }


@app.post("/process-audio", response_model=ProcessAudioResponse, tags=["STT"])
async def process_audio(
    file: Annotated[UploadFile, File(description="Audio file")],
    language_code: Annotated[
        SupportedLanguage,
        Form(description="Language code or 'unknown' for auto-detect")
    ] = SupportedLanguage.AUTO,
    num_speakers: Annotated[
        Optional[int],
        Form(description="Exact number of speakers if known")
    ] = None,
    speaker_labels: Annotated[
        Optional[str],
        Form(description="Comma separated speaker names. Example: Teacher,Student or Doctor,Patient")
    ] = None,
    db: Session = Depends(get_db),
    diarization_svc: DiarizationService = Depends(get_diarization_service),
    stt_svc: STTService = Depends(get_stt_service),
) -> ProcessAudioResponse:

    request_start = time.perf_counter()

    audio_bytes = await file.read()
    file_size = len(audio_bytes)
    content_type = file.content_type or "application/octet-stream"

    logger.info(
        "Received: name='%s' type='%s' size=%d bytes",
        file.filename, content_type, file_size,
    )

    try:
        validate_audio_file(content_type, file_size)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        audio_array, sample_rate = convert_to_mono_wav(audio_bytes)
        duration = get_audio_duration(audio_array, sample_rate)
    except Exception as e:
        logger.error("Audio conversion failed: %s", str(e))
        raise HTTPException(status_code=422, detail=f"Cannot decode audio: {str(e)}")

    logger.info("Running diarization on %.2f second audio...", duration)
    try:
        parsed_labels = None
        if speaker_labels:
            parsed_labels = [
                label.strip() 
                for label in speaker_labels.split(",")
                if label.strip()
            ]
            logger.info("Custom speaker labels: %s", parsed_labels)

        segments = await diarization_svc.diarize(
            audio_bytes=audio_bytes,
            num_speakers=num_speakers,
            custom_labels=parsed_labels,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Diarization error: {str(e)}")

    if not segments:
        return ProcessAudioResponse(
            success=True,
            total_speakers=0,
            duration_seconds=round(duration, 2),
            segments=[],
            full_transcript="",
            processing_time_seconds=round(time.perf_counter() - request_start, 2),
        )

    logger.info("Running STT on %d segments...", len(segments))
    try:
        enriched = await stt_svc.transcribe_segments(
            audio_array=audio_array,
            sample_rate=sample_rate,
            segments=segments,
            language_code=language_code.value,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Transcription error: {str(e)}")

    from database.models import seconds_to_hms

    speaker_segments = [
        SpeakerSegment(
            speaker=seg["speaker"],
            speaker_label=seg["speaker_label"],
            start=seg["start"],
            end=seg["end"],
            start_formatted=seconds_to_hms(seg["start"]),
            end_formatted=seconds_to_hms(seg["end"]),
            transcript=seg.get("transcript", ""),
            language_detected=seg.get("language_detected"),
        )
        for seg in enriched
    ]

    full_transcript = "\n".join(
        f"{seg.speaker_label}: {seg.transcript}"
        for seg in speaker_segments
        if seg.transcript
    )

    total_speakers = len(set(seg.speaker for seg in speaker_segments))
    processing_time = round(time.perf_counter() - request_start, 2)

    logger.info(
        "Done: %d speakers | %d segments | %.2fs",
        total_speakers, len(speaker_segments), processing_time,
    )
    # Save to database
    try:
        crud.save_transcript(
            db=db,
            filename=file.filename or "unknown",
            duration_seconds=round(duration, 2),
            total_speakers=total_speakers,
            full_transcript=full_transcript,
            language_detected=speaker_segments[0].language_detected if speaker_segments else None,
            language_requested=language_code.value,
            processing_time_seconds=processing_time,
            segments=[seg.model_dump() for seg in speaker_segments],
        )
    except Exception as e:
        logger.error("Failed to save transcript to database: %s", str(e))
        # Don't crash — saving to DB failing should not break the API response

    return ProcessAudioResponse(
        success=True,
        total_speakers=total_speakers,
        duration_seconds=round(duration, 2),
        segments=speaker_segments,
        full_transcript=full_transcript,
        processing_time_seconds=processing_time,
    )
@app.get("/transcripts", tags=["Database"])
async def get_transcripts(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """Get all saved transcripts, newest first."""
    transcripts = crud.get_all_transcripts(db, skip=skip, limit=limit)
    return {
        "total": len(transcripts),
        "transcripts": [t.to_dict() for t in transcripts],
    }


@app.get("/transcripts/search", tags=["Database"])
async def search_transcripts(
    language: Optional[str] = None,
    speakers: Optional[int] = None,
    keyword: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Search transcripts by language, speaker count, or keyword.
    Example: /transcripts/search?language=kn-IN&keyword=sugar
    """
    results = crud.search_transcripts(
        db=db,
        language=language,
        speaker_count=speakers,
        keyword=keyword,
    )
    return {
        "total": len(results),
        "transcripts": [t.to_dict() for t in results],
    }


@app.get("/transcripts/{transcript_id}", tags=["Database"])
async def get_transcript(transcript_id: int, db: Session = Depends(get_db)):
    """Get one specific transcript by ID."""
    transcript = crud.get_transcript_by_id(db, transcript_id)
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found")
    return transcript.to_dict()


@app.delete("/transcripts/{transcript_id}", tags=["Database"])
async def delete_transcript(transcript_id: int, db: Session = Depends(get_db)):
    """Delete a transcript by ID."""
    deleted = crud.delete_transcript(db, transcript_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Transcript not found")
    return {"success": True, "message": f"Transcript {transcript_id} deleted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )


import uuid
from collections import defaultdict

# Store job progress in memory
# In production this would be Redis
job_progress = {}


@app.post("/process-audio-async", tags=["STT"])
async def process_audio_async(
    file: Annotated[UploadFile, File(description="Audio file")],
    language_code: Annotated[
        SupportedLanguage,
        Form(description="Language code or 'unknown' for auto-detect")
    ] = SupportedLanguage.AUTO,
    num_speakers: Annotated[
        Optional[int],
        Form(description="Exact number of speakers if known")
    ] = None,
    speaker_labels: Annotated[
        Optional[str],
        Form(description="Comma separated speaker names")
    ] = None,
    db: Session = Depends(get_db),
    diarization_svc: DiarizationService = Depends(get_diarization_service),
    stt_svc: STTService = Depends(get_stt_service),
):
    """
    Async version — returns a job ID immediately.
    Poll GET /job/{job_id} to check progress.
    Perfect for long audio files 15-30 minutes.
    """
    job_id = str(uuid.uuid4())[:8]
    audio_bytes = await file.read()

    job_progress[job_id] = {
        "status": "queued",
        "progress": 0,
        "step": "Queued",
        "result": None,
        "error": None,
    }

    # Run in background — return job_id immediately to user
    asyncio.create_task(
        run_transcription_job(
            job_id=job_id,
            audio_bytes=audio_bytes,
            filename=file.filename,
            language_code=language_code,
            num_speakers=num_speakers,
            speaker_labels=speaker_labels,
            db=db,
            diarization_svc=diarization_svc,
            stt_svc=stt_svc,
        )
    )

    return {
        "job_id": job_id,
        "message": "Job started. Poll /job/{job_id} for progress.",
        "poll_url": f"/job/{job_id}",
    }


async def run_transcription_job(
    job_id: str,
    audio_bytes: bytes,
    filename: str,
    language_code,
    num_speakers,
    speaker_labels,
    db,
    diarization_svc,
    stt_svc,
):
    """Background task that runs the full pipeline and updates progress."""
    try:
        # Step 1
        job_progress[job_id].update({
            "status": "processing",
            "progress": 10,
            "step": "Validating and converting audio...",
        })
        audio_array, sample_rate = convert_to_mono_wav(audio_bytes)
        duration = get_audio_duration(audio_array, sample_rate)

        # Step 2
        job_progress[job_id].update({
            "progress": 25,
            "step": f"Running speaker diarization on {duration:.0f}s audio...",
        })

        parsed_labels = None
        if speaker_labels:
            parsed_labels = [l.strip() for l in speaker_labels.split(",") if l.strip()]

        segments = await diarization_svc.diarize(
            audio_bytes=audio_bytes,
            num_speakers=num_speakers,
            custom_labels=parsed_labels,
        )

        # Step 3
        job_progress[job_id].update({
            "progress": 60,
            "step": f"Transcribing {len(segments)} speaker segments...",
        })

        enriched = await stt_svc.transcribe_segments(
            audio_array=audio_array,
            sample_rate=sample_rate,
            segments=segments,
            language_code=language_code.value,
        )

        # Step 4
        job_progress[job_id].update({
            "progress": 85,
            "step": "Saving to database...",
        })

        from database.models import seconds_to_hms
        speaker_segments = [
            SpeakerSegment(
                speaker=seg["speaker"],
                speaker_label=seg["speaker_label"],
                start=seg["start"],
                end=seg["end"],
                start_formatted=seconds_to_hms(seg["start"]),
                end_formatted=seconds_to_hms(seg["end"]),
                transcript=seg.get("transcript", ""),
                language_detected=seg.get("language_detected"),
            )
            for seg in enriched
        ]

        full_transcript = "\n".join(
            f"{seg.speaker_label}: {seg.transcript}"
            for seg in speaker_segments if seg.transcript
        )

        total_speakers = len(set(seg.speaker for seg in speaker_segments))

        try:
            crud.save_transcript(
                db=db,
                filename=filename or "unknown",
                duration_seconds=round(duration, 2),
                total_speakers=total_speakers,
                full_transcript=full_transcript,
                language_detected=speaker_segments[0].language_detected if speaker_segments else None,
                language_requested=language_code.value,
                processing_time_seconds=0,
                segments=[seg.model_dump() for seg in speaker_segments],
            )
        except Exception as e:
            logger.error("DB save failed: %s", str(e))

        # Done
        job_progress[job_id].update({
            "status": "completed",
            "progress": 100,
            "step": "Done!",
            "result": {
                "success": True,
                "total_speakers": total_speakers,
                "duration_seconds": round(duration, 2),
                "segments": [seg.model_dump() for seg in speaker_segments],
                "full_transcript": full_transcript,
            },
        })

    except Exception as e:
        logger.error("Job %s failed: %s", job_id, str(e))
        job_progress[job_id].update({
            "status": "failed",
            "progress": 0,
            "step": "Failed",
            "error": str(e),
        })


@app.get("/job/{job_id}", tags=["STT"])
async def get_job_status(job_id: str):
    """
    Poll this endpoint to check transcription progress.
    Returns progress percentage and current step.
    """
    if job_id not in job_progress:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_progress[job_id]