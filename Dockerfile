# ============================================================
# Dockerfile for Thunai STT Platform
# ============================================================
#
# WHY DOCKER?
# "Works on my machine" is not a deployment strategy. Docker packages
# the app + all its dependencies (Python, ffmpeg, system libs) into
# a single portable image that runs identically everywhere.
#
# MULTI-STAGE BUILD (not used here for simplicity, but recommended
# for large projects to keep the final image small).
#
# BASE IMAGE CHOICE:
# python:3.11-slim — Debian-based, minimal footprint (~130MB vs
# ~900MB for the full image). We install only what we need.
# ============================================================

FROM python:3.11-slim

# --- System dependencies ---
# ffmpeg: required by pydub for audio format conversion (MP3, M4A, OGG → WAV)
# libsndfile1: C library that soundfile (Python) wraps for reading audio files
# build-essential: C compiler needed to build some Python extension modules
# We chain all apt commands in one RUN to minimize Docker layers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
# rm -rf /var/lib/apt/lists/ removes the package index cache,
# reducing the image size by ~50MB.

# --- Working directory ---
# All subsequent commands run from /app inside the container.
WORKDIR /app

# --- Install Python dependencies first (cache optimization) ---
# WHY COPY requirements.txt BEFORE the rest of the code?
# Docker caches layers. If we copy everything at once, any code
# change invalidates the pip install layer (slow, ~5 minutes).
# By copying requirements.txt first, pip install only re-runs
# when requirements.txt changes — not on every code edit.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Copy application code ---
COPY . .

# --- Non-root user for security ---
# Running as root in a container is a security risk.
# If the app is exploited, the attacker gets root-level access.
# Creating a dedicated user limits the blast radius.
RUN useradd --create-home --shell /bin/bash thunai
USER thunai

# --- Expose port ---
# Documents which port the app listens on.
# Docker doesn't enforce this — it's metadata for operators.
EXPOSE 8000

# --- Health check ---
# Docker will periodically call this endpoint to determine if
# the container is healthy. Unhealthy containers are restarted automatically.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

# --- Start the server ---
# We use uvicorn directly (not `python main.py`) because uvicorn
# is the production ASGI server — it handles signal management
# (SIGTERM for graceful shutdown) more reliably than Python's default.
# --workers=1: Pyannote loads a ~1GB model. Multiple workers would
# multiply memory usage. Use horizontal scaling (multiple containers)
# instead of multiple workers per container.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]