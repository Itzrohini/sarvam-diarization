# services/analytics_service.py
# ============================================================
# WHAT IS THE ANALYTICS SERVICE?
# It reads all saved transcripts from PostgreSQL and
# calculates business insights:
#   - Which languages are used most
#   - How many transcriptions per day
#   - Most common words customers say
#   - Average conversation duration
#   - Speaker talk time ratio


from sqlalchemy.orm import Session
from database.models import Transcript
from core.logger import get_logger
from collections import Counter
import re
from datetime import datetime, timedelta

logger = get_logger(__name__)

# Words to ignore when counting keywords
# These are common words that appear everywhere but mean nothing
STOP_WORDS = {
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "i", "you", "he", "she", "we", "they",
    "this", "that", "was", "are", "be", "have", "has", "had", "do",
    "does", "did", "will", "would", "can", "could", "may", "might",
    "ಮತ್ತು", "ಅಥವಾ", "ಇದು", "ಅದು", "ಈ", "ಅವರು", "ನಾನು", "ನೀವು",
    "और", "या", "यह", "वह", "मैं", "आप", "हम", "वे", "है", "हैं",
    "என்று", "இது", "அது", "நான்", "நீங்கள்", "அவர்கள்",
}


class AnalyticsService:
    """
    Calculates analytics from saved transcripts.
    All methods take a database session and return
    clean dicts ready for JSON API responses.
    """

    def get_summary(self, db: Session) -> dict:
        """
        Overall summary statistics.
        Returns total counts and averages.
        """
        transcripts = db.query(Transcript).all()

        if not transcripts:
            return {
                "total_transcriptions": 0,
                "total_audio_minutes": 0,
                "average_duration_seconds": 0,
                "average_speakers": 0,
                "average_processing_time": 0,
                "total_speakers_detected": 0,
            }

        total = len(transcripts)
        total_duration = sum(t.duration_seconds for t in transcripts)
        total_speakers = sum(t.total_speakers for t in transcripts)
        avg_processing = sum(t.processing_time_seconds for t in transcripts) / total

        logger.info("Analytics summary: %d transcripts, %.1f total minutes",
                    total, total_duration / 60)

        return {
            "total_transcriptions": total,
            "total_audio_minutes": round(total_duration / 60, 1),
            "average_duration_seconds": round(total_duration / total, 1),
            "average_speakers": round(total_speakers / total, 1),
            "average_processing_time": round(avg_processing, 1),
            "total_speakers_detected": total_speakers,
        }

    def get_language_breakdown(self, db: Session) -> list[dict]:
        """
        Count transcripts by detected language.
        Returns sorted list with percentages.

        Example output:
        [
          {"language": "kn-IN", "count": 45, "percentage": 42.5},
          {"language": "hi-IN", "count": 30, "percentage": 28.3},
        ]
        """
        transcripts = db.query(Transcript).all()

        if not transcripts:
            return []

        # Count each language
        lang_counter = Counter()
        for t in transcripts:
            lang = t.language_detected or "unknown"
            lang_counter[lang] += 1

        total = len(transcripts)

        # Map language codes to readable names
        lang_names = {
            "kn-IN": "Kannada",
            "hi-IN": "Hindi",
            "ta-IN": "Tamil",
            "te-IN": "Telugu",
            "ml-IN": "Malayalam",
            "en-IN": "English (Indian)",
            "unknown": "Auto Detected",
        }

        result = []
        for lang_code, count in lang_counter.most_common():
            result.append({
                "language_code": lang_code,
                "language_name": lang_names.get(lang_code, lang_code),
                "count": count,
                "percentage": round((count / total) * 100, 1),
            })

        return result

    def get_daily_transcriptions(self, db: Session, days: int = 30) -> list[dict]:
        """
        Count transcriptions per day for the last N days.
        Used to draw the bar chart on the dashboard.

        Example output:
        [
          {"date": "2026-04-01", "count": 12, "total_minutes": 45.3},
          {"date": "2026-04-02", "count": 8,  "total_minutes": 30.1},
        ]
        """
        since = datetime.utcnow() - timedelta(days=days)
        transcripts = db.query(Transcript).filter(
            Transcript.created_at >= since
        ).all()

        # Group by date
        daily = {}
        for t in transcripts:
            if t.created_at:
                date_str = t.created_at.strftime("%Y-%m-%d")
                if date_str not in daily:
                    daily[date_str] = {"count": 0, "total_duration": 0}
                daily[date_str]["count"] += 1
                daily[date_str]["total_duration"] += t.duration_seconds

        # Build sorted list filling in zero days
        result = []
        for i in range(days):
            date = (datetime.utcnow() - timedelta(days=days-1-i)).strftime("%Y-%m-%d")
            data = daily.get(date, {"count": 0, "total_duration": 0})
            result.append({
                "date": date,
                "count": data["count"],
                "total_minutes": round(data["total_duration"] / 60, 1),
            })

        return result

    def get_top_keywords(self, db: Session, top_n: int = 20) -> list[dict]:
        """
        Find the most common words across all transcripts.
        Excludes stop words (common words like "the", "and").
        Shows what customers actually talk about most.

        Example output for a retail shop:
        [
          {"word": "sugar",  "count": 145},
          {"word": "kilo",   "count": 132},
          {"word": "rice",   "count": 98},
        ]
        """
        transcripts = db.query(Transcript).all()

        if not transcripts:
            return []

        word_counter = Counter()

        for t in transcripts:
            if not t.full_transcript:
                continue

            # Extract words — letters only, lowercase
            # re.findall gets all word sequences
            words = re.findall(r'\b[a-zA-Z\u0C00-\u0CFF\u0900-\u097F\u0B80-\u0BFF\u0C80-\u0CFF\u0D00-\u0D7F]{2,}\b',
                               t.full_transcript.lower())

            # Filter stop words and short words
            for word in words:
                if word not in STOP_WORDS and len(word) > 2:
                    word_counter[word] += 1

        return [
            {"word": word, "count": count}
            for word, count in word_counter.most_common(top_n)
        ]

    def get_speaker_stats(self, db: Session) -> dict:
        """
        Analyse speaker patterns across all transcripts.
        Shows average talk time per speaker role.
        """
        transcripts = db.query(Transcript).all()

        if not transcripts:
            return {"speaker_stats": []}

        speaker_durations = Counter()
        speaker_counts = Counter()

        for t in transcripts:
            if not t.segments:
                continue
            for seg in t.segments:
                label = seg.get("speaker_label", "Unknown")
                duration = seg.get("end", 0) - seg.get("start", 0)
                speaker_durations[label] += duration
                speaker_counts[label] += 1

        total_duration = sum(speaker_durations.values()) or 1

        stats = []
        for label in speaker_durations:
            duration = speaker_durations[label]
            stats.append({
                "speaker_label": label,
                "total_seconds": round(duration, 1),
                "total_minutes": round(duration / 60, 1),
                "percentage": round((duration / total_duration) * 100, 1),
                "total_segments": speaker_counts[label],
            })

        # Sort by total speaking time
        stats.sort(key=lambda x: x["total_seconds"], reverse=True)

        return {"speaker_stats": stats}


# Singleton
_analytics_service = None


def get_analytics_service() -> AnalyticsService:
    global _analytics_service
    if _analytics_service is None:
        _analytics_service = AnalyticsService()
    return _analytics_service