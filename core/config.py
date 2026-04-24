from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    # Sarvam AI — your STT API key
    sarvam_api_key: str
    sarvam_api_url: str = "https://api.sarvam.ai/speech-to-text-translate"
    sarvam_language_code: str = "unknown"

    # HuggingFace — to download the Pyannote model
    huggingface_token: str

    # Audio limits
    max_audio_size_bytes: int = 100 * 1024 * 1024   # 100 MB
    target_sample_rate: int = 16000

    # App info
    app_name: str = "Thunai STT Platform"
    app_version: str = "1.0.0"
    debug: bool = False
    database_url: str = "postgresql://postgres:Rohini@28@localhost:5432/thunai_db"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache()
def get_settings() -> Settings:
    return Settings()