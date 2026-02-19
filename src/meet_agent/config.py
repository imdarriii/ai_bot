"""Application settings loaded from environment / .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Recall AI ──────────────────────────────────────────────
    recall_api_key: str = ""
    recall_region: str = "eu-central-1"  # Frankfurt (closest to Central Asia)

    # ── OpenAI (LLM) ──────────────────────────────────────────
    openai_api_key: str = ""

    # ── Deepgram (STT) ────────────────────────────────────────
    deepgram_api_key: str = ""

    # ── ElevenLabs (TTS) ──────────────────────────────────────
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel

    # ── Webhook server ─────────────────────────────────────────
    webhook_base_url: str = "https://localhost"
    webhook_port: int = 8080

    # ── Bot ────────────────────────────────────────────────────
    bot_name: str = "Alex"
    recordings_dir: Path = Path("./recordings")

    @property
    def recall_base_url(self) -> str:
        return f"https://{self.recall_region}.recall.ai"


settings = Settings()
