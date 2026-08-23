"""Application settings, loaded from environment / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "dev"

    database_url: str = "postgresql+psycopg://recoup:recoup@localhost:5434/recoup"

    razorpay_key_id: str = ""
    razorpay_key_secret: str = ""
    razorpay_webhook_secret: str = ""

    anthropic_api_key: str = ""

    # Quiet hours are enforced in IST. No customer contact inside this window.
    quiet_hours_start: int = Field(default=21, ge=0, le=23)
    quiet_hours_end: int = Field(default=9, ge=0, le=23)

    @field_validator("razorpay_key_id")
    @classmethod
    def _refuse_live_keys(cls, v: str) -> str:
        """Hard stop on live credentials.

        This project moves money-adjacent state around. Nothing here has been
        built or reviewed for production use, so a live key is always a
        misconfiguration rather than a deliberate choice.
        """
        if v.startswith("rzp_live_"):
            raise ValueError(
                "Live Razorpay key detected. Recoup is test-mode only; "
                "use a key starting with rzp_test_."
            )
        return v

    @property
    def razorpay_configured(self) -> bool:
        return bool(self.razorpay_key_id and self.razorpay_key_secret)


@lru_cache
def get_settings() -> Settings:
    return Settings()
