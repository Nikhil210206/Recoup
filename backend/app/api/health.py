"""Liveness and configuration reporting."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db

router = APIRouter(tags=["health"])


@router.get("/health")
def health(db: Session = Depends(get_db)) -> dict:
    settings = get_settings()
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:  # noqa: BLE001 - a health check must report, never raise
        db_ok = False

    return {
        "ok": db_ok,
        "env": settings.app_env,
        "database": "up" if db_ok else "down",
        # Reported so a misconfigured setup is visible at a glance rather than
        # surfacing as a confusing 400 on the first webhook.
        "razorpay_keys": settings.razorpay_configured,
        "webhook_secret": bool(settings.razorpay_webhook_secret),
        "anthropic_key": bool(settings.anthropic_api_key),
    }
