"""Background work, driven by an explicit tick rather than a worker daemon.

Everything the webhook path cannot afford to do synchronously happens here. A
tick is a plain HTTP call, so in production it is a cron entry, and in
development it is curl.

Deliberately not Celery or a queue broker. The work is a bounded scan over rows
with a status, the state lives in Postgres where the audit trail already is, and
a tick that is easy to run by hand is easy to reason about at 2am. A broker would
add a moving part without removing one.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Case, CaseStatus
from app.services.ingest import classify_pending

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.post("/classify-pending")
def run_classify_pending(limit: int = 25, db: Session = Depends(get_db)) -> dict:
    """Drain the deferred classification queue.

    These are cases whose error code is not in Razorpay's published tables, so
    the deterministic tier could not resolve them and the webhook handler parked
    them rather than blocking on a model call.
    """
    waiting = db.scalar(
        select(func.count())
        .select_from(Case)
        .where(Case.status == CaseStatus.PENDING_DIAGNOSIS)
    )
    processed = classify_pending(db, limit=limit)
    return {
        "queued_before": int(waiting or 0),
        "processed": processed,
        "remaining": max(int(waiting or 0) - processed, 0),
    }


@router.get("/queue")
def queue_depth(db: Session = Depends(get_db)) -> dict:
    """Case counts by status. The first thing to look at when something stalls."""
    rows = db.execute(
        select(Case.status, func.count()).group_by(Case.status)
    ).all()
    return {str(status): int(count) for status, count in rows}
