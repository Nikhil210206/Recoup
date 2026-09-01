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

# `live_allocator` is imported inside the endpoint below rather than here: it
# pulls pandas and pyarrow, and this module is on the import path of every
# process that serves a webhook. See the note in api/actions.py.

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.post("/classify-pending")
def run_classify_pending(limit: int = 25, db: Session = Depends(get_db)) -> dict:
    """Drain the deferred classification queue.

    These are cases whose error code is not in Razorpay's published tables, so
    the deterministic tier could not resolve them and the webhook handler parked
    them rather than blocking on a model call.
    """
    waiting = db.scalar(
        select(func.count()).select_from(Case).where(Case.status == CaseStatus.PENDING_DIAGNOSIS)
    )
    processed = classify_pending(db, limit=limit)
    return {
        "queued_before": int(waiting or 0),
        "processed": processed,
        "remaining": max(int(waiting or 0) - processed, 0),
    }


@router.post("/execute-due")
def run_execute_due(
    limit: int = 100,
    live: bool = False,
    force: bool = False,
    db: Session = Depends(get_db),
) -> dict:
    """Send contacts whose deferral has expired.

    Quiet hours defer a customer contact rather than dropping it, so something
    has to pick the deferred ones up again. Without this tick the deferral is
    indistinguishable from a silent drop -- and the ledger would say the contact
    was scheduled while nothing ever sent it.

    `live` defaults to False, matching every other execution path here.

    `force` sends contacts ahead of their cause-implied delay -- needed for a
    demonstration, and logged to the ledger as a human override. It does not
    bypass quiet hours.
    """
    from app.services import live_allocator

    return live_allocator.execute_due(db, limit=limit, live=live, force=force)


@router.get("/queue")
def queue_depth(db: Session = Depends(get_db)) -> dict:
    """Case counts by status. The first thing to look at when something stalls."""
    rows = db.execute(select(Case.status, func.count()).group_by(Case.status)).all()
    return {str(status): int(count) for status, count in rows}
