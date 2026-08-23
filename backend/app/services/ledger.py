"""Append-only audit ledger.

The only supported write path for `case_events`. Nothing else in the codebase
should insert into that table, and nothing at all should update or delete from
it.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import CaseEvent


def append(
    db: Session,
    *,
    case_id: str,
    event_type: str,
    actor: str,
    payload: dict[str, Any] | None = None,
) -> CaseEvent:
    """Append one event to a case's ledger.

    `seq` is computed inside the transaction. The unique constraint on
    (case_id, seq) turns a lost race into a failed insert rather than two
    events silently sharing a position in the history.
    """
    next_seq = db.scalar(
        select(func.coalesce(func.max(CaseEvent.seq), 0) + 1).where(CaseEvent.case_id == case_id)
    )

    event = CaseEvent(
        case_id=case_id,
        seq=next_seq,
        event_type=event_type,
        actor=actor,
        payload=payload or {},
    )
    db.add(event)
    db.flush()
    return event


def history(db: Session, case_id: str) -> list[CaseEvent]:
    """Full decision history for one case, in order.

    This is the query that has to answer "why was case X contacted twice?"
    """
    return list(
        db.scalars(select(CaseEvent).where(CaseEvent.case_id == case_id).order_by(CaseEvent.seq))
    )
