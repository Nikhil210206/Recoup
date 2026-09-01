"""Approval queue and allocation trigger.

The approval queue is the human-in-the-loop boundary. Two things land in it: an
action above a value threshold, and an action whose cause the classifier was not
confident about. Both are cases where acting confidently on a guess is worse than
waiting.

Retries never queue. They touch the gateway rather than the customer, so putting
them behind a human would stall the cheapest recovery path for no benefit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Body, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.allocator.budget import BudgetPolicy
from app.db import get_db
from app.models import Action, ActionStatus, Case, CaseStatus
from app.services import actions as action_tools

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from app.allocator.estimator import CauseRate
    from app.allocator.policy import Allocator

# The allocator, the estimator and the outcome enum are imported inside the
# functions that use them, NOT here. Importing them at module scope drags
# pandas, numpy, scikit-learn, scipy and pyarrow -- ~378MB -- into every process
# that imports `app.main`, including one that only ever answers a webhook.
#
# On a host that sleeps when idle, that import cost is paid on the cold start
# that a Razorpay webhook wakes up. Razorpay times out, retries, and the
# idempotency guard then rejects the retry, leaving the case unclassified. That
# is the Day 4 incident exactly, re-entering through the deployment rather than
# through the handler. Keep these imports lazy.
#
# `app.allocator.budget` is safe at module scope: it imports nothing heavy.

router = APIRouter(prefix="/actions", tags=["actions"])

_ESTIMATOR: CauseRate | None = None


def _build_allocator(budget: int) -> Allocator:
    """Construct the allocator used on the live path.

    The estimator is fitted once and cached. It is a group-by over ~20 cause and
    action combinations, so refitting per request would be wasteful, and refitting
    per request from *live* data would also make the live path non-reproducible
    against the measured one.
    """
    from app.allocator.estimator import CauseRate
    from app.allocator.policy import Allocator

    global _ESTIMATOR
    if _ESTIMATOR is None:
        from app.model.train import build_frames

        try:
            _ESTIMATOR = CauseRate().fit(build_frames("base", 42)["train"])
        except FileNotFoundError:
            # No dataset generated. The allocator still works -- ranking by value
            # with a constant uplift measured within 3% of the fitted estimator,
            # so this degrades rather than fails.
            from app.allocator.estimator import AmountOnly

            _ESTIMATOR = AmountOnly()
    return Allocator(
        estimator=_ESTIMATOR,
        budget_policy=BudgetPolicy(max_contacts_per_customer=2, max_total_contacts=budget),
    )


@router.post("/allocate")
def allocate(
    limit: int = 500,
    budget: int = 100,
    live: bool = False,
    db: Session = Depends(get_db),
) -> dict:
    """Plan over open cases and execute what survives every gate.

    `live` defaults to False, which records actions without calling Razorpay. An
    endpoint that creates real payment links every time it is hit is one nobody
    can safely run twice.
    """
    from app.services import live_allocator

    allocator = _build_allocator(budget)
    return live_allocator.allocate_and_execute(db, allocator, limit=limit, live=live)


@router.get("/pending")
def pending(limit: int = 50, db: Session = Depends(get_db)) -> list[dict]:
    """Actions waiting on a human, newest first."""
    rows = db.execute(
        select(Action, Case)
        .join(Case, Case.id == Action.case_id)
        .where(Action.status == ActionStatus.PENDING_APPROVAL)
        .order_by(Action.created_at.desc())
        .limit(limit)
    ).all()
    return [
        {
            "action_id": action.id,
            "case_id": case.id,
            "external_ref": case.external_ref,
            "channel": case.channel.value,
            "action_type": action.action_type,
            "amount_rupees": case.amount_paise / 100,
            "cause": case.cause,
            "cause_confidence": case.cause_confidence,
            "why_queued": action.approval_reason,
            "predicted_uplift": action.predicted_recovery_prob,
            "expected_value_rupees": (
                action.expected_value_paise / 100
                if action.expected_value_paise is not None
                else None
            ),
            "decided_by_rule": action.decided_by_rule,
        }
        for action, case in rows
    ]


def _load_pending(db: Session, action_id: str) -> Action:
    action = db.scalar(select(Action).where(Action.id == action_id))
    if action is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such action")
    if action.status != ActionStatus.PENDING_APPROVAL:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"action is {action.status.value}, not awaiting approval",
        )
    return action


@router.post("/{action_id}/approve")
def approve(
    action_id: str,
    approved_by: str = Body(..., embed=True),
    execute_now: bool = Body(default=True, embed=True),
    live: bool = Body(default=False, embed=True),
    db: Session = Depends(get_db),
) -> dict:
    """Release a queued action.

    `approved_by` is required and recorded. An approval with no name attached is
    not an approval anyone can be accountable for.
    """
    from app.simulation.outcomes import ActionType

    action = _load_pending(db, action_id)
    case = db.scalar(select(Case).where(Case.id == action.case_id))
    result = action_tools.approve(db, action, approved_by=approved_by)

    executed = False
    if execute_now and case is not None:
        if action.action_type == str(ActionType.RETRY):
            action_tools.schedule_retry(db, action, case, delay_h=1.0)
        else:
            action_tools.send_payment_link(db, action, case, live=live)
        # The case has now been acted on. Leaving it DIAGNOSED made an executed
        # case indistinguishable from an unworked one in every status query,
        # including the dashboard's.
        case.status = CaseStatus.ACTED
        db.commit()
        executed = True

    return {
        "action_id": action.id,
        "status": action.status.value,
        "approved_by": approved_by,
        "executed": executed,
        "detail": result.detail,
    }


@router.post("/{action_id}/reject")
def reject(
    action_id: str,
    rejected_by: str = Body(..., embed=True),
    reason: str = Body(..., embed=True),
    db: Session = Depends(get_db),
) -> dict:
    """Refuse a queued action permanently.

    The idempotency key stays claimed, so nothing can quietly re-propose the same
    action later and get a different answer from a different reviewer.
    """
    action = _load_pending(db, action_id)
    action_tools.reject(db, action, rejected_by=rejected_by, reason=reason)
    return {"action_id": action.id, "status": action.status.value, "reason": reason}


@router.get("/lookup/{external_ref}")
def lookup(external_ref: str, db: Session = Depends(get_db)) -> dict:
    """Find a case by its Razorpay payment id.

    Operationally the useful key: an incident starts with a payment id from a
    merchant, not with our internal case id.
    """
    case = db.scalar(select(Case).where(Case.external_ref == external_ref))
    if case is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no case for that reference")
    return {"id": case.id, "external_ref": case.external_ref, "status": case.status.value}


@router.get("/case/{case_id}")
def case_history(case_id: str, db: Session = Depends(get_db)) -> dict:
    """The full decision history for one case.

    This is the query that has to answer "why was this case contacted twice?"
    from the ledger alone, and it is the one to expect in a review.
    """
    case = db.scalar(select(Case).where(Case.id == case_id))
    if case is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such case")

    from app.services.ledger import history

    return {
        "case": {
            "id": case.id,
            "external_ref": case.external_ref,
            "channel": case.channel.value,
            "status": case.status.value,
            "amount_rupees": case.amount_paise / 100,
            "cause": case.cause,
            "cause_confidence": case.cause_confidence,
            "cause_method": case.cause_method.value if case.cause_method else None,
        },
        "actions": [
            {
                "id": a.id,
                "type": a.action_type,
                "attempt": a.attempt_no,
                "status": a.status.value,
                "idempotency_key": a.idempotency_key,
                "rule": a.decided_by_rule,
                "approval_reason": a.approval_reason,
                "approved_by": a.approved_by,
                "external_ref": a.external_ref,
            }
            for a in case.actions
        ],
        "ledger": [
            {
                "seq": e.seq,
                "event": e.event_type,
                "actor": e.actor,
                "at": e.created_at.isoformat(),
                "payload": e.payload,
            }
            for e in history(db, case.id)
        ],
    }
