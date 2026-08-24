"""Action tools: the only code permitted to do something a customer can see.

## The idempotency pattern, and why the order matters

Every tool follows the same sequence:

    1. INSERT the action row, keyed on (case_id, action_type, attempt_no)
    2. commit
    3. call the external service
    4. record the outcome

The insert comes **before** the external call, not after. That ordering is the
whole guarantee. If the process dies between steps 3 and 4, the row already
exists, so a retry finds it and refuses to send again -- the worst case is an
action recorded as `proposed` that actually went out, which a human can
reconcile. Reverse the order and the worst case is sending the same payment link
twice to a customer who has already paid, which nobody can un-send.

A `SELECT` before the insert would not be enough. Two concurrent webhook
deliveries can both find nothing and both proceed; the unique constraint is what
makes the race impossible rather than unlikely.

## Why a tool layer at all

The allocator decides; nothing else may act. Every externally visible effect goes
through a function here that validates its inputs, records what it did, and
cannot be reached from the model. That boundary is what makes the audit trail
trustworthy: if an action happened, there is a row, and if there is a row, one of
these functions produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Action, ActionStatus, Case
from app.services import ledger
from app.simulation.outcomes import CONTACT_ACTIONS, ActionType

#: Above this value an action needs a human before it goes out. A wrong recovery
#: attempt on a small payment is noise; on a large one it is a phone call from
#: the merchant.
HUMAN_APPROVAL_ABOVE_PAISE = 5_000_000  # Rs 50,000

#: Below this classification confidence a money-touching action needs a human,
#: whatever the amount. Acting confidently on a cause we are unsure of is how a
#: system does the wrong specific thing.
HUMAN_APPROVAL_BELOW_CONFIDENCE = 0.75


class ActionRefused(RuntimeError):
    """The action was not performed, and deliberately so."""


@dataclass(frozen=True)
class ActionResult:
    action: Action
    performed: bool
    #: True when an identical action already existed. Not an error -- it is the
    #: idempotency guarantee working.
    duplicate: bool = False
    detail: str = ""


def needs_human(case: Case, action_type: ActionType) -> str | None:
    """Whether this action must be approved first, and why.

    Retries are exempt: they touch the gateway rather than the customer, and
    queueing them for review would stall the cheapest recovery path behind a
    human for no benefit.
    """
    if action_type not in CONTACT_ACTIONS:
        return None
    if case.amount_paise >= HUMAN_APPROVAL_ABOVE_PAISE:
        return f"amount_above_{HUMAN_APPROVAL_ABOVE_PAISE // 100}_rupees"
    if (case.cause_confidence or 0.0) < HUMAN_APPROVAL_BELOW_CONFIDENCE:
        return f"cause_confidence_{case.cause_confidence:.2f}_below_threshold"
    return None


def claim(
    db: Session,
    case: Case,
    action_type: ActionType,
    *,
    attempt_no: int = 1,
    rule: str | None = None,
    predicted_uplift: float | None = None,
    expected_value_paise: int | None = None,
    cost_paise: int = 0,
) -> ActionResult:
    """Reserve the right to perform this action, exactly once.

    Returns `duplicate=True` if the action already exists, in which case the
    caller must not proceed. This is the function that makes redelivery safe.
    """
    key = Action.build_idempotency_key(case.id, str(action_type), attempt_no)

    existing = db.scalar(select(Action).where(Action.idempotency_key == key))
    if existing is not None:
        return ActionResult(existing, performed=False, duplicate=True,
                            detail="action already claimed")

    approval = needs_human(case, action_type)
    action = Action(
        case_id=case.id,
        action_type=str(action_type),
        attempt_no=attempt_no,
        idempotency_key=key,
        status=ActionStatus.PENDING_APPROVAL if approval else ActionStatus.PROPOSED,
        predicted_recovery_prob=predicted_uplift,
        expected_value_paise=expected_value_paise,
        cost_paise=cost_paise,
        approval_reason=approval,
        decided_by_rule=rule,
    )
    db.add(action)
    try:
        db.commit()
    except IntegrityError:
        # Lost a race with a concurrent claim. The constraint did its job.
        db.rollback()
        winner = db.scalar(select(Action).where(Action.idempotency_key == key))
        return ActionResult(winner, performed=False, duplicate=True,
                            detail="lost idempotency race")

    ledger.append(
        db,
        case_id=case.id,
        event_type="action.claimed" if not approval else "action.awaiting_approval",
        actor="allocator",
        payload={
            "action_type": str(action_type),
            "attempt_no": attempt_no,
            "idempotency_key": key,
            "rule": rule,
            "predicted_uplift": predicted_uplift,
            "expected_value_paise": expected_value_paise,
            "approval_reason": approval,
        },
    )
    db.commit()
    return ActionResult(action, performed=False, detail=approval or "claimed")


def _razorpay_client():
    settings = get_settings()
    if not settings.razorpay_configured:
        raise ActionRefused("Razorpay keys are not configured")
    import razorpay

    return razorpay.Client(auth=(settings.razorpay_key_id, settings.razorpay_key_secret))


def send_payment_link(
    db: Session, action: Action, case: Case, *, channel: str = "sms", live: bool = False
) -> ActionResult:
    """Create a Razorpay payment link for a case, and record it.

    `live=False` records the action without calling Razorpay. The default is
    deliberate: a test suite that creates real payment links on every run is a
    test suite that eventually sends something to somebody.

    Notifications are disabled even when live. This project has no business
    messaging a real phone number, and a payment link that quietly SMSes a
    stranger during a demo is the kind of mistake that is not recoverable.
    """
    if action.status not in (ActionStatus.PROPOSED, ActionStatus.EXECUTED):
        raise ActionRefused(f"action is {action.status}, not executable")

    if not live:
        return _record_executed(db, action, case, external_ref=None,
                                detail="recorded, not sent (live=False)")

    try:
        link = _razorpay_client().payment_link.create({
            "amount": case.amount_paise,
            "currency": case.currency,
            "description": f"Recovery for {case.external_ref}",
            "customer": {
                "contact": case.customer_contact or "",
                "email": case.customer_email or "",
            },
            # Never notify. Recoup decides *whether* to contact; it does not get
            # to have Razorpay message a real person as a side effect.
            "notify": {"sms": False, "email": False},
            "reminder_enable": False,
            "notes": {"recoup_case_id": case.id, "recoup_action_id": action.id},
        })
    except Exception as exc:  # noqa: BLE001 - any failure must be recorded, not raised
        # Includes a missing API key. A configuration error and a gateway
        # outage are different problems, but from the allocation pass's point of
        # view both mean "this did not happen", and one misconfigured deployment
        # must not take down a batch of two thousand cases. The ledger records
        # which it was.
        action.status = ActionStatus.FAILED
        db.commit()
        ledger.append(
            db,
            case_id=case.id,
            event_type="action.failed",
            actor="executor",
            payload={
                "action_id": action.id,
                "error": f"{type(exc).__name__}: {exc}"[:200],
            },
        )
        db.commit()
        return ActionResult(
            action, performed=False, detail=f"gateway error: {exc}"[:160]
        )

    return _record_executed(
        db,
        action,
        case,
        external_ref=link["id"],
        detail=f"payment link {link['id']}",
        extra={"short_url": link.get("short_url")},
    )


def schedule_retry(db: Session, action: Action, case: Case, *, delay_h: float) -> ActionResult:
    """Record a retry for later execution.

    A retry is not performed inline: the whole point of cause-aware timing is
    that the right moment is usually not now. The scheduled time is recorded and
    a tick executes it.
    """
    if action.status != ActionStatus.PROPOSED:
        raise ActionRefused(f"action is {action.status}, not executable")
    return _record_executed(db, action, case, external_ref=None,
                            detail=f"retry scheduled at +{delay_h:.2f}h",
                            extra={"scheduled_delay_h": delay_h})


def alert_merchant(db: Session, action: Action, case: Case, *, message: str) -> ActionResult:
    """Tell the merchant something only they can fix.

    The action for a merchant-configuration failure. There is no customer-facing
    option that works, and the useful output is an alert to the person who owns
    the setting.
    """
    return _record_executed(db, action, case, external_ref=None,
                            detail="merchant alerted", extra={"message": message})


def _record_executed(
    db: Session,
    action: Action,
    case: Case,
    *,
    external_ref: str | None,
    detail: str,
    extra: dict | None = None,
) -> ActionResult:
    action.status = ActionStatus.EXECUTED
    action.executed_at = datetime.now(UTC)
    action.external_ref = external_ref
    db.commit()

    ledger.append(
        db,
        case_id=case.id,
        event_type="action.executed",
        actor="executor",
        payload={
            "action_id": action.id,
            "action_type": action.action_type,
            "attempt_no": action.attempt_no,
            "external_ref": external_ref,
            "detail": detail,
            **(extra or {}),
        },
    )
    db.commit()
    return ActionResult(action, performed=True, detail=detail)


def approve(db: Session, action: Action, *, approved_by: str) -> ActionResult:
    """Release a queued action. Only a human reaches this."""
    if action.status != ActionStatus.PENDING_APPROVAL:
        raise ActionRefused(f"action is {action.status}, not awaiting approval")
    action.status = ActionStatus.PROPOSED
    action.approved_by = approved_by
    action.approved_at = datetime.now(UTC)
    db.commit()
    ledger.append(db, case_id=action.case_id, event_type="action.approved", actor="human",
                  payload={"action_id": action.id, "approved_by": approved_by,
                           "approval_reason": action.approval_reason})
    db.commit()
    return ActionResult(action, performed=False, detail="approved, awaiting execution")


def reject(db: Session, action: Action, *, rejected_by: str, reason: str) -> ActionResult:
    """Refuse a queued action permanently.

    Terminal. The idempotency key stays claimed, so nothing can quietly re-propose
    the same action later and get a different answer.
    """
    if action.status != ActionStatus.PENDING_APPROVAL:
        raise ActionRefused(f"action is {action.status}, not awaiting approval")
    action.status = ActionStatus.REJECTED
    action.approved_by = rejected_by
    action.approved_at = datetime.now(UTC)
    action.blocked_reason = reason[:128]
    db.commit()
    ledger.append(db, case_id=action.case_id, event_type="action.rejected", actor="human",
                  payload={"action_id": action.id, "rejected_by": rejected_by, "reason": reason})
    db.commit()
    return ActionResult(action, performed=False, detail="rejected")
