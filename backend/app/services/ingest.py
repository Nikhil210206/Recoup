"""Turn Razorpay webhook payloads into revenue-at-risk cases.

Each loss channel gets an adapter here. The allocator downstream never sees a
Razorpay payload -- it sees a Case. That boundary is what lets a fourth loss
channel be added later without touching allocation logic.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Case, CaseStatus, LossChannel
from app.services import ledger


def _entity(event: dict[str, Any], key: str) -> dict[str, Any]:
    return event.get("payload", {}).get(key, {}).get("entity", {}) or {}


def _customer_ref(payment: dict[str, Any]) -> str:
    """Best available stable identifier for the payer.

    Razorpay may or may not attach a customer_id depending on how checkout was
    configured, so fall back to contact, then email. Anonymous payers get a
    per-payment identity, which correctly means they carry no contact history.
    """
    for key in ("customer_id", "contact", "email"):
        value = payment.get(key)
        if value:
            return str(value)
    return f"anon:{payment.get('id', 'unknown')}"


def ingest_payment_failed(db: Session, event: dict[str, Any]) -> Case | None:
    """Adapter for the `payment.failed` webhook.

    Returns the Case, or None if this payment already has one -- a redelivered
    webhook must not open a second case for the same failure.
    """
    payment = _entity(event, "payment")
    payment_id = payment.get("id")
    if not payment_id:
        return None

    existing = db.scalar(
        select(Case).where(
            Case.external_ref == payment_id,
            Case.channel == LossChannel.FAILED_PAYMENT,
        )
    )
    if existing is not None:
        return None

    created_at = payment.get("created_at")
    detected_at = (
        datetime.fromtimestamp(created_at, tz=UTC)
        if isinstance(created_at, int)
        else datetime.now(UTC)
    )

    case = Case(
        external_ref=payment_id,
        order_ref=payment.get("order_id"),
        channel=LossChannel.FAILED_PAYMENT,
        merchant_id=str(event.get("account_id") or "unknown"),
        customer_id=_customer_ref(payment),
        amount_paise=int(payment.get("amount") or 0),
        currency=str(payment.get("currency") or "INR"),
        detected_at=detected_at,
        status=CaseStatus.OPEN,
        # Razorpay's taxonomy, stored exactly as received.
        error_code=payment.get("error_code"),
        error_source=payment.get("error_source"),
        error_step=payment.get("error_step"),
        error_reason=payment.get("error_reason"),
        error_description=payment.get("error_description"),
        payment_method=payment.get("method"),
        issuer=payment.get("bank") or payment.get("wallet"),
    )
    db.add(case)
    db.flush()

    ledger.append(
        db,
        case_id=case.id,
        event_type="case.opened",
        actor="webhook",
        payload={
            "channel": LossChannel.FAILED_PAYMENT.value,
            "razorpay_payment_id": payment_id,
            "order_id": payment.get("order_id"),
            "amount_paise": case.amount_paise,
            "error_code": case.error_code,
            "error_source": case.error_source,
            "error_step": case.error_step,
            "error_reason": case.error_reason,
        },
    )
    return case


def resolve_recovery(db: Session, event: dict[str, Any]) -> Case | None:
    """Adapter for `payment.captured` -- attributes a recovery on the live loop.

    Attribution matches on `order_id`, deliberately. Matching on customer would
    count any later unrelated purchase as a recovery and inflate every number in
    the evaluation.

    Only cases we actually acted on can be marked recovered. A customer who
    retried on their own, before the allocator did anything, is not a recovery
    this system gets to claim -- that distinction is the whole point of tracking
    unnecessary-contact cost, so it must not be blurred here.
    """
    payment = _entity(event, "payment")
    order_id = payment.get("order_id")
    if not order_id:
        return None

    case = db.scalar(
        select(Case)
        .where(Case.order_ref == order_id)
        .where(Case.status.in_([CaseStatus.ACTED, CaseStatus.ALLOCATED]))
        .order_by(Case.detected_at.desc())
    )
    if case is None:
        # Either no such case, or one we never acted on. Record the second case
        # as a self-recovery so the evaluation can separate "we recovered it"
        # from "it recovered itself".
        unacted = db.scalar(
            select(Case)
            .where(Case.order_ref == order_id)
            .where(Case.status.in_([CaseStatus.OPEN, CaseStatus.DIAGNOSED, CaseStatus.SUPPRESSED]))
            .order_by(Case.detected_at.desc())
        )
        if unacted is None:
            return None
        unacted.status = CaseStatus.CLOSED_UNRECOVERED
        ledger.append(
            db,
            case_id=unacted.id,
            event_type="case.self_recovered",
            actor="webhook",
            payload={
                "recovering_payment_id": payment.get("id"),
                "order_id": order_id,
                "amount_paise": int(payment.get("amount") or 0),
                "note": "customer paid without an intervention from Recoup",
            },
        )
        return unacted

    case.status = CaseStatus.RECOVERED
    ledger.append(
        db,
        case_id=case.id,
        event_type="case.recovered",
        actor="webhook",
        payload={
            "recovering_payment_id": payment.get("id"),
            "order_id": order_id,
            "amount_paise": int(payment.get("amount") or 0),
        },
    )
    return case
