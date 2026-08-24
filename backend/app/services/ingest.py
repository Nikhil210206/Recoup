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

from app.models import Action, Case, CaseStatus, CauseMethod, LossChannel
from app.services import classifier as classifier_mod
from app.services import ledger


def _entity(event: dict[str, Any], key: str) -> dict[str, Any]:
    return event.get("payload", {}).get(key, {}).get("entity", {}) or {}


def _ts(epoch: object) -> datetime:
    """Razorpay sends unix seconds. Fall back to now if absent."""
    return (
        datetime.fromtimestamp(epoch, tz=UTC)
        if isinstance(epoch, int)
        else datetime.now(UTC)
    )


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
        customer_contact=payment.get("contact"),
        customer_email=payment.get("email"),
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

    diagnose(db, case)
    return case


def diagnose(db: Session, case: Case) -> Case:
    """Assign a root cause inline, using the deterministic tier only.

    **The LLM tail must not run here.** Razorpay retries a webhook that does not
    respond promptly, and a cold local-model call measured at 31 seconds in this
    project. The sequence that produces is: delivery times out -> Razorpay
    retries -> our idempotency correctly rejects the duplicate -> the case is
    never classified at all, while Razorpay records repeated delivery failures.

    So the webhook path does what it can do in microseconds and returns. Anything
    the lookup cannot resolve is parked in `PENDING_DIAGNOSIS` and picked up by
    `classify_pending`, which runs outside the request.

    The ledger records the method and confidence, not just the answer. When
    someone asks why a case was suppressed or escalated, "a model said so at 0.62"
    and "Razorpay's own error code says so" are very different answers, and the
    audit trail has to distinguish them.
    """
    result = classifier_mod.default_classifier.classify(
        case.error_reason,
        case.error_source,
        case.error_step,
        description=case.error_description,
        payment_method=case.payment_method,
        allow_llm=False,
    )
    return _record_diagnosis(db, case, result, deferred_ok=True)


def classify_pending(db: Session, limit: int = 25) -> int:
    """Run the LLM tail over cases the deterministic tier could not resolve.

    Called from the tick endpoint, not from the webhook. Returns how many cases
    were processed.
    """
    pending = list(
        db.scalars(
            select(Case)
            .where(Case.status == CaseStatus.PENDING_DIAGNOSIS)
            .order_by(Case.detected_at)
            .limit(limit)
        )
    )
    for case in pending:
        result = classifier_mod.default_classifier.classify(
            case.error_reason,
            case.error_source,
            case.error_step,
            description=case.error_description,
            payment_method=case.payment_method,
        )
        _record_diagnosis(db, case, result, deferred_ok=False)
    db.commit()
    return len(pending)


def _record_diagnosis(
    db: Session, case: Case, result, *, deferred_ok: bool
) -> Case:
    """Persist a classification outcome and its justification.

    `deferred_ok` distinguishes the two callers. Inline, an unresolved case is
    *queued*; after the tail has run, an unresolved case is a genuine exception.
    """
    unresolved = result.cause is None

    if unresolved and deferred_ok:
        case.status = CaseStatus.PENDING_DIAGNOSIS
        event_type = "case.diagnosis_deferred"
    elif result.is_exception:
        case.status = CaseStatus.EXCEPTION
        event_type = "case.exception"
    else:
        case.status = CaseStatus.DIAGNOSED
        event_type = "case.diagnosed"

    case.cause = result.cause
    case.cause_confidence = result.confidence
    case.cause_method = (
        CauseMethod.DETERMINISTIC
        if result.method == "deterministic"
        else CauseMethod.LLM
        if result.method == "llm"
        else CauseMethod.UNMAPPED
    )

    ledger.append(
        db,
        case_id=case.id,
        event_type=event_type,
        actor="classifier",
        payload={
            "cause": result.cause,
            "confidence": result.confidence,
            "method": result.method,
            "matched_on": result.matched_on,
            "rationale": result.rationale,
            "model": result.model,
            "latency_ms": result.latency_ms,
            "cached": result.cached,
            "threshold": classifier_mod.CONFIDENCE_THRESHOLD,
        },
    )
    return case


def ingest_subscription_failed(db: Session, event: dict[str, Any]) -> Case | None:
    """Adapter for `subscription.pending` and `subscription.halted`.

    Razorpay sends `subscription.pending` when a charge attempt fails, and
    `subscription.halted` once its retries are exhausted. Both are revenue at
    risk; halted is simply later and worse.

    This channel behaves differently from a one-off failed payment in a way that
    matters to the allocator: a subscription has a live mandate, so a retry needs
    no customer involvement at all. Recovering it may cost zero contacts. An
    abandoned checkout, by contrast, can *only* be recovered by reaching the
    customer. Under a shared contact budget those two facts point in opposite
    directions, which is exactly the trade-off nothing currently arbitrates.
    """
    sub = _entity(event, "subscription")
    sub_id = sub.get("id")
    if not sub_id:
        return None

    payment = _entity(event, "payment")
    attempts = int(sub.get("auth_attempts") or 0)
    halted = event.get("event") == "subscription.halted"

    # One case per subscription per billing cycle, not per retry notification.
    cycle = sub.get("current_start") or sub.get("charge_at") or attempts
    external_ref = f"{sub_id}:{cycle}"

    existing = db.scalar(
        select(Case).where(
            Case.external_ref == external_ref,
            Case.channel == LossChannel.FAILED_SUBSCRIPTION,
        )
    )
    if existing is not None:
        return None

    amount = int(payment.get("amount") or sub.get("amount") or 0)
    case = Case(
        external_ref=external_ref,
        order_ref=payment.get("order_id"),
        channel=LossChannel.FAILED_SUBSCRIPTION,
        merchant_id=str(event.get("account_id") or "unknown"),
        customer_id=str(sub.get("customer_id") or f"sub:{sub_id}"),
        amount_paise=amount,
        currency=str(payment.get("currency") or "INR"),
        detected_at=_ts(event.get("created_at")),
        status=CaseStatus.OPEN,
        error_code=payment.get("error_code"),
        error_source=payment.get("error_source"),
        error_step=payment.get("error_step"),
        error_reason=payment.get("error_reason"),
        error_description=payment.get("error_description"),
        payment_method=payment.get("method"),
        attempt_no=max(attempts, 1),
    )
    db.add(case)
    db.flush()

    ledger.append(
        db,
        case_id=case.id,
        event_type="case.opened",
        actor="webhook",
        payload={
            "channel": LossChannel.FAILED_SUBSCRIPTION.value,
            "subscription_id": sub_id,
            "razorpay_event": event.get("event"),
            "auth_attempts": attempts,
            "halted": halted,
            "amount_paise": amount,
            "error_reason": case.error_reason,
        },
    )
    diagnose(db, case)
    return case


def ingest_abandoned_checkout(db: Session, event: dict[str, Any]) -> Case | None:
    """Adapter for Magic Checkout's abandoned-cart webhook.

    Razorpay documents the payload but not a stable event-name string, so this
    is matched on payload shape -- a `cart_token` with an
    `abandoned_checkout_url` -- rather than on an event name that may differ by
    integration. `webhooks.py` routes anything unrecognised here before giving up.

    There is no error code on this channel: nothing failed, the customer simply
    left. So the cause is known without any classification, and `retry` is
    meaningless -- there is no payment to re-attempt. The only route to the money
    is contacting the customer, which makes every one of these cases a claim on
    the contact budget.
    """
    cart = event.get("payload", {}).get("cart", {}) or event.get("payload", {}) or {}
    token = cart.get("cart_token") or cart.get("token")
    if not token:
        return None

    existing = db.scalar(
        select(Case).where(
            Case.external_ref == str(token),
            Case.channel == LossChannel.ABANDONED_CHECKOUT,
        )
    )
    if existing is not None:
        return None

    total = cart.get("line_items_total") or cart.get("amount") or 0
    customer = cart.get("customer") or {}
    case = Case(
        external_ref=str(token),
        channel=LossChannel.ABANDONED_CHECKOUT,
        merchant_id=str(event.get("account_id") or cart.get("shop_id") or "unknown"),
        customer_id=str(
            customer.get("id") or cart.get("phone") or cart.get("email") or f"cart:{token}"
        ),
        customer_contact=cart.get("phone"),
        customer_email=cart.get("email"),
        amount_paise=int(total),
        currency=str(cart.get("currency") or "INR"),
        detected_at=_ts(event.get("created_at")),
        status=CaseStatus.DIAGNOSED,
        # No failure occurred, so no Razorpay error fields exist. The cause is
        # structural rather than inferred, and the classifier is not consulted.
        cause="customer_abandoned",
        cause_confidence=1.0,
        cause_method=CauseMethod.DETERMINISTIC,
    )
    db.add(case)
    db.flush()

    ledger.append(
        db,
        case_id=case.id,
        event_type="case.opened",
        actor="webhook",
        payload={
            "channel": LossChannel.ABANDONED_CHECKOUT.value,
            "cart_token": token,
            "amount_paise": case.amount_paise,
            "checkout_url": cart.get("abandoned_checkout_url"),
            "note": "no error code on this channel; cause is structural",
        },
    )
    return case


def _find_case_for_payment(db: Session, payment: dict[str, Any]) -> tuple[Case | None, str]:
    """Work out which case, if any, a successful payment settles.

    Three strategies, most specific first. More than one is needed because a
    recovery does not necessarily arrive on the order that failed:

    **notes** -- a payment link we created carries `recoup_case_id` in its notes,
    and Razorpay copies link notes onto the resulting payment. This is exact.

    **payment_link_id** -- if the notes are missing, the payment still names the
    link, and the link id is stored on the action that created it.

    **order_id** -- the customer simply retried the original order. No link
    involved.

    The order-only version was the whole implementation, and it would have failed
    on every link-driven recovery: a payment link creates its *own* order, lazily,
    at payment time. `payment_link.create` returns `order_id: None`. So the
    captured payment carries an order the case has never heard of, and the
    recovery this system exists to cause would have been recorded as an unrelated
    payment.
    """
    notes = payment.get("notes") or {}
    case_id = notes.get("recoup_case_id")
    if case_id:
        case = db.scalar(select(Case).where(Case.id == str(case_id)))
        if case is not None:
            return case, "notes.recoup_case_id"

    link_id = payment.get("payment_link_id")
    if link_id:
        action = db.scalar(select(Action).where(Action.external_ref == str(link_id)))
        if action is not None:
            case = db.scalar(select(Case).where(Case.id == action.case_id))
            if case is not None:
                return case, "payment_link_id"

    order_id = payment.get("order_id")
    if order_id:
        case = db.scalar(
            select(Case)
            .where(Case.order_ref == str(order_id))
            .order_by(Case.detected_at.desc())
        )
        if case is not None:
            return case, "order_id"

    return None, "unmatched"


def resolve_recovery(db: Session, event: dict[str, Any]) -> Case | None:
    """Adapter for `payment.captured` -- attributes a recovery on the live loop.

    Attribution is deliberately conservative about *credit*. Only a case we
    actually acted on can be marked recovered. A customer who paid before the
    allocator did anything is recorded as a self-recovery, because blurring those
    two is exactly how recovery numbers inflate.
    """
    payment = _entity(event, "payment")
    case, matched_by = _find_case_for_payment(db, payment)
    if case is None:
        return None

    amount = int(payment.get("amount") or 0)
    acted = case.status in (CaseStatus.ACTED, CaseStatus.ALLOCATED)

    if acted:
        case.status = CaseStatus.RECOVERED
        event_type = "case.recovered"
        note = None
    elif case.status in (
        CaseStatus.OPEN,
        CaseStatus.DIAGNOSED,
        CaseStatus.PENDING_DIAGNOSIS,
        CaseStatus.SUPPRESSED,
    ):
        case.status = CaseStatus.CLOSED_UNRECOVERED
        event_type = "case.self_recovered"
        note = "customer paid without an intervention from Recoup"
    else:
        return None

    ledger.append(
        db,
        case_id=case.id,
        event_type=event_type,
        actor="webhook",
        payload={
            "recovering_payment_id": payment.get("id"),
            "order_id": payment.get("order_id"),
            "payment_link_id": payment.get("payment_link_id"),
            "amount_paise": amount,
            "matched_by": matched_by,
            **({"note": note} if note else {}),
        },
    )
    return case
