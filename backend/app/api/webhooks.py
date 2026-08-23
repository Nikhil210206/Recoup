"""Razorpay webhook receiver.

The ordering in `receive` is deliberate and is the part worth reading:

  1. Verify the signature over the raw bytes. Reject before parsing anything.
  2. Claim the delivery by inserting the Razorpay event id, which carries a
     unique constraint. A redelivery loses that insert and returns early.
  3. Only then dispatch to an adapter.

Steps 2 and 3 are in that order because Razorpay retries on non-2xx and can
deliver the same event more than once even on success. Doing the work first and
deduplicating afterwards is how you send a customer two payment links for one
failure.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Header, Request, Response, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.models import WebhookDelivery
from app.services import ingest
from app.services.razorpay_signature import verify

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

# Event types we act on. Anything else is acknowledged and recorded but ignored,
# so an unexpected event never 500s and never triggers a Razorpay retry storm.
HANDLERS = {
    "payment.failed": ingest.ingest_payment_failed,
    "payment.captured": ingest.resolve_recovery,
    # A failed subscription charge, and the same subscription once Razorpay has
    # exhausted its own retries.
    "subscription.pending": ingest.ingest_subscription_failed,
    "subscription.halted": ingest.ingest_subscription_failed,
}

#: Magic Checkout's abandoned-cart webhook has a documented payload but no
#: documented event-name string, and integrations differ. Rather than guess a
#: name, an unrecognised event carrying a cart payload is routed by shape.
SHAPE_HANDLERS = [
    (
        lambda e: bool(
            (e.get("payload", {}).get("cart") or e.get("payload", {})).get("cart_token")
        ),
        ingest.ingest_abandoned_checkout,
    ),
]


@router.post("/razorpay", status_code=status.HTTP_200_OK)
async def receive(
    request: Request,
    response: Response,
    x_razorpay_signature: str | None = Header(default=None),
    x_razorpay_event_id: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> dict:
    settings = get_settings()

    # 1. Signature over the exact bytes received. Never re-serialise first.
    raw_body = await request.body()
    if not verify(raw_body, x_razorpay_signature, settings.razorpay_webhook_secret):
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"ok": False, "error": "invalid_signature"}

    try:
        event = json.loads(raw_body)
    except json.JSONDecodeError:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"ok": False, "error": "malformed_json"}

    event_type = event.get("event", "unknown")

    # Razorpay should always send an event id; fall back to a content hash so a
    # missing header cannot silently disable deduplication.
    event_id = x_razorpay_event_id or f"sha256:{_body_digest(raw_body)}"

    # 2. Claim the delivery. The unique constraint is the guarantee, not a
    #    preceding SELECT -- a check-then-insert races under concurrent retries.
    delivery = WebhookDelivery(
        event_id=event_id,
        event_type=event_type,
        signature_valid=True,
        raw_body=raw_body.decode("utf-8", errors="replace"),
    )
    db.add(delivery)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        return {"ok": True, "duplicate": True, "event_id": event_id}

    # 3. Dispatch.
    handler = HANDLERS.get(event_type)
    if handler is None:
        handler = next((h for matches, h in SHAPE_HANDLERS if matches(event)), None)
    if handler is None:
        delivery.result = "ignored_event_type"
        delivery.processed_at = datetime.now(UTC)
        db.commit()
        return {"ok": True, "handled": False, "event": event_type}

    case = handler(db, event)
    delivery.result = f"case:{case.id}" if case else "no_case"
    delivery.processed_at = datetime.now(UTC)
    db.commit()

    return {
        "ok": True,
        "handled": True,
        "event": event_type,
        "case_id": case.id if case else None,
    }


def _body_digest(raw_body: bytes) -> str:
    import hashlib

    return hashlib.sha256(raw_body).hexdigest()
