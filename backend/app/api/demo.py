"""The console's interactive demonstration.

A reader can pick a Razorpay failure reason and watch the system decide. The
point is not that something happens -- it is that a *different* thing happens
for each reason, chosen by the published taxonomy rather than by a rule someone
wrote for the demo.

**What is real here and what is not.** The delivery is injected: no webhook
arrives from Razorpay, and this endpoint constructs the payload itself. Once
constructed it goes through the same adapter, the same classifier, the same
allocator and the same gates as a genuine `payment.failed`. Nothing is scripted
and no outcome is canned -- if the taxonomy changes, this page changes with it.

`live` is not a parameter. Every demonstration records its actions without
calling Razorpay, so no payment link can ever be created from this page.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import taxonomy
from app.db import get_db
from app.models import Case, CaseStatus, LossChannel

router = APIRouter(prefix="/api/demo", tags=["demo"])

#: Identifies everything this endpoint creates, so a demonstration case is never
#: mistaken for a seeded one or -- far more importantly -- for the real recovery.
DEMO_MERCHANT = "DEMO_CONSOLE"

#: An upper bound on how much a public page may write to the database. Without
#: it, the demo is an unauthenticated row generator pointed at production.
MAX_DEMO_CASES = 150

#: The five reasons offered, chosen because each produces a visibly different
#: decision. Two of them produce no customer contact at all, which is the half
#: of the argument a recovery demo usually leaves out.
OFFERED: tuple[dict[str, str], ...] = (
    {
        "error_reason": "insufficient_funds",
        "label": "Not enough money in the account",
        "expect": "Retry — but later, not now",
    },
    {
        "error_reason": "card_expired",
        "label": "The card has expired",
        "expect": "Ask for a different method — never a retry",
    },
    {
        "error_reason": "bank_technical_error",
        "label": "The issuing bank is down",
        "expect": "Retry, and do not message the customer",
    },
    {
        "error_reason": "payment_risk_check_failed",
        "label": "Blocked by a risk check",
        "expect": "Refuse to act at all",
    },
    {
        "error_reason": "international_transaction_not_allowed",
        "label": "The merchant's own configuration blocked it",
        "expect": "Alert the merchant, contact nobody",
    },
)

_OFFERED_BY_REASON = {o["error_reason"]: o for o in OFFERED}


@router.get("/causes")
def causes() -> list[dict[str, Any]]:
    """The reasons a reader may pick, with what the taxonomy says about each.

    Served rather than hardcoded in the page, so the console cannot drift from
    the taxonomy it is demonstrating.
    """
    out = []
    for offer in OFFERED:
        result = taxonomy.classify(offer["error_reason"], "customer", "payment_authorization")
        cause = taxonomy.get(result.cause) if result.cause else None
        out.append(
            {
                **offer,
                "cause": result.cause,
                "cause_label": cause.label if cause else None,
                "who_can_fix": cause.who_can_fix.value if cause else None,
                "retry_policy": cause.retry_policy.value if cause else None,
                "contact_ok": cause.contact_ok if cause else None,
            }
        )
    return out


@router.post("/simulate")
def simulate(
    error_reason: str = Body(..., embed=True),
    amount_paise: int = Body(default=249900, embed=True),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Inject one failed payment and let the system decide what to do about it.

    Only the five offered reasons are accepted. This endpoint writes to the
    database from a public page, so it takes a closed set rather than whatever
    string a caller supplies.
    """
    if error_reason not in _OFFERED_BY_REASON:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"unknown reason; pick one of {sorted(_OFFERED_BY_REASON)}",
        )
    if not 100 <= amount_paise <= 10_000_000:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "amount out of range")

    made = db.scalar(
        select(func.count()).select_from(Case).where(Case.merchant_id == DEMO_MERCHANT)
    )
    if (made or 0) >= MAX_DEMO_CASES:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "demonstration limit reached; run `make seed` to reset the console",
        )

    # `ingest_payment_failed` diagnoses inline before returning -- calling
    # diagnose() again here appended a second `case.diagnosed` to an append-only
    # ledger for a diagnosis that happened once.
    from app.services.ingest import ingest_payment_failed

    token = secrets.token_hex(6).upper()
    event = {
        "account_id": DEMO_MERCHANT,
        "payload": {
            "payment": {
                "entity": {
                    "id": f"pay_DEMO{token}",
                    "order_id": f"order_DEMO{token}",
                    "amount": amount_paise,
                    "currency": "INR",
                    "method": "card",
                    "bank": "HDFC",
                    "contact": "+910000000000",
                    "email": None,
                    "created_at": int(datetime.now(UTC).timestamp()),
                    "error_code": "BAD_REQUEST_ERROR",
                    "error_source": "customer",
                    "error_step": "payment_authorization",
                    "error_reason": error_reason,
                    "error_description": _OFFERED_BY_REASON[error_reason]["label"],
                    "notes": {"recoup_demo": True},
                }
            }
        },
    }

    case = ingest_payment_failed(db, event)
    if case is None:  # pragma: no cover - the id is freshly random
        raise HTTPException(status.HTTP_409_CONFLICT, "case already exists")
    db.commit()

    if case.status == CaseStatus.DIAGNOSED:
        from app.api.actions import _build_allocator
        from app.services import live_allocator

        # live=False is not a default here, it is the only option. A public page
        # must not be able to create a payment link.
        live_allocator.allocate_and_execute(db, _build_allocator(10), live=False, cases=[case])
        db.commit()

    db.refresh(case)
    return {
        "case_id": case.id,
        "external_ref": case.external_ref,
        "channel": LossChannel.FAILED_PAYMENT.value,
        "simulated": True,
        "note": "Injected payload; real classifier, allocator and gates. No Razorpay call.",
    }
