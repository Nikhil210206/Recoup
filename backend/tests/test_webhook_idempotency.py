"""Signature verification and webhook deduplication.

The duplicate-delivery test is the one that matters. Razorpay redelivers
webhooks; if a redelivery opens a second case, the merchant's customer gets two
payment links for one failed payment. That is a money bug, not a cosmetic one,
so it gets a test rather than a comment.
"""

from __future__ import annotations

import json

import pytest

from app.services.razorpay_signature import compute_signature, verify

SECRET = "whsec_test_example"


def _body(payment_id: str = "pay_TEST0001", order_id: str = "order_TEST0001") -> bytes:
    return json.dumps(
        {
            "entity": "event",
            "account_id": "acc_TEST",
            "event": "payment.failed",
            "payload": {
                "payment": {
                    "entity": {
                        "id": payment_id,
                        "order_id": order_id,
                        "amount": 849900,
                        "currency": "INR",
                        "status": "failed",
                        "method": "upi",
                        "error_code": "BAD_REQUEST_ERROR",
                        "error_source": "customer",
                        "error_step": "payment_authentication",
                        "error_reason": "incorrect_otp",
                        "error_description": "Payment failed due to incorrect OTP",
                        "contact": "+919000000001",
                        "created_at": 1755800000,
                    }
                }
            },
        }
    ).encode("utf-8")


class TestSignature:
    def test_valid_signature_accepted(self):
        body = _body()
        assert verify(body, compute_signature(body, SECRET), SECRET) is True

    def test_tampered_body_rejected(self):
        body = _body()
        sig = compute_signature(body, SECRET)
        tampered = body.replace(b'"amount": 849900', b'"amount": 100')
        assert verify(tampered, sig, SECRET) is False

    def test_missing_signature_rejected(self):
        assert verify(_body(), None, SECRET) is False

    def test_unset_secret_fails_closed(self):
        """No secret configured must reject, not accept."""
        body = _body()
        assert verify(body, compute_signature(body, SECRET), "") is False

    def test_wrong_secret_rejected(self):
        body = _body()
        assert verify(body, compute_signature(body, "other_secret"), SECRET) is False

    def test_reserialised_body_does_not_verify(self):
        """Guards the most common integration mistake: signing the parsed dict
        instead of the raw bytes. Key order and spacing differ, so the digest
        differs, and it looks like a Razorpay problem rather than ours."""
        body = _body()
        sig = compute_signature(body, SECRET)
        reserialised = json.dumps(json.loads(body)).encode("utf-8")
        if reserialised != body:
            assert verify(reserialised, sig, SECRET) is False


@pytest.mark.integration
class TestDuplicateDelivery:
    """Requires Postgres (`make db`). Run with `pytest -m integration`."""

    def test_replayed_webhook_creates_one_case(self, client, db_session):
        from app.models import Case, CaseEvent

        body = _body()
        headers = {
            "X-Razorpay-Signature": compute_signature(body, SECRET),
            "X-Razorpay-Event-Id": "evt_TEST_DUPLICATE_0001",
            "Content-Type": "application/json",
        }

        responses = [
            client.post("/webhooks/razorpay", content=body, headers=headers) for _ in range(5)
        ]

        assert responses[0].json()["handled"] is True
        assert all(r.json().get("duplicate") is True for r in responses[1:])

        cases = db_session.query(Case).filter(Case.external_ref == "pay_TEST0001").all()
        assert len(cases) == 1, "a redelivered webhook opened a second case"

        events = db_session.query(CaseEvent).filter(CaseEvent.case_id == cases[0].id).all()
        opened = [e for e in events if e.event_type == "case.opened"]
        assert len(opened) == 1, "duplicate ledger entry for one failure"
