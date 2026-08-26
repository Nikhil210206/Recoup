"""End-to-end demo: webhook in, decision out, audit trail intact.

    make live-demo      (needs `make db` and a running `make api`)

Walks five cases whose correct treatment differs, and shows the system reaching a
different answer for each -- including two where the right answer is to do
nothing, and one where the right answer needs a human.

Nothing is sent to anybody: actions are recorded, not dispatched. That is the
default everywhere in this codebase, not a demo-only setting.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.request

from app.config import get_settings

#: 8000 is a common default and was already taken by another local project on
#: the machine this was built on. Overridable rather than hardcoded, for the
#: same reason the database port is.
BASE = os.environ.get("RECOUP_API", "http://localhost:8000")

#: Chosen so each row exercises a different branch of the decision tree.
SCENARIOS = [
    ("pay_DEMO_EXPIRED", "card_expired", "customer", 249_900,
     "expired card -> switch method, not retry"),
    ("pay_DEMO_OUTAGE", "bank_technical_error", "bank", 549_900,
     "bank outage -> retry, and do NOT message the customer"),
    ("pay_DEMO_BIG", "card_expired", "customer", 9_900_000,
     "same cause, large amount -> needs a human"),
    ("pay_DEMO_FRAUD", "payment_risk_check_failed", "issuer", 199_900,
     "fraud block -> suppress, never work around it"),
    ("pay_DEMO_MERCHANT", "international_transaction_not_allowed", "business", 349_900,
     "merchant's own setting -> alert the merchant, zero customer contact"),
]


def _post(path: str, body=None, raw: bytes | None = None, headers=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body else b"")
    request = urllib.request.Request(
        BASE + path, data=data, method="POST",
        headers=headers or {"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(request, timeout=180).read())


def _get(path: str):
    return json.loads(urllib.request.urlopen(BASE + path, timeout=60).read())


def _send_webhook(payment_id, reason, source, amount, secret):
    body = json.dumps({
        "entity": "event", "account_id": "acc_DEMO", "event": "payment.failed",
        "payload": {"payment": {"entity": {
            "id": payment_id, "order_id": f"order_{payment_id}", "amount": amount,
            "currency": "INR", "status": "failed", "method": "card",
            "error_code": "BAD_REQUEST_ERROR", "error_source": source,
            "error_step": "payment_authorization", "error_reason": reason,
            "contact": "+919000000101", "email": "demo@example.com",
            "created_at": int(time.time()) - 1800,
        }}},
    }).encode()
    return _post("/webhooks/razorpay", raw=body, headers={
        "Content-Type": "application/json",
        "X-Razorpay-Signature": hmac.new(secret.encode(), body, hashlib.sha256).hexdigest(),
        "X-Razorpay-Event-Id": f"evt_{payment_id}",
    })


def main() -> None:
    secret = get_settings().razorpay_webhook_secret
    if not secret:
        raise SystemExit("RAZORPAY_WEBHOOK_SECRET is not set in .env")
    try:
        _get("/health")
    except urllib.error.URLError:
        raise SystemExit("API is not running. Start it with `make api`.") from None

    print("Recoup live demo -- webhook to decision, nothing sent to anybody\n")

    print("1. five failed payments arrive, each needing a different answer")
    for payment_id, reason, source, amount, note in SCENARIOS:
        response = _send_webhook(payment_id, reason, source, amount, secret)
        marker = "case" if response.get("case_id") else "dup "
        print(f"   {marker}  Rs {amount / 100:>10,.0f}  {reason:38} {note}")

    print("\n2. allocate")
    result = _post("/actions/allocate?budget=100&live=false")
    for key in ("considered", "executed", "queued_for_approval", "suppressed"):
        print(f"   {key:22} {result.get(key)}")

    print("\n3. what it decided, and why")
    for payment_id, *_ in SCENARIOS:
        case_id = next(
            (c["id"] for c in [_lookup(payment_id)] if c), None
        )
        if case_id is None:
            continue
        detail = _get(f"/actions/case/{case_id}")
        case = detail["case"]
        if detail["actions"]:
            action = detail["actions"][0]
            verdict = f"{action['type']} ({action['status']})"
            why = action["approval_reason"] or action["rule"] or "-"
        else:
            suppression = next(
                (e for e in detail["ledger"] if e["event"] == "case.suppressed"), None
            )
            verdict = "NOTHING"
            why = suppression["payload"]["rule"] if suppression else case["status"]
        print(f"   {case['external_ref']:22} {case['cause']:26} -> {verdict:34} [{why}]")

    print("\n4. the approval queue")
    for item in _get("/actions/pending"):
        print(
            f"   Rs {item['amount_rupees']:>10,.0f}  "
            f"{item['action_type']:22} {item['why_queued']}"
        )

    print("\n5. replay the whole allocation four more times")
    for i in range(4):
        again = _post("/actions/allocate?budget=100&live=false")
        print(f"   pass {i + 2}: executed={again['executed']} "
              f"queued={again.get('queued_for_approval', 0)} stopped={again['stopped']}")
    print("   Nothing is sent twice. Idempotency is a unique constraint, not a check.")


def _lookup(payment_id: str) -> dict | None:
    """Find a case by its Razorpay payment id, via the audit endpoint."""
    try:
        return _get(f"/actions/lookup/{payment_id}")
    except urllib.error.HTTPError:
        return None


if __name__ == "__main__":
    main()
