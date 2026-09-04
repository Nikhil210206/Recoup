"""The real loop: a live Razorpay payment, failed and then recovered.

    make real-loop

Everything else in this repository is measured against a simulator. This runs
against Razorpay's actual test-mode API, with real webhooks arriving over a real
tunnel, and it is the only evidence that the system works on the thing it claims
to work on.

Two steps need a human, and deliberately so: entering card details is not
something this program does. It pauses, tells you exactly what to type, and waits
for the webhook.

Nothing here touches live money. The app refuses to start with a `rzp_live_` key,
and Razorpay notifications are disabled on every link it creates.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import razorpay

from app.api.guard import HEADER as GUARD_HEADER
from app.config import get_settings

BASE = os.environ.get("RECOUP_API", "http://localhost:8000")
POLL_SECONDS = 3
POLL_TIMEOUT = 300


@dataclass
class Step:
    number: int
    title: str


def _get(path: str):
    return json.loads(urllib.request.urlopen(BASE + path, timeout=60).read())


def _post(path: str, body: dict | None = None):
    data = json.dumps(body).encode() if body else b""
    headers = {"Content-Type": "application/json"}

    # This script is the one legitimate caller of the privileged parameters: it
    # allocates live and forces past the cause delay, because a hard decline now
    # waits six hours and a demonstration cannot. It therefore has to present
    # the same token the scheduler does.
    token = get_settings().tasks_token
    if token:
        headers[GUARD_HEADER] = token

    request = urllib.request.Request(BASE + path, data=data, method="POST", headers=headers)
    return json.loads(urllib.request.urlopen(request, timeout=180).read())


def _banner(step: Step) -> None:
    print(f"\n{'─' * 74}\n{step.number}. {step.title}\n{'─' * 74}")


def _wait_for(description: str, check, timeout: int = POLL_TIMEOUT):
    """Poll until `check()` returns something truthy, or give up loudly."""
    print(f"   waiting for {description} ", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = check()
        if result:
            print(" ok")
            return result
        print(".", end="", flush=True)
        time.sleep(POLL_SECONDS)
    print(" TIMED OUT")
    return None


def _case_for(payment_id: str):
    try:
        return _get(f"/actions/lookup/{payment_id}")
    except urllib.error.HTTPError:
        return None


def main() -> None:
    settings = get_settings()
    if not settings.razorpay_configured:
        raise SystemExit("Razorpay test keys are not set in .env")
    if not settings.razorpay_key_id.startswith("rzp_test_"):
        raise SystemExit("refusing to run against anything but a test key")
    try:
        _get("/health")
    except urllib.error.URLError:
        raise SystemExit("API is not running. Start it with `make api`.") from None

    client = razorpay.Client(
        auth=(settings.razorpay_key_id, settings.razorpay_key_secret)
    )

    print("Recoup -- real Razorpay test-mode loop")
    print("Every other number in this repository comes from a simulator.")
    print("This one does not.")

    # ---- 1. a real order that a customer will fail to pay -------------------
    _banner(Step(1, "create a real order in Razorpay test mode"))
    order = client.order.create({
        "amount": 249900, "currency": "INR",
        "receipt": f"recoup-real-{int(time.time())}",
        "notes": {"purpose": "recoup_real_loop"},
    })
    print(f"   order   {order['id']}   Rs {order['amount'] / 100:,.0f}")

    checkout = client.payment_link.create({
        "amount": order["amount"], "currency": "INR",
        "description": "Recoup real-loop: please FAIL this payment",
        "customer": {"name": "Recoup Test", "contact": "+919000000042",
                     "email": "test@example.com"},
        "notify": {"sms": False, "email": False}, "reminder_enable": False,
    })
    print(f"   pay at  {checkout['short_url']}")
    started_at = int(time.time())

    _banner(Step(2, "FAIL that payment (this part needs you)"))
    print("   Open the link above and pay with a card, then make it FAIL.")
    print()
    print("   On the simulated OTP screen, click the FAILURE button. Typing a")
    print("   short OTP does not work -- the form rejects anything outside 4-10")
    print("   digits before it ever reaches Razorpay, so no payment is attempted")
    print("   and no webhook fires.")
    print()
    print("   Card: any test card from")
    print("     https://razorpay.com/docs/payments/payments/test-card-details/")
    print("   4111 1111 1111 1111 is international and this account is")
    print("   domestic-only, so it fails at initiation with a merchant-config")
    print("   cause -- a real failure, but the loop ends there, because the")
    print("   correct answer is to alert the merchant and not the customer.")
    print()
    print("   Press ENTER once you have submitted a failing payment.")
    input("   > ")

    def _newly_failed():
        """The failed payment *this run* produced.

        Filtering by time matters more than it looks. Without it this picked the
        first failed payment in the account's recent history -- which was a
        payment from a previous day -- and then waited forever for a case that
        had long since been truncated. The script sat there looking correct.
        """
        for payment in client.payment.all({"count": 20, "from": started_at})["items"]:
            if payment["status"] == "failed" and payment["created_at"] >= started_at:
                return payment
        return None

    failed = _wait_for("the payment.failed webhook", _newly_failed)
    if not failed:
        raise SystemExit(
            "no failed payment from this run appeared.\n"
            "  - did the payment actually fail? a rejected OTP form is not a\n"
            "    failed payment; the payment has to be submitted and declined\n"
            "  - is the tunnel still up, and is the webhook URL current?"
        )
    print(f"   failed payment {failed['id']}  reason={failed.get('error_reason')}")

    case = _wait_for(
        "Recoup to open a case", lambda: _case_for(failed["id"])
    )
    if not case:
        raise SystemExit(
            "the webhook never reached us. Check the tunnel and the Razorpay "
            "webhook URL."
        )

    detail = _get(f"/actions/case/{case['id']}")
    print(f"   case {case['id'][:8]}  cause={detail['case']['cause']} "
          f"({detail['case']['cause_method']}, confidence "
          f"{detail['case']['cause_confidence']})")

    # ---- 3. decide, and act for real ----------------------------------------
    _banner(Step(3, "allocate, and create a REAL payment link"))
    result = _post("/actions/allocate?budget=10&live=true")
    for key in ("considered", "executed", "queued_for_approval", "suppressed"):
        print(f"   {key:22} {result.get(key)}")

    detail = _get(f"/actions/case/{case['id']}")
    actions = detail["actions"]
    if not actions:
        print("\n   No action was taken. That is a decision, not a failure:")
        for entry in detail["ledger"]:
            if entry["event"] in ("case.suppressed", "case.stopped"):
                print(f"     {entry['event']}: {entry['payload'].get('rule')} "
                      f"-- {entry['payload'].get('reason')}")
        return

    action = actions[0]
    print(f"   action  {action['type']}  status={action['status']}")

    # The allocator now schedules a customer contact for the moment the cause
    # implies -- six hours after a hard decline, forty-eight after insufficient
    # funds. That is correct and it is also longer than anyone will sit here, so
    # the operator override sends it early. It is written to the ledger as a
    # human decision; see `case.schedule_overridden` in the history below.
    deferred = any(e["event"] == "case.action_deferred" for e in detail["ledger"])
    if deferred:
        due = next(
            e["payload"] for e in reversed(detail["ledger"])
            if e["event"] == "case.action_deferred"
        )
        print(f"   scheduled for {due['due_at']} ({due['waiting_h']}h away)")
        print("   overriding the schedule, as an operator would for a demo")
        pushed = _post("/tasks/execute-due?force=true&live=true")
        print(f"   sent {pushed.get('sent')}, held for quiet hours "
              f"{pushed.get('still_in_quiet_hours')}")
        if pushed.get("still_in_quiet_hours"):
            raise SystemExit(
                "It is quiet hours in IST (21:00-09:00). The override does not "
                "bypass those -- run this between 09:00 and 21:00 IST."
            )
        detail = _get(f"/actions/case/{case['id']}")
        action = detail["actions"][0]

    if action["status"] == "pending_approval":
        print(f"   queued for a human: {action['approval_reason']}")
        print("   approving it now, as the reviewer would")
        _post(f"/actions/{action['id']}/approve",
              {"approved_by": "real-loop", "execute_now": True, "live": True})
        detail = _get(f"/actions/case/{case['id']}")
        action = detail["actions"][0]

    link_id = action.get("external_ref")
    if not link_id:
        raise SystemExit("no payment link was created; nothing to pay")

    link = client.payment_link.fetch(link_id)
    print(f"   payment link {link_id}")
    print(f"   pay at       {link['short_url']}")

    # ---- 4. the customer pays --------------------------------------------
    _banner(Step(4, "PAY the recovery link (this part needs you too)"))
    print("   Open the link above and pay it successfully.")
    print("   Razorpay's test success card: 4111 1111 1111 1111 usually works")
    print("   for a domestic-enabled account; otherwise use any success card")
    print("   from the docs above, any future expiry, any CVV.")
    print()
    print("   Press ENTER once the payment succeeds.")
    input("   > ")

    recovered = _wait_for(
        "Recoup to attribute the recovery",
        lambda: (
            _get(f"/actions/case/{case['id']}")["case"]["status"] == "recovered"
            and _get(f"/actions/case/{case['id']}")
        ),
    )

    _banner(Step(5, "what the audit trail says"))
    final = recovered or _get(f"/actions/case/{case['id']}")
    print(f"   case status: {final['case']['status']}")
    print()
    for entry in final["ledger"]:
        matched = entry["payload"].get("matched_by")
        suffix = f"   [matched by {matched}]" if matched else ""
        print(f"   {entry['seq']:>2}  {entry['event']:26} {entry['actor']:12}{suffix}")

    if final["case"]["status"] == "recovered":
        print("\n   Real order, real failure, real webhook, real decision,")
        print("   real payment link, real payment, recovery attributed.")
    else:
        print("\n   The recovery was not attributed. The ledger above shows how far")
        print("   it got; the likely cause is the payment.captured webhook not")
        print("   arriving, which means the tunnel or the webhook subscription.")
        sys.exit(1)


if __name__ == "__main__":
    main()
