"""Print one case's audit trail, and check it against Razorpay.

    make ledger                      the real recovery on the deployed service
    make ledger CASE=pay_XXXX        any payment reference
    make ledger CASE=<case-uuid>     or a case id directly

Two things are printed, and the separation is the point.

The **ledger** is what Recoup recorded: what it saw, what it decided, and why.
The **cross-check** asks Razorpay the same questions independently and prints
whatever comes back. If the two disagree, this says so.

An earlier version of this file printed "cross-checked against Razorpay: the
link reads paid" as a hardcoded string. It would have printed identically had
the link been unpaid. That is exactly the class of defect `INCIDENTS.md` is
full of -- a plausible claim, in the right shape, verifying nothing -- so the
check is now a real API call whose result is reported rather than assumed.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

from app.config import get_settings

#: Defaults to the deployed service. Override for a local API.
BASE = os.environ.get("RECOUP_API", "https://recoup-yti2.onrender.com")

#: The real recovery documented in `docs/evidence/real_loop_deployed.json`.
DEFAULT_CASE = "be4d3bdf-7df8-4c2b-9ef4-281754c4d099"

#: What each event means, for someone reading a ledger for the first time.
NOTE = {
    "case.opened": "a Razorpay webhook arrived",
    "case.diagnosed": "deterministic lookup, no model",
    "action.claimed": "the taxonomy chose the action",
    "case.action_deferred": "held for the cause-implied delay",
    "case.schedule_overridden": "a human sent it early",
    "case.suppressed": "refused; no action has positive value",
    "case.stopped": "a stopping rule declined it",
    "action.executed": "payment link created",
    "case.recovered": "attributed by notes.recoup_case_id",
}

RULE = "─" * 74


def _get(path: str):
    return json.load(urllib.request.urlopen(BASE + path, timeout=60))


def _resolve(ref: str) -> str:
    """Accept a Razorpay payment reference or a case id."""
    if not ref.startswith("pay_"):
        return ref
    return _get(f"/actions/lookup/{ref}")["id"]


def _cross_check(link_id: str | None, paid_by: str | None) -> None:
    """Ask Razorpay directly. Report what it says, or that we could not ask."""
    print(f"\n{RULE}\n  CROSS-CHECK — asking Razorpay directly\n{RULE}\n")

    if not link_id:
        print("  no payment link on this case; nothing to cross-check.")
        return

    try:
        import razorpay

        s = get_settings()
        client = razorpay.Client(auth=(s.razorpay_key_id, s.razorpay_key_secret))
        link = client.payment_link.fetch(link_id)

        print(f"  payment link {link_id}")
        print(f"    status     {link['status']}")
        print(f"    reference  {link.get('reference_id')}")

        agrees = link["status"] == "paid"
        if paid_by:
            payment = client.payment.fetch(paid_by)
            print(f"  capturing payment {paid_by}")
            print(f"    status     {payment['status']}")
            print(f"    amount     Rs {payment['amount'] / 100:,.0f}")
            agrees = agrees and payment["status"] == "captured"

        print(f"\n  Razorpay {'AGREES with' if agrees else 'DISAGREES with'} our ledger.")
    except Exception as exc:  # noqa: BLE001 - name the failure, claim nothing
        print(f"  could not reach Razorpay: {exc}")
        print("  (no cross-check performed — the ledger above is unverified here)")


def main() -> None:
    ref = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CASE
    try:
        detail = _get(f"/actions/case/{_resolve(ref)}")
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"no such case at {BASE}: {ref} ({exc.code})") from None
    except urllib.error.URLError as exc:
        raise SystemExit(f"could not reach {BASE}: {exc.reason}") from None

    case, ledger = detail["case"], detail["ledger"]
    action = (detail["actions"] or [{}])[0]

    print(f"\n{RULE}\n  RECOUP — case audit trail\n  {BASE}\n{RULE}\n")
    print(f"  payment      {case['external_ref']}")
    print(f"  amount       Rs {case['amount_rupees']:,.0f}")
    print(f"  cause        {case['cause']}  "
          f"({case['cause_method']}, confidence {case['cause_confidence']})")
    print(f"  action       {action.get('type') or '— none taken —'}")
    print(f"  link         {action.get('external_ref') or '—'}")
    print(f"  final        {case['status'].upper()}\n")

    print(f"{RULE}\n  THE APPEND-ONLY LEDGER\n{RULE}\n")
    for event in ledger:
        print(f"  {event['seq']:>2}  {event['at'][11:19]}  "
              f"{event['event']:26} {event['actor']:10} {NOTE.get(event['event'], '')}")

    _cross_check(
        action.get("external_ref"),
        next(
            (e["payload"].get("recovering_payment_id")
             for e in ledger if e["event"] == "case.recovered"),
            None,
        ),
    )
    print(f"{RULE}\n")


if __name__ == "__main__":
    main()
