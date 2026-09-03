"""The console's interactive demonstration.

This endpoint writes to the database from a page anyone can open, and its whole
claim is that the decision comes from the taxonomy rather than from a branch
written for the demo. Both of those need tests: one so it cannot become an
unauthenticated row generator, the other so it cannot quietly become theatre.
"""

from __future__ import annotations

import pytest

from app.api.demo import DEMO_MERCHANT, OFFERED
from app.models import Case, CaseEvent

pytestmark = pytest.mark.integration


def _simulate(client, reason: str):
    res = client.post("/api/demo/simulate", json={"error_reason": reason})
    assert res.status_code == 200, res.text
    return res.json()


def _history(client, case_id: str) -> dict:
    res = client.get(f"/actions/case/{case_id}")
    assert res.status_code == 200, res.text
    return res.json()


class TestTheDecisionComesFromTheTaxonomy:
    """Five reasons, five different outcomes, none of them hardcoded here."""

    def test_each_offered_reason_reaches_its_documented_cause(self, client):
        res = client.get("/api/demo/causes")
        assert res.status_code == 200
        offers = {o["error_reason"]: o for o in res.json()}
        assert set(offers) == {o["error_reason"] for o in OFFERED}
        # Every offer must classify. An offer the taxonomy cannot map would show
        # a reader "unclassified" and prove the opposite of the intended point.
        assert all(o["cause"] for o in offers.values())

    def test_a_risk_block_is_refused_outright(self, client):
        case_id = _simulate(client, "payment_risk_check_failed")["case_id"]
        data = _history(client, case_id)
        assert data["case"]["cause"] == "risk_blocked"
        assert data["case"]["status"] == "suppressed"
        assert data["actions"] == []
        assert any(e["event"] == "case.suppressed" for e in data["ledger"])

    def test_a_merchant_failure_alerts_the_merchant_and_contacts_nobody(self, client):
        case_id = _simulate(client, "international_transaction_not_allowed")["case_id"]
        data = _history(client, case_id)
        assert data["case"]["cause"] == "merchant_config"
        assert [a["type"] for a in data["actions"]] == ["merchant_alert"]

    def test_an_expired_card_is_never_answered_with_a_retry(self, client):
        """The thesis in one assertion. If this ever returns `retry`, the demo is
        arguing against the README on the same page."""
        case_id = _simulate(client, "card_expired")["case_id"]
        data = _history(client, case_id)
        assert data["case"]["cause"] == "card_expired"
        assert [a["type"] for a in data["actions"]] == ["method_switch_prompt"]

    def test_the_five_outcomes_are_actually_different(self, client):
        """The section's claim is that the decision *changes*. If four of five
        produced the same action a reader would learn nothing, and every
        individual assertion above would still pass."""
        outcomes = set()
        for offer in OFFERED:
            data = _history(client, _simulate(client, offer["error_reason"])["case_id"])
            actions = tuple(a["type"] for a in data["actions"])
            outcomes.add((data["case"]["cause"], actions or ("<refused>",)))
        assert len(outcomes) == len(OFFERED), outcomes


class TestItCannotBeAbused:
    def test_an_unlisted_reason_is_refused(self, client):
        res = client.post("/api/demo/simulate", json={"error_reason": "'; DROP TABLE cases--"})
        assert res.status_code == 400

    @pytest.mark.parametrize("amount", [0, 99, 10_000_001])
    def test_absurd_amounts_are_refused(self, client, amount):
        res = client.post(
            "/api/demo/simulate",
            json={"error_reason": "card_expired", "amount_paise": amount},
        )
        assert res.status_code == 400

    def test_it_never_creates_a_payment_link(self, client, db_session):
        """`live` is not a parameter, and passing one must not smuggle it in."""
        res = client.post(
            "/api/demo/simulate",
            json={"error_reason": "card_expired", "live": True},
        )
        assert res.status_code == 200
        data = _history(client, res.json()["case_id"])
        for action in data["actions"]:
            assert action.get("external_ref") in (None, ""), (
                "a demonstration created a real Razorpay object"
            )

    def test_everything_it_writes_is_attributable_to_the_demo(self, client, db_session):
        case_id = _simulate(client, "insufficient_funds")["case_id"]
        case = db_session.get(Case, case_id)
        assert case.merchant_id == DEMO_MERCHANT
        assert case.external_ref.startswith("pay_DEMO")

    def test_the_ledger_records_each_step_exactly_once(self, client, db_session):
        """An earlier version diagnosed twice -- `ingest_payment_failed` already
        diagnoses inline -- and appended a second `case.diagnosed` for a
        diagnosis that happened once. In an append-only ledger that is not a
        cosmetic duplicate; it is a false record of a second decision."""
        case_id = _simulate(client, "insufficient_funds")["case_id"]
        events = [
            e.event_type
            for e in db_session.query(CaseEvent).filter(CaseEvent.case_id == case_id)
        ]
        assert len(events) == len(set(events)), events
        assert events.count("case.diagnosed") == 1
