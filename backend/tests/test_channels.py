"""Multi-channel loss ingestion and arbitration.

The product thesis is that nothing arbitrates between recovery agents competing
for one customer's tolerance. That claim is only testable if the channels
actually pull in different directions, so these tests pin the asymmetry rather
than just the plumbing.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from app.simulation import assumptions as A
from app.simulation.episode import run_episode
from app.simulation.generator import GeneratorConfig, generate
from app.simulation.outcomes import CONTACT_ACTIONS
from app.simulation.policies import CauseAware, ContactEverything, observable_cause

SECRET = "whsec_test_example"


@pytest.fixture(scope="module")
def cases():
    return generate(GeneratorConfig(seed=42, days=45))["cases"]


class TestChannelAsymmetry:
    def test_all_three_channels_are_generated(self, cases):
        assert set(cases.channel.unique()) == set(A.CHANNEL_MIX)

    def test_abandoned_checkout_cannot_be_retried(self, cases):
        """Nothing failed -- the customer left. There is no payment to
        re-attempt, so the only route to the money is a contact. This is what
        makes the channel a pure claim on the contact budget."""
        ac = cases[cases.channel == "abandoned_checkout"]
        assert len(ac) > 0
        assert (ac.latent_p_retry == 0).all()
        assert ac.error_reason.isna().all()
        assert (ac.latent_p_nudge > 0).mean() > 0.9

    def test_subscriptions_can_be_recovered_without_contacting_anyone(self, cases):
        """A live mandate means the merchant re-attempts the charge on its own.
        Recovery can cost zero contacts -- the opposite of an abandoned cart."""
        subs = cases[cases.channel == "failed_subscription"]
        payments = cases[cases.channel == "failed_payment"]
        assert len(subs) > 0
        assert subs.latent_p_retry.mean() > 0
        # Same causes, but the mandate makes retry strictly more effective.
        shared = set(subs.latent_cause) & set(payments.latent_cause)
        assert shared, "channels share no causes; the comparison is meaningless"

    def test_the_channels_disagree_about_what_recovery_costs(self, cases):
        """If every channel had the same contact economics there would be
        nothing to arbitrate and the allocator would have no job."""
        per_channel = {}
        for channel, grp in cases.groupby("channel"):
            eps = [run_episode(r, ContactEverything()) for _, r in grp.iterrows()]
            contacts = sum(e.contacts_used for e in eps)
            incremental = sum(e.incremental_paise for e in eps)
            per_channel[channel] = incremental / contacts if contacts else float("inf")

        best, worst = max(per_channel.values()), min(per_channel.values())
        assert best > worst * 1.5, f"channels have near-identical economics: {per_channel}"


class TestChannelAwarePolicy:
    def test_cause_alone_is_not_enough_to_act(self, cases):
        """`customer_abandoned` is retryable on a failed payment and not
        retryable on an abandoned checkout. A policy reading only the cause left
        the entire abandoned-checkout channel unrecovered while spending zero
        contacts -- which looked frugal in the totals rather than broken."""
        ac = cases[cases.channel == "abandoned_checkout"]
        acted = [run_episode(r, CauseAware()) for _, r in ac.head(200).iterrows()]
        assert any(
            s.action in CONTACT_ACTIONS for e in acted for s in e.steps
        ), "channel-aware policy never contacts an abandoned checkout"

    def test_cause_is_derivable_without_ground_truth_on_every_channel(self, cases):
        for _, row in cases.groupby("channel").head(30).iterrows():
            got = observable_cause(row)
            if got is not None:
                assert got == row.latent_cause


@pytest.mark.integration
class TestChannelIngestion:
    """Live adapters against the running app. Requires Postgres."""

    def _post(self, client, body: dict, event_id: str):
        raw = json.dumps(body).encode()
        return client.post(
            "/webhooks/razorpay",
            content=raw,
            headers={
                "X-Razorpay-Signature": hmac.new(
                    SECRET.encode(), raw, hashlib.sha256
                ).hexdigest(),
                "X-Razorpay-Event-Id": event_id,
                "Content-Type": "application/json",
            },
        )

    def test_subscription_pending_opens_a_case(self, client, db_session):
        from app.models import Case, LossChannel

        r = self._post(client, {
            "entity": "event", "account_id": "acc_T", "event": "subscription.pending",
            "payload": {
                "subscription": {"entity": {
                    "id": "sub_TEST01", "customer_id": "cust_T1", "auth_attempts": 1,
                    "current_start": 1755900000, "amount": 49900}},
                "payment": {"entity": {
                    "id": "pay_SUB01", "order_id": "order_SUB01", "amount": 49900,
                    "currency": "INR", "method": "card",
                    "error_reason": "insufficient_funds", "error_source": "customer",
                    "error_step": "payment_authorization"}},
            },
        }, "evt_SUB_PENDING_01")
        assert r.json()["handled"] is True

        case = db_session.query(Case).filter(Case.external_ref.like("sub_TEST01%")).one()
        assert case.channel == LossChannel.FAILED_SUBSCRIPTION
        assert case.cause == "insufficient_funds"

    def test_abandoned_cart_is_routed_by_payload_shape(self, client, db_session):
        """Razorpay documents the abandoned-cart payload but not a stable event
        name, so routing matches on shape rather than guessing a string."""
        from app.models import Case, LossChannel

        r = self._post(client, {
            "entity": "event", "account_id": "acc_T", "event": "some.undocumented.name",
            "payload": {"cart": {
                "cart_token": "cart_TEST01", "line_items_total": 189900,
                "currency": "INR", "phone": "+919000000123",
                "abandoned_checkout_url": "https://example.test/checkout/abc"}},
        }, "evt_CART_01")
        assert r.json()["handled"] is True

        case = db_session.query(Case).filter(Case.external_ref == "cart_TEST01").one()
        assert case.channel == LossChannel.ABANDONED_CHECKOUT
        # No classifier involved: nothing failed, so the cause is structural.
        assert case.cause == "customer_abandoned"
        assert case.error_reason is None

    def test_payment_captured_attributes_a_recovery(self, client, db_session):
        """The live-loop money shot, previously untested. A failed payment
        followed by a successful one on the same order must close the case."""
        from app.models import Case, CaseStatus

        self._post(client, {
            "entity": "event", "account_id": "acc_T", "event": "payment.failed",
            "payload": {"payment": {"entity": {
                "id": "pay_REC_FAIL", "order_id": "order_RECOVERY01", "amount": 250000,
                "currency": "INR", "method": "upi", "error_reason": "payment_timed_out",
                "error_source": "customer", "error_step": "payment_authorization",
                "created_at": 1755900000}}},
        }, "evt_REC_FAIL")

        case = db_session.query(Case).filter(Case.external_ref == "pay_REC_FAIL").one()
        case.status = CaseStatus.ACTED  # the allocator would have acted by now
        db_session.commit()

        self._post(client, {
            "entity": "event", "account_id": "acc_T", "event": "payment.captured",
            "payload": {"payment": {"entity": {
                "id": "pay_REC_OK", "order_id": "order_RECOVERY01", "amount": 250000,
                "currency": "INR", "method": "upi"}}},
        }, "evt_REC_OK")

        db_session.refresh(case)
        assert case.status == CaseStatus.RECOVERED

    def test_a_customer_who_pays_unprompted_is_not_claimed_as_a_recovery(
        self, client, db_session
    ):
        """If we never acted, the money is not ours to claim. Blurring this is
        exactly how recovery numbers get inflated."""
        from app.models import Case, CaseStatus

        self._post(client, {
            "entity": "event", "account_id": "acc_T", "event": "payment.failed",
            "payload": {"payment": {"entity": {
                "id": "pay_SELF_FAIL", "order_id": "order_SELF01", "amount": 120000,
                "currency": "INR", "method": "upi", "error_reason": "payment_timed_out",
                "error_source": "customer", "error_step": "payment_authorization",
                "created_at": 1755900000}}},
        }, "evt_SELF_FAIL")

        self._post(client, {
            "entity": "event", "account_id": "acc_T", "event": "payment.captured",
            "payload": {"payment": {"entity": {
                "id": "pay_SELF_OK", "order_id": "order_SELF01", "amount": 120000,
                "currency": "INR", "method": "upi"}}},
        }, "evt_SELF_OK")

        case = db_session.query(Case).filter(Case.external_ref == "pay_SELF_FAIL").one()
        assert case.status == CaseStatus.CLOSED_UNRECOVERED
        events = [e.event_type for e in case.events]
        assert "case.self_recovered" in events
