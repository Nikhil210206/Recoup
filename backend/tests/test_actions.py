"""Action tools, stopping rules, and the approval queue.

Everything here guards an externally visible effect. A bug in this file does not
produce a wrong number in a report -- it sends a stranger a payment link, twice.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from datetime import UTC, datetime, timedelta

import pytest

from app.allocator.budget import BudgetPolicy
from app.allocator.estimator import AmountOnly
from app.allocator.policy import Allocator
from app.models import Action, ActionStatus, Case, CaseStatus
from app.services import actions as action_tools
from app.services import live_allocator
from app.simulation.outcomes import ActionType

SECRET = "whsec_test_example"
pytestmark = pytest.mark.integration


def _allocator(budget: int = 100) -> Allocator:
    """Deliberately uses the constant estimator.

    The cause-rate estimator needs a generated dataset, and these tests are about
    gates and side effects, not ranking quality. Measured, the two are within 3%
    of each other anyway.
    """
    return Allocator(
        estimator=AmountOnly(),
        budget_policy=BudgetPolicy(max_contacts_per_customer=2, max_total_contacts=budget),
    )


def _webhook(client, payment_id: str, reason: str, source: str, amount: int, event_id: str):
    body = json.dumps({
        "entity": "event", "account_id": "acc_T", "event": "payment.failed",
        "payload": {"payment": {"entity": {
            "id": payment_id, "order_id": f"order_{payment_id}", "amount": amount,
            "currency": "INR", "status": "failed", "method": "card",
            "error_code": "BAD_REQUEST_ERROR", "error_source": source,
            "error_step": "payment_authorization", "error_reason": reason,
            "contact": "+919000000101", "email": "t@example.com",
            # Recent, so the recovery-window stopping rule does not fire.
            "created_at": int(time.time()) - 3600,
        }}},
    }).encode()
    return client.post("/webhooks/razorpay", content=body, headers={
        "X-Razorpay-Signature": hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest(),
        "X-Razorpay-Event-Id": event_id,
        "Content-Type": "application/json",
    })


def _case(db, external_ref: str) -> Case:
    return db.query(Case).filter(Case.external_ref == external_ref).one()


class TestIdempotency:
    def test_repeated_allocation_executes_once(self, client, db_session):
        """The core guarantee. A tick that runs every five minutes must not send
        a customer a payment link every five minutes."""
        _webhook(client, "pay_IDEM1", "card_expired", "customer", 249900, "e_idem1")

        first = live_allocator.allocate_and_execute(db_session, _allocator())
        assert first["executed"] == 1

        for _ in range(4):
            again = live_allocator.allocate_and_execute(db_session, _allocator())
            assert again["executed"] == 0

        actions = db_session.query(Action).all()
        assert len(actions) == 1
        assert actions[0].status == ActionStatus.EXECUTED

    def test_claiming_the_same_action_twice_is_refused(self, client, db_session):
        _webhook(client, "pay_IDEM2", "card_expired", "customer", 249900, "e_idem2")
        case = _case(db_session, "pay_IDEM2")

        first = action_tools.claim(db_session, case, ActionType.PAYMENT_LINK_SMS)
        second = action_tools.claim(db_session, case, ActionType.PAYMENT_LINK_SMS)

        assert not first.duplicate
        assert second.duplicate
        assert second.action.id == first.action.id

    def test_the_idempotency_key_is_the_documented_tuple(self, client, db_session):
        _webhook(client, "pay_IDEM3", "card_expired", "customer", 249900, "e_idem3")
        case = _case(db_session, "pay_IDEM3")
        result = action_tools.claim(db_session, case, ActionType.RETRY, attempt_no=2)
        assert result.action.idempotency_key == f"{case.id}:retry:2"


class TestStoppingRules:
    def test_a_case_awaiting_approval_is_not_re_planned(self, client, db_session):
        """Regression, and the nastiest bug of the day.

        Idempotency is keyed on (case, action_type, attempt_no) and each pass
        increments the attempt, so a case sitting in the approval queue got a
        *new* action every pass. Three passes put the same Rs 99,000 case in the
        queue three times, and a reviewer could have approved it three times and
        sent three payment links. The key was working exactly as designed and
        protecting nothing, because what repeated was the decision, not the
        attempt.
        """
        _webhook(client, "pay_STOP1", "card_expired", "customer", 9_900_000, "e_stop1")

        live_allocator.allocate_and_execute(db_session, _allocator())
        for _ in range(3):
            result = live_allocator.allocate_and_execute(db_session, _allocator())
            assert result["stopped"].get("action_already_pending") == 1

        pending = (
            db_session.query(Action)
            .filter(Action.status == ActionStatus.PENDING_APPROVAL)
            .all()
        )
        assert len(pending) == 1, "the same case was queued more than once"

    def test_a_recovered_case_is_left_alone(self, client, db_session):
        _webhook(client, "pay_STOP2", "card_expired", "customer", 249900, "e_stop2")
        case = _case(db_session, "pay_STOP2")
        case.status = CaseStatus.RECOVERED
        db_session.commit()

        rule = live_allocator.check_stopping_rules(db_session, case, datetime.now(UTC))
        assert rule is not None and rule.name == "terminal_status"

    def test_a_stale_case_is_left_alone(self, client, db_session):
        """Past the window the customer has bought elsewhere. Attributing a later
        payment to us would be generous to the point of dishonesty."""
        _webhook(client, "pay_STOP3", "card_expired", "customer", 249900, "e_stop3")
        case = _case(db_session, "pay_STOP3")
        case.detected_at = datetime.now(UTC) - timedelta(days=30)
        db_session.commit()

        rule = live_allocator.check_stopping_rules(db_session, case, datetime.now(UTC))
        assert rule is not None and rule.name == "past_recovery_window"

    def test_an_unclassified_case_is_never_acted_on(self, client, db_session):
        _webhook(client, "pay_STOP4", "some_unpublished_code", "customer", 249900, "e_stop4")
        case = _case(db_session, "pay_STOP4")
        assert case.status == CaseStatus.PENDING_DIAGNOSIS

        rule = live_allocator.check_stopping_rules(db_session, case, datetime.now(UTC))
        assert rule is not None and rule.name == "awaiting_diagnosis"

    def test_attempts_are_capped(self, client, db_session):
        _webhook(client, "pay_STOP5", "card_expired", "customer", 249900, "e_stop5")
        case = _case(db_session, "pay_STOP5")
        for i in range(live_allocator.MAX_ACTIONS_PER_CASE):
            result = action_tools.claim(
                db_session, case, ActionType.PAYMENT_LINK_SMS, attempt_no=i + 1
            )
            result.action.status = ActionStatus.EXECUTED
        db_session.commit()

        rule = live_allocator.check_stopping_rules(db_session, case, datetime.now(UTC))
        assert rule is not None and rule.name == "max_actions_reached"

    def test_the_ledger_is_not_spammed_by_a_repeating_stop(self, client, db_session):
        """An append-only ledger is only useful if what gets appended is a change.
        A case in the approval queue is examined on every tick, and logging
        'still pending' each time buried the events that mattered."""
        _webhook(client, "pay_STOP6", "card_expired", "customer", 9_900_000, "e_stop6")
        for _ in range(6):
            live_allocator.allocate_and_execute(db_session, _allocator())

        case = _case(db_session, "pay_STOP6")
        stops = [e for e in case.events if e.event_type == "case.stopped"]
        assert len(stops) == 1, f"{len(stops)} identical stop rows written"


class TestApprovalQueue:
    def test_a_large_amount_needs_a_human(self, client, db_session):
        _webhook(client, "pay_APP1", "card_expired", "customer", 9_900_000, "e_app1")
        result = live_allocator.allocate_and_execute(db_session, _allocator())
        assert result["queued_for_approval"] == 1
        assert result["executed"] == 0

        queued = db_session.query(Action).one()
        assert queued.status == ActionStatus.PENDING_APPROVAL
        assert "amount_above" in queued.approval_reason

    def test_a_small_amount_does_not(self, client, db_session):
        _webhook(client, "pay_APP2", "card_expired", "customer", 100_000, "e_app2")
        result = live_allocator.allocate_and_execute(db_session, _allocator())
        assert result["executed"] == 1
        assert result["queued_for_approval"] == 0

    def test_a_retry_never_needs_approval(self, client, db_session):
        """A retry touches the gateway, not the customer. Queueing it would stall
        the cheapest recovery path behind a human for no benefit."""
        _webhook(client, "pay_APP3", "bank_technical_error", "bank", 99_000_000, "e_app3")
        case = _case(db_session, "pay_APP3")
        assert action_tools.needs_human(case, ActionType.RETRY) is None
        assert action_tools.needs_human(case, ActionType.PAYMENT_LINK_SMS) is not None

    def test_low_cause_confidence_needs_a_human(self, client, db_session):
        """Acting confidently on a cause we are unsure of is how a system does
        the wrong *specific* thing."""
        _webhook(client, "pay_APP4", "card_expired", "customer", 100_000, "e_app4")
        case = _case(db_session, "pay_APP4")
        case.cause_confidence = 0.55
        db_session.commit()
        assert action_tools.needs_human(case, ActionType.PAYMENT_LINK_SMS) is not None

    def test_approval_requires_a_named_approver(self, client, db_session):
        _webhook(client, "pay_APP5", "card_expired", "customer", 9_900_000, "e_app5")
        live_allocator.allocate_and_execute(db_session, _allocator())
        action = db_session.query(Action).one()

        response = client.post(f"/actions/{action.id}/approve", json={"approved_by": "nikhil"})
        assert response.status_code == 200
        db_session.refresh(action)
        assert action.approved_by == "nikhil"
        assert action.approved_at is not None

    def test_approving_advances_the_case(self, client, db_session):
        """Leaving the case DIAGNOSED made an executed case indistinguishable
        from an unworked one in every status query."""
        _webhook(client, "pay_APP6", "card_expired", "customer", 9_900_000, "e_app6")
        live_allocator.allocate_and_execute(db_session, _allocator())
        action = db_session.query(Action).one()
        client.post(f"/actions/{action.id}/approve", json={"approved_by": "nikhil"})

        case = _case(db_session, "pay_APP6")
        db_session.refresh(case)
        assert case.status == CaseStatus.ACTED

    def test_rejection_is_terminal(self, client, db_session):
        """The idempotency key stays claimed, so nothing can quietly re-propose
        the same action later and get a different answer from a different
        reviewer."""
        _webhook(client, "pay_APP7", "card_expired", "customer", 9_900_000, "e_app7")
        live_allocator.allocate_and_execute(db_session, _allocator())
        action = db_session.query(Action).one()

        assert client.post(
            f"/actions/{action.id}/reject",
            json={"rejected_by": "nikhil", "reason": "customer already contacted offline"},
        ).status_code == 200

        db_session.refresh(action)
        assert action.status == ActionStatus.REJECTED

        # A second decision on the same action must be refused.
        assert client.post(
            f"/actions/{action.id}/approve", json={"approved_by": "someone_else"}
        ).status_code == 409

    def test_an_unknown_action_is_a_404(self, client):
        assert client.post(
            "/actions/00000000-0000-0000-0000-000000000000/approve",
            json={"approved_by": "nikhil"},
        ).status_code == 404


class TestSuppression:
    def test_a_fraud_block_is_suppressed_with_a_reason(self, client, db_session):
        _webhook(client, "pay_SUP1", "payment_risk_check_failed", "issuer", 249900, "e_sup1")
        live_allocator.allocate_and_execute(db_session, _allocator())

        case = _case(db_session, "pay_SUP1")
        assert case.status == CaseStatus.SUPPRESSED
        assert db_session.query(Action).count() == 0
        suppressions = [e for e in case.events if e.event_type == "case.suppressed"]
        assert suppressions and suppressions[0].payload["rule"] == "risk_suppression"

    def test_a_merchant_config_failure_alerts_the_merchant_and_nobody_else(
        self, client, db_session
    ):
        """The day-0 discovery, closed properly.

        No customer action can clear a merchant setting, so every customer-facing
        option has expected value zero and non-zero cost. But there *is* a correct
        action: tell the merchant. For a while the rule was named
        `merchant_alert_only` and emitted no alert -- the case was silently
        suppressed and the one recoverable thing about it was lost.
        """
        _webhook(
            client, "pay_SUP2", "international_transaction_not_allowed", "business",
            349900, "e_sup2",
        )
        live_allocator.allocate_and_execute(db_session, _allocator())

        actions = db_session.query(Action).all()
        assert len(actions) == 1
        assert actions[0].action_type == str(ActionType.MERCHANT_ALERT)
        assert actions[0].status == ActionStatus.EXECUTED
        # A merchant alert costs nothing and consumes no contact budget: it goes
        # to the merchant, not to a customer.
        assert actions[0].cost_paise == 0

    def test_a_merchant_alert_does_not_spend_the_contact_budget(self, client, db_session):
        _webhook(
            client, "pay_SUP3", "international_transaction_not_allowed", "business",
            349900, "e_sup3",
        )
        result = live_allocator.allocate_and_execute(db_session, _allocator())
        assert result["budget"]["total_contacts"] == 0


class TestExecutionSafety:
    def test_payment_links_are_recorded_without_being_sent_by_default(
        self, client, db_session
    ):
        """`live=False` is the default everywhere. An allocation pass that creates
        real payment links every time it runs is one nobody can safely run
        twice."""
        _webhook(client, "pay_EXEC1", "card_expired", "customer", 249900, "e_exec1")
        live_allocator.allocate_and_execute(db_session, _allocator(), live=False)

        action = db_session.query(Action).one()
        assert action.status == ActionStatus.EXECUTED
        assert action.external_ref is None  # nothing was created at Razorpay

    def test_an_already_executed_action_is_not_re_executed(self, client, db_session):
        _webhook(client, "pay_EXEC2", "card_expired", "customer", 249900, "e_exec2")
        case = _case(db_session, "pay_EXEC2")
        claimed = action_tools.claim(db_session, case, ActionType.RETRY)
        action_tools.schedule_retry(db_session, claimed.action, case, delay_h=1.0)

        with pytest.raises(action_tools.ActionRefused):
            action_tools.schedule_retry(db_session, claimed.action, case, delay_h=1.0)

    def test_every_executed_action_leaves_a_ledger_entry(self, client, db_session):
        """If an action happened there is a row, and if there is a row one of
        these functions produced it."""
        _webhook(client, "pay_EXEC3", "card_expired", "customer", 249900, "e_exec3")
        live_allocator.allocate_and_execute(db_session, _allocator())

        case = _case(db_session, "pay_EXEC3")
        events = [e.event_type for e in case.events]
        assert "action.claimed" in events
        assert "action.executed" in events


class TestQuietHours:
    """Quiet hours are a fact about when a person is asleep.

    They were evaluated against the server's UTC clock while the policy was
    written in IST, which misclassified 10 of 24 hours: a message could be sent
    at 01:00 IST because UTC read 19:30, and blocked at 13:00 IST because UTC
    read 07:30. A customer-protection rule enforced in the wrong timezone
    protects nobody and blocks the wrong sends.
    """

    def test_quiet_hours_are_evaluated_in_ist(self):
        from app.allocator.budget import BudgetPolicy

        policy = BudgetPolicy(quiet_hours_start=21, quiet_hours_end=9)
        # 19:30 UTC is 01:00 IST -- the middle of the night.
        assert policy.is_quiet_at(datetime(2026, 8, 24, 19, 30, tzinfo=UTC))
        # 07:30 UTC is 13:00 IST -- the middle of the afternoon.
        assert not policy.is_quiet_at(datetime(2026, 8, 24, 7, 30, tzinfo=UTC))

    def test_a_deferred_send_lands_outside_the_quiet_window(self):
        from app.allocator.budget import IST, BudgetPolicy

        policy = BudgetPolicy(quiet_hours_start=21, quiet_hours_end=9)
        middle_of_night = datetime(2026, 8, 24, 19, 30, tzinfo=UTC)  # 01:00 IST
        allowed = policy.next_allowed_time(middle_of_night)

        assert not policy.is_quiet_at(allowed)
        assert allowed > middle_of_night
        assert allowed.astimezone(IST).hour == policy.quiet_hours_end

    def test_a_send_already_outside_quiet_hours_is_not_moved(self):
        from app.allocator.budget import BudgetPolicy

        policy = BudgetPolicy()
        daytime = datetime(2026, 8, 24, 7, 30, tzinfo=UTC)  # 13:00 IST
        assert policy.next_allowed_time(daytime) == daytime


class TestRejectionIsTerminal:
    def test_a_rejected_case_is_not_re_proposed(self, client, db_session):
        """A reviewer said no. Asking again on the next tick until they say yes
        is not a workflow, it is attrition.

        Before this, rejection made the *action* terminal while the case stayed
        DIAGNOSED, so the next pass proposed a fresh action at attempt_no+1.
        Three reject cycles produced three rejected actions for one case.
        """
        _webhook(client, "pay_REJ_T", "card_expired", "customer", 9_900_000, "e_rej_t")

        live_allocator.allocate_and_execute(db_session, _allocator())
        action = db_session.query(Action).one()
        client.post(
            f"/actions/{action.id}/reject",
            json={"rejected_by": "nikhil", "reason": "handled offline"},
        )

        for _ in range(3):
            result = live_allocator.allocate_and_execute(db_session, _allocator())
            assert result["stopped"].get("human_rejected") == 1

        assert db_session.query(Action).count() == 1


class TestLiveExecution:
    """The `live=True` branch -- the only code that creates something at Razorpay.

    Exercised against a stubbed client rather than the real API. A test suite
    that creates real payment links on every run eventually messages a stranger,
    and the genuine end-to-end call belongs in the recorded demo, not in CI.
    """

    def _stub_client(self, monkeypatch, captured: dict):
        class _Links:
            @staticmethod
            def create(payload):
                captured.update(payload)
                return {"id": "plink_STUB123", "short_url": "https://rzp.io/x/stub"}

        class _Client:
            payment_link = _Links()

        monkeypatch.setattr(action_tools, "_razorpay_client", lambda: _Client())

    def test_a_live_send_records_the_razorpay_reference(
        self, client, db_session, monkeypatch
    ):
        captured: dict = {}
        self._stub_client(monkeypatch, captured)

        _webhook(client, "pay_LIVE1", "card_expired", "customer", 249900, "e_live1")
        case = _case(db_session, "pay_LIVE1")
        claimed = action_tools.claim(db_session, case, ActionType.PAYMENT_LINK_SMS)
        result = action_tools.send_payment_link(
            db_session, claimed.action, case, live=True
        )

        assert result.performed
        assert claimed.action.external_ref == "plink_STUB123"
        assert claimed.action.status == ActionStatus.EXECUTED

    def test_a_live_send_never_asks_razorpay_to_notify_anyone(
        self, client, db_session, monkeypatch
    ):
        """Recoup decides *whether* to contact someone. It does not get to have
        Razorpay SMS a real phone number as a side effect."""
        captured: dict = {}
        self._stub_client(monkeypatch, captured)

        _webhook(client, "pay_LIVE2", "card_expired", "customer", 249900, "e_live2")
        case = _case(db_session, "pay_LIVE2")
        claimed = action_tools.claim(db_session, case, ActionType.PAYMENT_LINK_SMS)
        action_tools.send_payment_link(db_session, claimed.action, case, live=True)

        assert captured["notify"] == {"sms": False, "email": False}
        assert captured["reminder_enable"] is False

    def test_a_live_send_carries_ids_back_for_reconciliation(
        self, client, db_session, monkeypatch
    ):
        captured: dict = {}
        self._stub_client(monkeypatch, captured)

        _webhook(client, "pay_LIVE3", "card_expired", "customer", 249900, "e_live3")
        case = _case(db_session, "pay_LIVE3")
        claimed = action_tools.claim(db_session, case, ActionType.PAYMENT_LINK_SMS)
        action_tools.send_payment_link(db_session, claimed.action, case, live=True)

        assert captured["notes"]["recoup_case_id"] == case.id
        assert captured["amount"] == case.amount_paise

    def test_a_gateway_failure_is_recorded_not_raised(
        self, client, db_session, monkeypatch
    ):
        """A Razorpay outage must not take down an allocation pass. The action is
        marked failed and the ledger says why."""
        class _Boom:
            payment_link = type(
                "L", (), {"create": staticmethod(lambda payload: (_ for _ in ()).throw(
                    RuntimeError("razorpay is down")))}
            )()

        monkeypatch.setattr(action_tools, "_razorpay_client", lambda: _Boom())

        _webhook(client, "pay_LIVE4", "card_expired", "customer", 249900, "e_live4")
        case = _case(db_session, "pay_LIVE4")
        claimed = action_tools.claim(db_session, case, ActionType.PAYMENT_LINK_SMS)
        result = action_tools.send_payment_link(
            db_session, claimed.action, case, live=True
        )

        assert not result.performed
        assert claimed.action.status == ActionStatus.FAILED
        assert any(e.event_type == "action.failed" for e in case.events)

    def test_missing_razorpay_keys_fail_the_action_not_the_batch(
        self, client, db_session, monkeypatch
    ):
        """A missing API key and a gateway outage are different problems, but
        from an allocation pass's point of view both mean "this did not happen".
        One misconfigured deployment must not take down a batch of two thousand
        cases, so it is recorded like any other failure."""
        from app.config import Settings

        monkeypatch.setattr(
            "app.services.actions.get_settings",
            lambda: Settings(razorpay_key_id="", razorpay_key_secret=""),
        )
        _webhook(client, "pay_LIVE5", "card_expired", "customer", 249900, "e_live5")
        case = _case(db_session, "pay_LIVE5")
        claimed = action_tools.claim(db_session, case, ActionType.PAYMENT_LINK_SMS)

        result = action_tools.send_payment_link(
            db_session, claimed.action, case, live=True
        )
        assert not result.performed
        assert claimed.action.status == ActionStatus.FAILED
        assert any(e.event_type == "action.failed" for e in case.events)


class TestLiveHistory:
    def test_the_live_path_computes_real_customer_history(self, client, db_session):
        """The live path used to hand the feature builder zeros for every history
        field. Nothing broke, because the estimator in use ignores them -- but the
        model was measured on real history and would have run on placeholders."""
        for i in range(3):
            _webhook(client, f"pay_HIST{i}", "card_expired", "customer", 249900, f"e_hist{i}")

        cases = live_allocator.open_cases(db_session)
        frame = live_allocator.to_frame(cases, db_session)
        assert (frame.customer_prior_failures > 0).any()
        assert frame.has_prior_history.any()

    def test_hour_of_day_is_recorded_in_ist(self, client, db_session):
        """The feature means 'what time was it where the customer is'. The store
        holds UTC."""
        from app.allocator.budget import IST

        _webhook(client, "pay_TZ1", "card_expired", "customer", 249900, "e_tz1")
        cases = live_allocator.open_cases(db_session)
        frame = live_allocator.to_frame(cases, db_session)

        case = cases[0]
        detected = case.detected_at
        if detected.tzinfo is None:
            detected = detected.replace(tzinfo=UTC)
        assert int(frame.iloc[0].hour_ist) == detected.astimezone(IST).hour


class TestFailedActionsDoNotEarnCredit:
    """The worst bug this project produced, pinned.

    A payment link failed to create. The case was marked `ACTED` anyway. The
    customer later paid through a different route, `resolve_recovery` saw an
    `ACTED` case, and the system credited itself with a recovery it had done
    nothing to cause.

    Every other guard here exists to prevent exactly that -- the counterfactual
    attribution, the self-recovery distinction, the unnecessary-contact metric --
    and one unconditional status assignment on the live path undid all of them.
    """

    def _break_razorpay(self, monkeypatch):
        class _Broken:
            payment_link = type(
                "L", (), {"create": staticmethod(
                    lambda payload: (_ for _ in ()).throw(RuntimeError("boom"))
                )}
            )()

        monkeypatch.setattr(action_tools, "_razorpay_client", lambda: _Broken())

    def test_a_failed_action_leaves_the_case_unworked(
        self, client, db_session, monkeypatch
    ):
        self._break_razorpay(monkeypatch)
        _webhook(client, "pay_FAILEX1", "card_expired", "customer", 249900, "e_failex1")

        result = live_allocator.allocate_and_execute(
            db_session, _allocator(), live=True
        )
        assert result["executed"] == 0
        assert result["failed_to_execute"] == 1

        case = _case(db_session, "pay_FAILEX1")
        db_session.refresh(case)
        assert case.status != CaseStatus.ACTED, (
            "a case whose action failed was marked as acted on"
        )

    def test_a_payment_after_a_failed_action_is_a_self_recovery(
        self, client, db_session, monkeypatch
    ):
        """The end-to-end version. If we did nothing, the money is not ours to
        claim, however convenient the timing."""
        from app.services.ingest import resolve_recovery

        self._break_razorpay(monkeypatch)
        _webhook(client, "pay_FAILEX2", "card_expired", "customer", 249900, "e_failex2")
        live_allocator.allocate_and_execute(db_session, _allocator(), live=True)

        case = _case(db_session, "pay_FAILEX2")
        resolve_recovery(db_session, {
            "event": "payment.captured",
            "payload": {"payment": {"entity": {
                "id": "pay_LATER", "order_id": case.order_ref, "amount": 249900,
            }}},
        })
        db_session.commit()
        db_session.refresh(case)

        assert case.status == CaseStatus.CLOSED_UNRECOVERED
        events = [e.event_type for e in case.events]
        assert "case.self_recovered" in events
        assert "case.recovered" not in events

    def test_the_failure_is_recorded_in_the_ledger(
        self, client, db_session, monkeypatch
    ):
        self._break_razorpay(monkeypatch)
        _webhook(client, "pay_FAILEX3", "card_expired", "customer", 249900, "e_failex3")
        live_allocator.allocate_and_execute(db_session, _allocator(), live=True)

        case = _case(db_session, "pay_FAILEX3")
        events = [e.event_type for e in case.events]
        assert "action.failed" in events
        assert "case.action_did_not_execute" in events


class TestRazorpayFieldLimits:
    def test_reference_id_fits_razorpays_limit(self, client, db_session, monkeypatch):
        """Razorpay caps `reference_id` at 40 characters. "recoup_" plus a
        36-character UUID is 43, so every live link creation failed -- and the
        one run that worked only did so because the server was still executing
        older code that never sent the field.
        """
        captured: dict = {}

        class _Links:
            @staticmethod
            def create(payload):
                captured.update(payload)
                return {"id": "plink_OK", "short_url": "https://rzp.io/x/ok"}

        monkeypatch.setattr(
            action_tools, "_razorpay_client",
            lambda: type("C", (), {"payment_link": _Links()})(),
        )

        _webhook(client, "pay_REFLEN", "card_expired", "customer", 249900, "e_reflen")
        case = _case(db_session, "pay_REFLEN")
        claimed = action_tools.claim(db_session, case, ActionType.PAYMENT_LINK_SMS)
        action_tools.send_payment_link(db_session, claimed.action, case, live=True)

        assert len(captured["reference_id"]) <= 40
        # Still has to be unique per case, or two cases collide in Razorpay.
        assert case.id.replace("-", "")[:20] in captured["reference_id"]
