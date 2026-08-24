"""The allocator: safety gates, budget arithmetic, and the component ablation.

Most of these pin *negative* results. Three components in this allocator
contribute nothing measurable, and they are kept for stated reasons rather than
quietly deleted or quietly credited. A test that fails when a useless component
starts appearing useful is as valuable as one that fails when a useful component
breaks.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app import taxonomy
from app.allocator.budget import BudgetLedger, BudgetPolicy
from app.allocator.estimator import AmountOnly, CauseRate
from app.allocator.policy import Allocator, preferred_action, summarise
from app.model.train import build_frames
from app.simulation.arms import load
from app.simulation.outcomes import CONTACT_ACTIONS, ActionType
from app.simulation.policies import observable_cause


@pytest.fixture(scope="module")
def cases():
    c = load("base", 42)
    return c[c.split == "test"].reset_index(drop=True)


@pytest.fixture(scope="module")
def estimator():
    return CauseRate().fit(build_frames("base", 42)["train"])


@pytest.fixture(scope="module")
def plan(cases, estimator):
    allocator = Allocator(
        estimator=estimator, budget_policy=BudgetPolicy(max_contacts_per_customer=2)
    )
    return allocator.plan(cases)


class TestBudgetArithmetic:
    def test_contact_cap_is_enforced_per_customer(self):
        ledger = BudgetLedger(policy=BudgetPolicy(max_contacts_per_customer=2))
        now = datetime(2026, 8, 26, 12, tzinfo=UTC)
        for _ in range(2):
            assert ledger.can_contact("cust_A", now)[0]
            ledger.spend("cust_A", now)
        allowed, why = ledger.can_contact("cust_A", now)
        assert not allowed
        assert why == "customer_contact_cap"

    def test_the_cap_is_a_rolling_window_not_a_lifetime(self):
        from datetime import timedelta

        ledger = BudgetLedger(policy=BudgetPolicy(max_contacts_per_customer=1, window_days=7))
        t0 = datetime(2026, 8, 1, 12, tzinfo=UTC)
        ledger.spend("cust_A", t0)
        assert not ledger.can_contact("cust_A", t0 + timedelta(days=1))[0]
        assert ledger.can_contact("cust_A", t0 + timedelta(days=8))[0]

    def test_global_budget_is_enforced(self):
        ledger = BudgetLedger(policy=BudgetPolicy(max_total_contacts=3))
        now = datetime(2026, 8, 26, 12, tzinfo=UTC)
        for i in range(3):
            assert ledger.can_contact(f"cust_{i}", now)[0]
            ledger.spend(f"cust_{i}", now)
        assert ledger.can_contact("cust_new", now)[1] == "global_budget_exhausted"

    def test_quiet_hours_wrap_midnight(self):
        policy = BudgetPolicy(quiet_hours_start=21, quiet_hours_end=9)
        assert policy.in_quiet_hours(23)
        assert policy.in_quiet_hours(3)
        assert not policy.in_quiet_hours(12)
        assert policy.next_allowed_hour(3) == 9


class TestSafetyGates:
    def test_a_fraud_block_is_never_acted_on(self, plan, cases):
        """There is no correct action. Working around a fraud control is not a
        recovery, whatever it would be worth."""
        decisions, _ = plan
        by_case = {d.case_id: d for d in decisions}
        blocked = cases[cases.latent_cause == "risk_blocked"]
        assert len(blocked) > 0
        for case_id in blocked.case_id:
            assert not by_case[str(case_id)].acted

    def test_a_merchant_misconfiguration_alerts_the_merchant_and_nobody_else(
        self, plan, cases
    ):
        """No customer action can clear a merchant setting -- but there is still a
        correct output, and it is not silence. The rule was called
        `merchant_alert_only` while emitting no alert at all, which threw away the
        one recoverable thing about the case.
        """
        decisions, _ = plan
        by_case = {d.case_id: d for d in decisions}
        misconfigured = cases[cases.latent_cause == "merchant_config"]
        assert len(misconfigured) > 0
        for case_id in misconfigured.case_id:
            decision = by_case[str(case_id)]
            assert decision.action == ActionType.MERCHANT_ALERT
            # Goes to the merchant, so it spends none of the customer's patience.
            assert not decision.uses_contact
            assert decision.cost_paise == 0

    def test_no_contact_where_the_taxonomy_forbids_it(self, plan, cases):
        """Messaging a customer about their bank's downtime blames them for an
        outage they cannot affect."""
        decisions, _ = plan
        by_case = {d.case_id: d for d in decisions}
        for _, row in cases.iterrows():
            cause_key = observable_cause(row)
            if cause_key is None or taxonomy.get(cause_key).contact_ok:
                continue
            d = by_case[str(row.case_id)]
            assert d.action not in CONTACT_ACTIONS

    def test_abandoned_checkouts_are_never_retried(self, plan, cases):
        decisions, _ = plan
        by_case = {d.case_id: d for d in decisions}
        for case_id in cases[cases.channel == "abandoned_checkout"].case_id:
            assert by_case[str(case_id)].action != ActionType.RETRY

    def test_every_decision_records_a_reason_and_a_rule(self, plan):
        """A suppression is a decision. 'Nothing happened' needs a reason as much
        as an action does, or the audit trail cannot answer why a case was left
        alone."""
        decisions, _ = plan
        for d in decisions:
            assert d.reason, d.case_id
            assert d.rule, d.case_id

    def test_blocked_attempts_are_counted(self, plan):
        """'Zero policy violations' is true by construction and proves nothing.
        The number that shows the gate is load-bearing is how often it refused."""
        _, ledger = plan
        assert sum(ledger.blocked.values()) > 0


class TestActionSelection:
    def test_taxonomy_picks_the_action_not_the_model(self, cases):
        """The fitted uplift model, asked to choose an action, picked the
        true-best one on 3 of 11 causes -- it always chose a payment link,
        because that is right on average and the per-cause interaction is a
        second-order effect inside a ROC-0.62 signal. The taxonomy encodes the
        answer directly."""
        instrument_cases = cases[cases.latent_cause.isin(["card_expired", "invalid_instrument"])]
        assert len(instrument_cases) > 0
        for _, row in instrument_cases.iterrows():
            cause = taxonomy.get(observable_cause(row))
            assert preferred_action(row, cause) == ActionType.METHOD_SWITCH_PROMPT

    def test_bank_outages_get_a_retry_and_no_message(self, cases):
        outages = cases[cases.latent_cause == "transient_bank_downtime"]
        assert len(outages) > 0
        for _, row in outages.head(50).iterrows():
            cause = taxonomy.get(observable_cause(row))
            assert preferred_action(row, cause) == ActionType.RETRY

    def test_free_retries_are_not_rationed(self, plan):
        """A retry touches the gateway, not the customer. The EV floor exists to
        ration contacts, and applying it to retries suppressed hundreds of free
        actions -- which cost more value than every other gate combined."""
        decisions, _ = plan
        retries = [d for d in decisions if d.action == ActionType.RETRY]
        assert retries
        assert all(d.rule == "unrationed" for d in retries)


class TestComponentAblation:
    """Pins which components measurably contribute, including the ones that
    do not."""

    def _run(self, cases, estimator, cap=2, floor=5_000, budget=600):
        from app.allocator.bake_off import run_bake_off

        allocator = Allocator(
            estimator=estimator,
            budget_policy=BudgetPolicy(max_contacts_per_customer=cap),
            min_ev_paise=floor,
        )
        return run_bake_off(cases, allocator, {}, budget)[0].result

    def test_value_ordering_beats_arrival_order(self, cases, estimator):
        """The allocator's real contribution. A per-case agent works its queue in
        arrival order; ordering the same budget by value is worth roughly +20%.

        This corrects a claim from the model work. 'Ranking is worth ~1%' compared
        EV-ranking against AMOUNT-ranking -- both value-aware. Against arrival
        order, ordering by value matters a great deal.
        """
        from app.allocator.bake_off import run_bake_off
        from app.simulation.policies import CauseAware

        allocator = Allocator(
            estimator=estimator, budget_policy=BudgetPolicy(max_contacts_per_customer=2)
        )
        results = {
            r.result.arm: r.result
            for r in run_bake_off(cases, allocator, {"B4_cause_aware": CauseAware()}, 600)
        }
        allocated = results["RECOUP_allocator"].incremental_paise
        arrival = results["B4_cause_aware"].incremental_paise
        assert allocated > arrival * 1.10

    def test_the_ml_estimator_does_not_earn_its_place(self, cases, estimator):
        """Kept as a test rather than a comment because it is a claim about the
        system that could stop being true. A cause-rate group-by tracks true
        uplift better (+0.362) than the fitted per-case model (+0.275): the
        structure here is per-cause, and a per-case model adds variance without
        adding signal."""
        from pathlib import Path

        import joblib

        from app.allocator.estimator import ModelUplift

        path = Path(__file__).resolve().parents[2] / "artifacts" / "uplift.base.seed42.joblib"
        if not path.exists():
            pytest.skip("uplift model artifact not built; run `make model`")

        simple = self._run(cases, estimator)
        learned = self._run(cases, ModelUplift(joblib.load(path)))
        assert simple.incremental_paise > learned.incremental_paise

    def test_a_group_by_matches_no_estimate_at_all(self, cases, estimator):
        """Honest null result: with ranking worth ~1% over amount, the estimator
        cannot move the outcome much. It is retained because the EV floor and the
        merchant-facing forecast both need a magnitude, not because it improves
        the allocation."""
        with_estimate = self._run(cases, estimator)
        without = self._run(cases, AmountOnly())
        ratio = with_estimate.incremental_paise / without.incremental_paise
        assert 0.97 < ratio < 1.03

    def test_the_contact_cap_costs_almost_nothing(self, cases, estimator):
        """The collision guard is a customer-protection feature, not a revenue
        feature. It should cost close to nothing -- if it ever costs a lot, the
        trade-off deserves re-examining rather than silent acceptance."""
        capped = self._run(cases, estimator, cap=2)
        uncapped = self._run(cases, estimator, cap=99)
        assert capped.incremental_paise >= uncapped.incremental_paise * 0.97


class TestSummary:
    def test_summary_accounts_for_every_case(self, plan):
        decisions, _ = plan
        s = summarise(decisions)
        assert s["acted"] + s["suppressed"] == s["cases"]
        assert s["contacts"] + s["free_retries"] == s["acted"]


class TestHonestDecomposition:
    """Pins where the value actually comes from, including where it does not.

    Two harness bugs made an earlier version of this comparison flattering:
    contact-everything was executed for a single step (its first rung is a retry,
    so it was scored on one WhatsApp with its retry discarded), and every
    baseline pooled all three loss channels into one queue -- which had already
    solved the cross-agent collision problem for free, making the control a
    strawman *against* the allocator's own case.
    """

    def _controls(self, cases, estimator, budget=600):
        from app.allocator.bake_off import ranked, run_bake_off
        from app.allocator.siloed import run_siloed
        from app.simulation.policies import CauseAware

        allocator = Allocator(
            estimator=estimator, budget_policy=BudgetPolicy(max_contacts_per_customer=2)
        )
        alloc = run_bake_off(cases, allocator, {}, budget)[0].result
        arrival = next(
            r.result
            for r in run_bake_off(
                cases,
                Allocator(
                    estimator=estimator,
                    budget_policy=BudgetPolicy(max_contacts_per_customer=99),
                ),
                {"arrival": CauseAware()},
                budget,
            )
            if r.result.arm == "arrival"
        )
        pooled = next(
            r.result
            for r in run_bake_off(
                ranked(cases),
                Allocator(
                    estimator=estimator,
                    budget_policy=BudgetPolicy(max_contacts_per_customer=99),
                ),
                {"pooled": CauseAware()},
                budget,
            )
            if r.result.arm == "pooled"
        )
        siloed, stats = run_siloed(cases, CauseAware, budget)
        return alloc, arrival, pooled, siloed, stats

    def test_value_ordering_is_the_real_ranking_win(self, cases, estimator):
        """Ordering a finite budget by value, rather than working a queue in
        arrival order, is worth roughly +18%. This is the claim that survives."""
        _, arrival, pooled, _, _ = self._controls(cases, estimator)
        lift = (pooled.incremental_paise - arrival.incremental_paise) / arrival.incremental_paise
        assert lift > 0.10

    def test_the_allocator_does_not_beat_a_ranked_pooled_agent(self, cases, estimator):
        """The negative result, pinned deliberately.

        Against an idealised single agent handed an already-sorted queue, the
        allocator's shared budget, contact cap, EV floor and uplift estimator net
        out at roughly zero. If this ever starts passing as a *win*, something has
        changed and the claim needs re-deriving rather than quietly upgrading.
        """
        alloc, _, pooled, _, _ = self._controls(cases, estimator)
        ratio = alloc.incremental_paise / pooled.incremental_paise
        assert 0.95 < ratio < 1.05, (
            f"allocator vs pooled+ranked is {ratio:.3f}; it was ~0.987 when measured, "
            "and a large move in either direction means the decomposition is stale"
        )

    def test_the_allocator_roughly_matches_siloed_agents_on_revenue(self, cases, estimator):
        """Against the architecture that actually exists -- one agent per loss
        channel, none aware of the others -- the allocator is within a couple of
        percent. Its case is governance, not revenue."""
        alloc, _, _, siloed, _ = self._controls(cases, estimator)
        ratio = alloc.incremental_paise / siloed.incremental_paise
        assert 0.95 < ratio < 1.10

    def test_the_collision_guard_binds_rarely_in_this_data(self, cases, estimator):
        """Only ~9.5% of customers have cases in more than one channel, so the
        cross-agent collision this layer prevents is uncommon here. Recorded so
        the governance claim is stated at its true size."""
        _, _, _, _, stats = self._controls(cases, estimator)
        collisions = stats["collisions"]
        assert collisions["contacted_by_multiple_channels"] < (
            0.1 * collisions["customers_contacted"]
        )

    def test_the_allocator_never_exceeds_its_contact_cap(self, cases, estimator):
        """What the governance layer does guarantee, unconditionally: no customer
        is spent more than the cap, across every channel at once. Siloed agents
        cannot make this promise because none of them can see the others."""
        from app.allocator.siloed import contact_collisions, plan_siloed
        from app.simulation.policies import CauseAware

        allocator = Allocator(
            estimator=estimator,
            budget_policy=BudgetPolicy(max_contacts_per_customer=2, max_total_contacts=600),
        )
        _, ledger = allocator.plan(cases)
        assert ledger.summary()["max_contacts_to_one_customer"] <= 2

        plans, _ = plan_siloed(cases, CauseAware, 600)
        assert contact_collisions(cases, plans)["max_contacts_to_one_customer"] > 2
