"""Invariants of the outcome simulator and episode runner.

The simulator is the substrate every headline number is measured through. If it
is wrong, nothing downstream is worth reading, and the failure mode is silent:
plausible numbers, wrong conclusions.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.simulation import assumptions as A
from app.simulation.episode import run_episode, self_recovery_time
from app.simulation.generator import GeneratorConfig, generate
from app.simulation.outcomes import (
    CONTACT_ACTIONS,
    ActionType,
    CaseDraw,
    _timing_factor,
    simulate_action,
    simulate_no_action,
)
from app.simulation.policies import (
    BASELINES,
    ORACLE,
    CauseAware,
    ContactEverything,
    DoNothing,
    RazorpayT3,
)


@pytest.fixture(scope="module")
def cases() -> pd.DataFrame:
    return generate(GeneratorConfig(seed=42, days=30))["cases"]


class TestCommonRandomNumbers:
    def test_draws_are_stable_for_a_case(self, cases):
        cid = str(cases.iloc[0].case_id)
        assert CaseDraw.for_case(cid) == CaseDraw.for_case(cid)

    def test_draws_differ_between_cases(self, cases):
        a = CaseDraw.for_case(str(cases.iloc[0].case_id))
        b = CaseDraw.for_case(str(cases.iloc[1].case_id))
        assert a != b

    def test_arms_are_compared_on_identical_draws(self, cases):
        """The point of common random numbers: two policies face the same
        customer with the same latent willingness, so a difference between them
        is the policy and not the dice."""
        row = cases.iloc[0]
        before = CaseDraw.for_case(str(row.case_id))
        run_episode(row, ContactEverything())
        run_episode(row, RazorpayT3())
        assert CaseDraw.for_case(str(row.case_id)) == before


class TestAttribution:
    def test_do_nothing_never_claims_causation(self, cases):
        outs = [simulate_no_action(r) for _, r in cases.iterrows()]
        assert not any(o.caused_by_us for o in outs)
        assert not any(o.unnecessary_contact for o in outs)

    def test_recovered_is_not_the_same_as_caused(self, cases):
        """The distinction the whole project rests on. If these ever coincide,
        the counterfactual has stopped doing any work."""
        outs = [
            simulate_action(r, ActionType.PAYMENT_LINK_SMS, delay_h=2.0)
            for _, r in cases.iterrows()
        ]
        recovered = sum(o.recovered for o in outs)
        caused = sum(o.caused_by_us for o in outs)
        assert caused < recovered

    def test_causation_requires_no_pending_self_recovery(self, cases):
        outs = [
            simulate_action(r, ActionType.PAYMENT_LINK_SMS, delay_h=2.0)
            for _, r in cases.iterrows()
        ]
        assert not any(o.caused_by_us and o.would_have_self_recovered for o in outs)

    def test_merchant_config_can_never_be_recovered_by_contact(self, cases):
        """Structurally impossible, so any contact spend on these is pure loss.
        This is the case that punishes an agent which treats every failure as
        'chase the customer'."""
        mc = cases[cases.latent_cause == "merchant_config"]
        assert len(mc) > 0
        for _, r in mc.iterrows():
            o = simulate_action(r, ActionType.PAYMENT_LINK_WHATSAPP, delay_h=1.0)
            assert not o.caused_by_us


class TestTiming:
    def test_acting_at_the_right_moment_beats_acting_late(self):
        best = A.RECOVERABILITY["authentication_failed"].best_delay_h
        assert _timing_factor(best, best) > _timing_factor(24.0, best) * 10

    def test_slow_causes_are_not_helped_by_acting_immediately(self):
        best = A.RECOVERABILITY["insufficient_funds"].best_delay_h
        assert _timing_factor(best, best) > _timing_factor(0.1, best) * 10

    def test_timing_peaks_at_the_causes_best_delay(self):
        for cause, rec in A.RECOVERABILITY.items():
            if rec.best_delay_h <= 0:
                continue
            peak = _timing_factor(rec.best_delay_h, rec.best_delay_h)
            assert peak >= _timing_factor(rec.best_delay_h * 8, rec.best_delay_h), cause
            assert peak >= _timing_factor(rec.best_delay_h / 8, rec.best_delay_h), cause


class TestEpisodeRunner:
    def test_failed_action_does_not_end_the_episode(self, cases):
        """Regression. `simulate_action` used to report a *pending* self-recovery
        as the outcome of a failed action, so the episode ended and every
        subsequent contact the policy would have sent never fired. Contact-
        everything was scored as far cheaper and less spammy than it is.
        """
        # Cases where the customer WOULD self-recover, but only much later than
        # the policy's second scheduled action. The policy must still get there.
        late = [
            r
            for _, r in cases.iterrows()
            if (t := self_recovery_time(r)) is not None and t > 30.0
        ]
        assert late, "fixture has no late self-recovering cases to exercise this"

        reached_a_contact = 0
        for r in late:
            ep = run_episode(r, ContactEverything())
            # Every intermediate step must have failed; the episode may only end
            # on a success, on the policy stopping, or on the ladder running out.
            for s in ep.steps[:-1]:
                assert not s.outcome.recovered, "episode continued past a recovery"
            if any(s.action in CONTACT_ACTIONS for s in ep.steps):
                reached_a_contact += 1

        assert reached_a_contact > 0, (
            "no late self-recovering case ever reached a contact step; a failed "
            "early retry is ending the episode again"
        )

    def test_contact_everything_incurs_unnecessary_contacts(self, cases):
        """A policy that messages everyone must be charged for messaging people
        who were going to pay anyway. Zero here means the cost side is broken."""
        eps = [run_episode(r, ContactEverything()) for _, r in cases.iterrows()]
        assert sum(e.unnecessary_contacts for e in eps) > 0

    def test_do_nothing_takes_no_steps_and_costs_nothing(self, cases):
        eps = [run_episode(r, DoNothing()) for _, r in cases.iterrows()]
        assert all(not e.steps for e in eps)
        assert sum(e.cost_paise for e in eps) == 0
        assert sum(e.incremental_paise for e in eps) == 0

    def test_do_nothing_still_collects_self_recoveries(self, cases):
        """Money that arrives unprompted is money the merchant keeps. The floor
        is not zero, and every other arm must be measured against it."""
        eps = [run_episode(r, DoNothing()) for _, r in cases.iterrows()]
        assert sum(e.amount_recovered_paise for e in eps) > 0

    def test_razorpay_t3_makes_exactly_three_attempts(self, cases):
        eps = [run_episode(r, RazorpayT3()) for _, r in cases.iterrows()]
        assert max(len(e.steps) for e in eps) <= 3

    def test_cause_aware_never_contacts_a_suppressed_cause(self, cases):
        """`merchant_config` and `risk_blocked` must receive zero customer
        contact: one is the merchant's own setting, the other must not be worked
        around."""
        suppressed = cases[cases.latent_cause.isin(["merchant_config", "risk_blocked"])]
        assert len(suppressed) > 0
        for _, r in suppressed.iterrows():
            ep = run_episode(r, CauseAware())
            assert all(s.action not in CONTACT_ACTIONS for s in ep.steps)


class TestArmComparison:
    def test_every_arm_beats_doing_nothing_on_gross_recovery(self, cases):
        results = {}
        for pol in BASELINES:
            eps = [run_episode(r, pol) for _, r in cases.iterrows()]
            results[pol.name] = sum(e.amount_recovered_paise for e in eps)
        floor = results["B0_do_nothing"]
        for name, gross in results.items():
            assert gross >= floor, name

    def test_cause_aware_is_more_contact_efficient(self, cases):
        """Knowing the cause buys most of the recovery for a fraction of the
        customer contact.

        The threshold here was 3x when the dataset had one loss channel, and the
        measured margin was 8.5x. Adding abandoned checkouts cut it to ~2.6x, and
        that is the honest number rather than a regression: an abandoned checkout
        has no payment to re-attempt, so the cause-aware policy must spend a
        contact on a channel where it has no cheap option. Part of its former
        edge came from a world where retrying was always available, which was a
        less realistic world.
        """
        eff = {}
        for pol in (ContactEverything(), CauseAware()):
            eps = [run_episode(r, pol) for _, r in cases.iterrows()]
            contacts = sum(e.contacts_used for e in eps)
            eff[pol.name] = sum(e.incremental_paise for e in eps) / max(contacts, 1)
        assert eff["B4_cause_aware"] > eff["B2_contact_everything"] * 2


class TestGroundTruthBoundary:
    """The counterfactual boundary, enforced structurally rather than by care.

    A policy that reads `latent_*` is not a policy -- it is the answer key. Every
    number produced by such an arm is worthless, and it is the first thing a
    reviewer would grep for.
    """

    def test_no_policy_reads_a_latent_field(self):
        import ast
        import inspect

        from app.simulation import policies

        source = inspect.getsource(policies)
        tree = ast.parse(source)

        offenders: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            # The oracle is exempt by design: it exists to define the ceiling and
            # is never ranked against the arms.
            if getattr(policies, node.name, None) is not None and getattr(
                getattr(policies, node.name), "reads_ground_truth", False
            ):
                continue
            for sub in ast.walk(node):
                is_latent_literal = (
                    isinstance(sub, ast.Constant)
                    and isinstance(sub.value, str)
                    and sub.value.startswith("latent_")
                )
                if is_latent_literal:
                    offenders.append(f"{node.name} reads {sub.value!r}")

        assert not offenders, "policies must not read ground truth: " + "; ".join(offenders)

    def test_cause_aware_classifies_from_observable_fields(self, cases):
        """It must reach the same cause the generator used, via Razorpay's error
        fields alone. If it cannot, the arm is either cheating or broken."""
        from app import taxonomy

        row = cases.iloc[0]
        got = taxonomy.classify(row.error_reason, row.error_source, row.error_step)
        assert got.cause == row.latent_cause

    def test_cause_aware_refuses_to_guess_an_unmapped_reason(self, cases):
        """An error code Razorpay has not published must go to the exception
        list, not to an invented action."""
        row = cases.iloc[0].copy()
        row["error_reason"] = "some_future_reason_2027"
        row["error_source"] = "customer"
        assert CauseAware().next_action(row, []) is None


class TestOracleCeiling:
    def test_oracle_is_a_genuine_upper_bound(self, cases):
        """If any arm exceeds the ceiling, the ceiling is not a ceiling and
        '% of ceiling' is meaningless. An earlier single-shot oracle was beaten
        by a four-step baseline at 126%."""
        from app.simulation.arms import evaluate
        from app.simulation.policies import ORACLE

        ceiling = evaluate(cases, ORACLE, 0.55)
        for pol in BASELINES:
            arm = evaluate(cases, pol, 0.55)
            assert arm.incremental_paise <= ceiling.incremental_paise, (
                f"{arm.arm} exceeded the oracle ceiling"
            )

    def test_ceiling_leaves_headroom(self, cases):
        """If every arm sat at the ceiling there would be nothing for the
        allocator to win, and the project would have no thesis."""
        from app.simulation.arms import evaluate
        from app.simulation.policies import ContactEverything

        ceiling = evaluate(cases, ORACLE, 0.55)
        best = evaluate(cases, ContactEverything(), 0.55)
        assert best.incremental_paise < ceiling.incremental_paise * 0.95


class TestContactAccounting:
    def test_action_after_payment_costs_nothing(self, cases):
        """A real system reads payment status before sending. Charging for a
        message to someone who already paid penalises contact-heavy arms for
        messages they would never send -- 6.4% of all charged contacts, before
        this was fixed."""
        from app.simulation.episode import run_episode

        eps = [run_episode(r, ContactEverything()) for _, r in cases.iterrows()]
        for e in eps:
            for s in e.steps:
                if s.outcome.reason == "self_recovered_before_action":
                    assert s.outcome.contacts_used == 0
                    assert s.outcome.cost_paise == 0
                    assert not s.outcome.unnecessary_contact

    def test_unnecessary_means_contacted_someone_who_would_have_paid(self, cases):
        """Every unnecessary contact must correspond to a real self-recovery.
        Otherwise the metric is measuring something else."""
        outs = [
            simulate_action(r, ActionType.PAYMENT_LINK_SMS, delay_h=0.5)
            for _, r in cases.iterrows()
        ]
        for o in outs:
            if o.unnecessary_contact:
                assert o.would_have_self_recovered

    def test_draw_streams_are_indexed_independently(self):
        """Retries and contacts consume separate random streams. Clamping the
        contact index against the retry array's length worked only because both
        happened to be the same size."""
        d = CaseDraw.for_case("boundary-check")
        assert len(d.u_retry) == len(d.u_nudge)
        assert d.u_retry != d.u_nudge


class TestPairedComparison:
    def test_arm_compared_with_itself_shows_no_difference(self, cases):
        """A sanity check on the statistics: the same policy against itself must
        produce a zero difference and an interval containing zero."""
        from app.simulation.arms import evaluate, paired_bootstrap

        a = evaluate(cases, RazorpayT3(), 0.55)
        b = evaluate(cases, RazorpayT3(), 0.55)
        st = paired_bootstrap(a, b, n_boot=500)
        assert st["mean_diff_rs"] == 0.0
        assert not st["significant"]

    def test_beating_do_nothing_is_significant(self, cases):
        from app.simulation.arms import evaluate, paired_bootstrap
        from app.simulation.policies import DoNothing

        st = paired_bootstrap(
            evaluate(cases, ContactEverything(), 0.55),
            evaluate(cases, DoNothing(), 0.55),
            n_boot=500,
        )
        assert st["significant"] and st["mean_diff_rs"] > 0
