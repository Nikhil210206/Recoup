"""Properties the synthetic dataset must hold.

These are not "does the code run" tests. Each one guards a claim the evaluation
later depends on, and each would silently invalidate results if it broke.
"""

from __future__ import annotations

import pytest

from app import taxonomy
from app.simulation import assumptions as A
from app.simulation.generator import CAUSE_TO_RAZORPAY, GeneratorConfig, generate


@pytest.fixture(scope="module")
def dataset():
    """Full-size dataset. Only the size and sanity tests need this."""
    return generate(GeneratorConfig(seed=A.DEFAULT_SEED))


def small(**kw) -> GeneratorConfig:
    """A short window, for properties that hold at any scale.

    Determinism and the world sweep do not depend on dataset size, and
    regenerating 12k cases four times turns a fast check into a slow one.
    """
    return GeneratorConfig(days=20, **kw)


@pytest.fixture(scope="module")
def cases(dataset):
    return dataset["cases"]


class TestDistributions:
    def test_all_mixes_are_probability_distributions(self):
        for method, mix in A.CAUSE_MIX_BY_METHOD.items():
            assert abs(sum(mix.values()) - 1.0) < 1e-9, method
            assert all(v >= 0 for v in mix.values()), method

    def test_method_mix_sums_to_one(self):
        assert abs(sum(p.value for p in A.METHOD_MIX.values()) - 1.0) < 1e-9

    def test_every_cause_in_a_mix_has_recoverability_and_a_razorpay_reason(self):
        """A cause that can be generated but has no recovery semantics, or no
        Razorpay reason to emit, would crash the generator or silently skew the
        outcome model."""
        for mix in A.CAUSE_MIX_BY_METHOD.values():
            for cause in mix:
                assert cause in A.RECOVERABILITY, cause
                assert cause in CAUSE_TO_RAZORPAY, cause
                assert cause in taxonomy.CAUSES, cause


class TestSizeAndSplit:
    def test_meets_minimum_case_count(self, cases):
        assert len(cases) >= 8_000

    def test_calibration_holdout_is_large_enough(self, cases):
        """Calibration on a few hundred cases produces noisy reliability curves,
        which is exactly the thing the project claims to do carefully."""
        assert (cases.split == "calibration").sum() >= 2_000

    def test_split_is_chronological_not_random(self, cases):
        """A random split lets a customer's later behaviour inform a prediction
        about their earlier failure. Every metric inflates and none of it
        survives production."""
        bounds = cases.groupby("split").failed_at.agg(["min", "max"])
        assert bounds.loc["train", "max"] <= bounds.loc["calibration", "min"]
        assert bounds.loc["calibration", "max"] <= bounds.loc["test", "min"]


class TestDeterminism:
    def test_same_seed_produces_identical_data(self):
        a = generate(small(seed=7))["cases"]
        b = generate(small(seed=7))["cases"]
        assert a.equals(b)

    def test_different_seed_produces_different_data(self):
        a = generate(small(seed=7))["cases"]
        b = generate(small(seed=8))["cases"]
        assert not a.equals(b)


class TestClassifierRoundTrip:
    def test_every_generated_reason_classifies_back_to_its_cause(self, cases):
        """The single most important property here. The generator emits
        Razorpay's real error reasons, and the same deterministic map that
        classifies a live webhook must recover the cause exactly. If this drifts,
        the classifier is being scored against a taxonomy built to flatter it."""
        got = cases.apply(
            lambda r: taxonomy.classify(r.error_reason, r.error_source, r.error_step).cause,
            axis=1,
        )
        assert (got == cases.latent_cause).all()


class TestLatentBoundary:
    def test_latent_fields_are_clearly_namespaced(self, cases):
        """Anything the agent must not see is prefixed `latent_`. The prefix is
        the enforcement mechanism for the counterfactual boundary."""
        assert {"latent_p_self_recover", "latent_p_retry", "latent_p_nudge"} <= set(cases.columns)

    def test_merchant_config_is_structurally_unrecoverable(self, cases):
        """No customer action can clear a merchant setting. Both customer-facing
        recovery probabilities must be exactly zero, so any agent that contacts
        these cases provably spends money for an impossible return."""
        mc = cases[cases.latent_cause == "merchant_config"]
        assert len(mc) > 0, "merchant_config never generated; the trap case is missing"
        assert (mc.latent_p_retry == 0).all()
        assert (mc.latent_p_nudge == 0).all()
        assert (mc.error_source == "business").all()

    def test_probabilities_are_in_range(self, cases):
        for col in ("latent_p_self_recover", "latent_p_retry", "latent_p_nudge"):
            assert cases[col].between(0, 1).all(), col


class TestExternalSanity:
    def test_best_case_recovery_stays_within_a_plausible_band(self, cases):
        """Razorpay publishes 'up to 20%' recovery for its own product. A
        simulator implying 70% would be modelling a world that does not exist,
        and every downstream comparison would be meaningless."""
        best = cases[["latent_p_retry", "latent_p_nudge"]].max(axis=1)
        incremental = (best - cases.latent_p_self_recover).clip(lower=0)
        by_value = (incremental * cases.amount_paise).sum() / cases.amount_paise.sum()
        assert 0.05 < by_value < 0.45, f"implied ceiling {by_value:.1%} is implausible"


class TestWorlds:
    def test_worlds_change_recoverability_but_not_which_payments_failed(self):
        """The world sweep must vary one thing. If the failures themselves
        changed too, a difference in results could not be attributed."""
        base = generate(small(seed=42, world="base"))["cases"]
        pess = generate(small(seed=42, world="pessimistic"))["cases"]

        assert base.payment_id.equals(pess.payment_id)
        assert base.latent_cause.equals(pess.latent_cause)
        assert pess.latent_p_retry.mean() < base.latent_p_retry.mean()


class TestNoLookAhead:
    """Regressions for bugs found in the day-1 review.

    All four shipped green: every test passed and every distribution summed to
    1.0 while the features were quietly wrong. They are pinned here because the
    no-leakage claim is load-bearing for the whole evaluation.
    """

    def test_customer_history_is_monotonic_in_time(self, cases):
        """History must accumulate in chronological order, not processing order.

        The original loop ran day -> merchant -> transaction, so a customer active
        at two merchants on the same day could carry "prior" history containing
        events that happen later in wall-clock time. 41% of customers are
        multi-merchant, so this contaminated a large fraction of the training
        features rather than a handful of edge cases.
        """
        ordered = cases.sort_values("failed_at")
        for _, grp in ordered.groupby("customer_id"):
            assert grp.customer_prior_attempts.is_monotonic_increasing
            assert grp.customer_prior_failures.is_monotonic_increasing

    def test_no_negative_time_gaps(self, cases):
        gaps = cases.hours_since_last_failure.dropna()
        assert (gaps >= 0).all(), "time ran backwards between a customer's failures"

    def test_missing_history_is_nan_not_a_sentinel(self, cases):
        """A model reading -1 as a duration treats 'never failed before' as
        'failed an hour in the future'."""
        h = cases.hours_since_last_failure
        assert (h.dropna() != -1).all()
        assert h.isna().sum() == (~cases.has_prior_failure).sum()

    def test_prior_features_exclude_the_current_attempt(self, cases):
        """`prior_*` must describe what was knowable strictly before this
        failure. Mixing the current attempt into one field but not another makes
        the features internally inconsistent."""
        first = cases[cases.customer_prior_attempts == 0]
        assert (first.customer_prior_failures == 0).all()
        assert first.customer_observed_success_rate.isna().all()

    def test_weekend_uplift_lands_on_saturday_and_sunday(self, cases):
        """An earlier version rotated the weekday encoding by one and applied the
        weekend uplift to Friday and Saturday."""
        flagged = set(cases[cases.is_weekend].failed_at.dt.day_name().unique())
        assert flagged == {"Saturday", "Sunday"}

        per_day = cases.groupby(cases.failed_at.dt.day_name()).size()
        days_in_window = cases.failed_at.dt.normalize().drop_duplicates()
        n_per_name = days_in_window.dt.day_name().value_counts()
        rate = per_day / n_per_name
        weekday_rate = rate[["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]].mean()
        weekend_rate = rate[["Saturday", "Sunday"]].mean()
        assert weekend_rate > weekday_rate * 1.05

    def test_no_timestamp_straddles_a_split_boundary(self, cases):
        """Cases sharing one instant must not land on both sides of an
        ostensibly clean chronological division."""
        assert (cases.groupby("failed_at").split.nunique() == 1).all()

        bounds = cases.groupby("split").failed_at.agg(["min", "max"])
        assert bounds.loc["train", "max"] < bounds.loc["calibration", "min"]
        assert bounds.loc["calibration", "max"] < bounds.loc["test", "min"]


class TestIdentity:
    def test_ids_are_unique(self, cases):
        """Ids are content hashes, so collisions are possible in principle and
        would silently merge two distinct failures into one case."""
        for col in ("case_id", "payment_id", "order_id"):
            assert not cases[col].duplicated().any(), col
