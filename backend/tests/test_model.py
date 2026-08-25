"""Uplift model, calibration, and the lever comparison.

The tests that matter here are not "does it fit". They are the ones that would
catch the model quietly becoming untrustworthy: leakage into the features, a
calibrator that makes probabilities worse being applied anyway, and the lever
result silently flipping.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd
import pytest

from app.evaluation import BUDGET_FRACTION
from app.model import calibration as cal
from app.model import features as feat
from app.model.dataset import CONTROL_SHARE, build_training_frame, split_arms
from app.model.levers import compare
from app.model.miscalibration import allocate_by_ev, distort
from app.model.uplift import UpliftModel
from app.simulation.arms import load
from app.simulation.outcomes import ActionType


@pytest.fixture(scope="module")
def cases():
    return load("base", 42)


@pytest.fixture(scope="module")
def frames(cases):
    return {
        split: build_training_frame(cases[cases.split == split], seed=7)
        for split in ("train", "calibration", "test")
    }


@pytest.fixture(scope="module")
def fitted(frames):
    calib = frames["calibration"]
    mid = len(calib) // 2
    return UpliftModel(kind="logistic").fit(
        frames["train"], calib.iloc[mid:], calib.iloc[:mid]
    )


class TestNoLeakage:
    def test_features_module_never_references_latent_state(self):
        """The feature builder is where leakage would be most damaging and least
        visible: every downstream number would look better and none would be
        real."""
        tree = ast.parse(inspect.getsource(feat))
        offenders = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith("latent_")
        ]
        assert not offenders, f"features read ground truth: {offenders}"

    def test_built_features_contain_no_latent_columns(self, cases):
        built = feat.build(cases.head(200))
        assert not [c for c in built.columns if "latent" in c]

    def test_the_target_is_observable_in_production(self, frames):
        """`recovered` is what a live system sees. `caused_by_us` is not, and a
        model trained on it could never be retrained on real traffic."""
        frame = frames["train"]
        assert "recovered" in frame.columns
        assert "_truth_caused" in frame.columns  # retained for scoring only
        cat, num = feat.columns(treated=True)
        assert "_truth_caused" not in [*cat, *num]


class TestRandomisedExploration:
    def test_both_arms_are_populated(self, frames):
        control, treated = split_arms(frames["train"])
        assert len(control) > 200
        assert len(treated) > 1000

    def test_control_share_is_roughly_as_configured(self, frames):
        control, treated = split_arms(frames["train"])
        share = len(control) / (len(control) + len(treated))
        assert abs(share - CONTROL_SHARE) < 0.05

    def test_actions_are_spread_not_concentrated(self, frames):
        """A model trained on logs from one policy learns that policy's blind
        spots. Randomising is what makes the treated model's estimates valid off
        the policy that generated the data."""
        counts = frames["train"].action.value_counts(normalize=True)
        assert counts.drop("none", errors="ignore").max() < 0.45

    def test_treatment_raises_recovery(self, frames):
        control, treated = split_arms(frames["train"])
        assert treated.recovered.mean() > control.recovered.mean()


class TestCalibration:
    def test_ece_is_zero_for_a_perfectly_calibrated_model(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.05, 0.95, 20_000)
        y = (rng.random(20_000) < p).astype(int)
        assert cal.expected_calibration_error(y, p) < 0.02

    def test_ece_detects_systematic_over_confidence(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.05, 0.95, 20_000)
        y = (rng.random(20_000) < p).astype(int)
        assert cal.expected_calibration_error(y, np.clip(p + 0.15, 0, 1)) > 0.10

    def test_calibration_is_applied_only_when_it_helps(self, fitted):
        """Isotonic regression on a few hundred rows can add variance without
        removing bias. Applying it regardless -- because it is 'best practice' --
        would degrade the exact quantity the allocator spends from, while the
        write-up claimed an improvement."""
        for arm in (fitted.baseline, fitted.treated):
            assert arm.chosen_method in {"raw", "isotonic", "sigmoid"}
            chosen = arm.calibration_scores[arm.chosen_method].brier
            assert chosen == min(m.brier for m in arm.calibration_scores.values())

    def test_distortions_preserve_ranking_exactly(self):
        """The whole miscalibration argument depends on this: if a distortion
        changed the ranking, the experiment would be measuring two different
        models rather than two different price scales."""
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(0)
        p = rng.beta(2, 3, 5000)
        y = (rng.random(5000) < p).astype(int)
        base = roc_auc_score(y, p)
        for kind in ("overconfident", "underconfident", "compressed"):
            assert abs(roc_auc_score(y, distort(p, kind)) - base) < 1e-9

    def test_over_confidence_inflates_the_forecast(self):
        """The real cost of miscalibration here. A merchant plans against the
        forecast, and a product that promises double what it delivers stops
        being trusted however much it recovers."""
        amounts = np.full(500, 100_000.0)
        uplift = np.linspace(0.05, 0.6, 500)
        cost = np.full(500, 25.0)
        chosen = allocate_by_ev(amounts, uplift, cost, 100)
        honest = (amounts[chosen] * uplift[chosen]).sum()
        inflated = (amounts[chosen] * distort(uplift, "overconfident")[chosen]).sum()
        assert inflated > honest * 1.15


class TestUpliftModel:
    def test_uplift_is_never_negative(self, fitted, cases):
        """A negative uplift says acting makes recovery less likely, which is not
        a coherent basis for spending money. Not acting already expresses it."""
        built = feat.build(cases[cases.split == "test"].head(500))
        u = fitted.uplift(built, ActionType.PAYMENT_LINK_WHATSAPP, 2.0)
        assert (u >= 0).all()

    def test_best_action_returns_one_choice_per_case(self, fitted, cases):
        built = feat.build(cases[cases.split == "test"].head(300))
        candidates = [(ActionType.RETRY, 1.0), (ActionType.PAYMENT_LINK_WHATSAPP, 2.0)]
        u, actions, delays = fitted.best_action(built, candidates)
        assert len(u) == len(actions) == len(delays) == len(built)
        assert set(actions) <= {a for a, _ in candidates}

    def test_model_beats_chance_on_held_out_data(self, fitted, frames):
        test = frames["test"]
        treated = test[test.treated]
        m = cal.evaluate(treated.recovered.astype(int), fitted.treated.predict(treated))
        assert m.roc_auc > 0.55


class TestLevers:
    def test_action_selection_dominates_case_ranking(self, cases):
        """The finding that repointed the project.

        The original thesis was that ranking by expected value rather than
        transaction amount was the lever. Measured against an *oracle* that knows
        true uplift, ranking is worth about 1%, because corr(EV, amount) = 0.94 --
        amounts span ~5,000x and uplift ~5x, so the product barely reorders.

        Choosing the right action at the right time is worth several hundred
        percent, because the actions are not substitutes.

        This test exists so the claim cannot silently invert.
        """
        test = cases[cases.split == "test"].reset_index(drop=True)
        results = compare(test, budget=int(len(test) * BUDGET_FRACTION))

        action_lift = max(r.lift for r in results if r.lever == "action")
        ranking_lift = max(r.lift for r in results if r.lever == "ranking")

        assert action_lift > 1.0, "action selection should be worth >100%"
        assert ranking_lift < 0.15, "ranking should be worth <15% even with oracle uplift"
        assert action_lift > ranking_lift * 10

    def test_suppression_refuses_futile_cases(self, cases):
        """Where no customer action can help -- a fraud block, a merchant
        configuration -- the plan must be to do nothing."""
        from app.model.levers import cause_aware_plan

        test = cases[cases.split == "test"].reset_index(drop=True)
        actions, _ = cause_aware_plan(test)
        futile = test.latent_cause.isin(["risk_blocked", "merchant_config"]).to_numpy()
        assert futile.sum() > 0
        assert all(actions[i] is None for i in np.flatnonzero(futile))

    def test_abandoned_checkouts_are_never_planned_for_retry(self, cases):
        from app.model.levers import cause_aware_plan

        test = cases[cases.split == "test"].reset_index(drop=True)
        actions, _ = cause_aware_plan(test)
        for i in np.flatnonzero((test.channel == "abandoned_checkout").to_numpy()):
            assert actions[i] != ActionType.RETRY


class TestReproducibility:
    """The training pipeline must produce identical data on identical inputs.

    It did not. `build_frames` derived each split's exploration seed from
    `hash(split)`, and Python randomises string hashing per process unless
    PYTHONHASHSEED is pinned. The same command produced held-out ROC-AUC of
    0.588, 0.604, 0.606 and 0.629 on four consecutive runs, and the saved
    artifact recorded whichever ran last.

    Nothing failed. The evaluation simply was not reproducible while claiming to
    be, which is worse than an obviously broken one -- every number in the
    write-up would have been a number nobody could reproduce, including me.
    """

    def test_split_seeds_are_fixed_not_hashed(self):
        from app.model.train import SPLIT_SEED_OFFSET

        assert set(SPLIT_SEED_OFFSET) == {"train", "calibration", "test"}
        # Distinct, so each split explores with its own stream.
        assert len(set(SPLIT_SEED_OFFSET.values())) == 3

    def test_training_frames_are_identical_across_calls(self):
        from app.model.train import build_frames

        a = build_frames("base", 42)
        b = build_frames("base", 42)
        for split in a:
            pd.testing.assert_frame_equal(a[split], b[split])

    def test_different_exploration_seeds_produce_different_data(self, cases):
        """Determinism must not come from the explorer ignoring its seed."""
        train_cases = cases[cases.split == "train"]
        a = build_training_frame(train_cases, seed=1)
        b = build_training_frame(train_cases, seed=2)
        assert not a.action.equals(b.action)

    def test_fitted_model_is_deterministic(self, frames):
        calib = frames["calibration"]
        mid = len(calib) // 2
        preds = []
        for _ in range(2):
            model = UpliftModel(kind="logistic").fit(
                frames["train"], calib.iloc[mid:], calib.iloc[:mid]
            )
            test = frames["test"]
            preds.append(model.treated.predict(test[test.treated]))
        np.testing.assert_allclose(preds[0], preds[1])


class TestExperimentScoping:
    def test_sub_experiments_respect_the_requested_world(self):
        """`_bayes_ceiling`, `_lever_comparison` and `_miscalibration_experiment`
        each hardcoded ("base", 42). Running `--world pessimistic` trained on
        pessimistic data and then reported a ceiling and a lever study computed
        on base data -- a silent mismatch between the model and its own
        evaluation."""
        import inspect

        from app.model import train

        source = inspect.getsource(train)
        assert 'load_cases("base", 42)' not in source
        for name in ("_bayes_ceiling", "_lever_comparison", "_miscalibration_experiment"):
            fn = getattr(train, name)
            params = inspect.signature(fn).parameters
            assert "world" in params and "seed" in params, name

    def test_lever_finding_holds_in_every_world(self):
        """The conclusion that repointed the project must survive a different
        world, not just a different seed. Measured: action selection is worth
        +519% to +529% and ranking +0.9% to +1.1% across all three."""
        for world in ("base", "pessimistic", "optimistic"):
            cases = load(world, 42)
            test = cases[cases.split == "test"].reset_index(drop=True)
            results = compare(test, budget=int(len(test) * BUDGET_FRACTION))
            action = max(r.lift for r in results if r.lever == "action")
            ranking = max(r.lift for r in results if r.lever == "ranking")
            assert action > 1.0, world
            assert ranking < 0.15, world
