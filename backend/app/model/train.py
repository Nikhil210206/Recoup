"""Fit, calibrate and evaluate the uplift model.

    python -m app.model.train --world base --seed 42

Prints its configuration before any numbers, and states which split each number
came from. Three separate defects in this project were green checks measuring
something other than what I believed.

Splits, all chronological:

    train        fit both arm models
    calibration  split in half: model selection, then calibrator fitting
    test         evaluated once, at the end

Model selection happens on a slice the calibrator never sees, and the calibrator
is fitted on a slice the models were not trained on. Calibrating on training data
fits the calibrator to the same over-confidence it exists to remove.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.model import calibration as cal
from app.model.dataset import build_training_frame, summarise
from app.model.uplift import UpliftModel
from app.simulation.arms import load
from app.simulation.generator import WORLDS

ARTIFACTS = Path(__file__).resolve().parents[3] / "artifacts"


#: Fixed per-split offsets so each split explores with a different random
#: stream while the whole run stays reproducible.
#:
#: This was `hash(split)`, which is randomised per process in Python unless
#: PYTHONHASHSEED is pinned. Every run therefore trained on different data:
#: the same command produced ROC-AUC of 0.588, 0.604, 0.606 and 0.629 on four
#: consecutive runs, and the saved artifact recorded whichever one happened to
#: run last. Nothing failed, and a reproducibility claim was quietly false.
SPLIT_SEED_OFFSET = {"train": 0, "calibration": 101, "test": 202}


def build_frames(world: str, seed: int) -> dict[str, pd.DataFrame]:
    cases = load(world, seed)
    return {
        split: build_training_frame(
            cases[cases.split == split], seed=seed + offset
        )
        for split, offset in SPLIT_SEED_OFFSET.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the uplift model.")
    ap.add_argument("--world", choices=sorted(WORLDS), default="base")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", action="store_true", help="persist the fitted model")
    args = ap.parse_args()

    print("Recoup uplift model")
    print(f"  world : {args.world}")
    print(f"  seed  : {args.seed}")
    print("  SIMULATION BENCHMARK -- not production Razorpay data")
    print()

    frames = build_frames(args.world, args.seed)
    train, calib_all, test = frames["train"], frames["calibration"], frames["test"]

    # Chronological halves: selection first, calibration second.
    mid = len(calib_all) // 2
    select, calib = calib_all.iloc[:mid], calib_all.iloc[mid:]

    print("data")
    for name, frame in [("train", train), ("select", select), ("calib", calib), ("test", test)]:
        s = summarise(frame)
        print(
            f"  {name:8} n={s['n']:>6,}  control={s['control_n']:>5,}  "
            f"treated={s['treated_n']:>5,}  "
            f"control recovery={s['control_recovery_rate']:.1%}  "
            f"treated={s['treated_recovery_rate']:.1%}"
        )

    # --- model selection, on `select` -----------------------------------------
    print("\nmodel selection (fitted on train, scored on select -- never on test)")
    hdr = f"  {'model':26}{'ROC':>8}{'PR':>8}{'Brier':>9}{'ECE':>8}{'bias':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    chosen, best_brier = None, float("inf")
    for kind in ("logistic", "gbm"):
        candidate = UpliftModel(kind=kind).fit(train, calib, select)
        candidate.treated.evaluate(select[select.treated])
        m = candidate.treated.metrics_calibrated
        print("  " + m.row(f"{kind} (treated arm)"))
        if m.brier < best_brier:
            chosen, best_brier = kind, m.brier
    print(f"\n  selected: {chosen}  (lowest Brier, not highest AUC -- expected value")
    print("            is a price, and a well-ranked but biased model prices everything wrong)")

    # --- final fit and test evaluation ---------------------------------------
    model = UpliftModel(kind=chosen).fit(train, calib, select)
    model.baseline.evaluate(test[~test.treated])
    model.treated.evaluate(test[test.treated])

    print("\nheld-out test performance")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for arm in (model.baseline, model.treated):
        print("  " + arm.metrics_raw.row(f"{arm.name} raw"))
        print("  " + arm.metrics_calibrated.row(f"{arm.name} calibrated"))

    print("\nreliability, treated arm, BEFORE calibration")
    treated_test = test[test.treated]
    y_test = treated_test.recovered.astype(int)
    print(cal.render_reliability(y_test, model.treated.predict_raw(treated_test)))
    print("\nreliability, treated arm, AFTER calibration")
    print(cal.render_reliability(y_test, model.treated.predict(treated_test)))

    raw, calibrated = model.treated.metrics_raw, model.treated.metrics_calibrated
    print("\nwhat calibration changed")
    print(f"  ROC-AUC : {raw.roc_auc:.3f} -> {calibrated.roc_auc:.3f}   (ranking, barely moves)")
    print(f"  Brier   : {raw.brier:.4f} -> {calibrated.brier:.4f}")
    print(f"  ECE     : {raw.ece:.4f} -> {calibrated.ece:.4f}")
    print(f"  bias    : {raw.bias:+.3f} -> {calibrated.bias:+.3f}   (predicted minus observed)")
    print("  A ranking metric cannot see the difference. An allocator pricing")
    print("  cases in rupees can see nothing else.")

    print("\n  calibration chosen per arm (best of raw / isotonic / sigmoid,")
    print("  scored on a slice used for nothing else):")
    for arm in (model.baseline, model.treated):
        scores = "  ".join(
            f"{k}={m.brier:.4f}" for k, m in sorted(arm.calibration_scores.items())
        )
        print(f"      {arm.name:10} -> {arm.chosen_method:9}  ({scores})")
    print("  Calibration is applied when it earns its place. Isotonic on a few")
    print("  hundred rows can add variance without removing bias, and applying it")
    print("  regardless would degrade the exact quantity the allocator spends from.")

    _bayes_ceiling(test, model, args.world, args.seed)
    _lever_comparison(args.world, args.seed)
    _miscalibration_experiment(model, args.world, args.seed)

    if args.save:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        path = ARTIFACTS / f"uplift.{args.world}.seed{args.seed}.joblib"
        joblib.dump(model, path)
        meta = {
            "world": args.world, "seed": args.seed, "kind": chosen,
            "test_brier_calibrated": calibrated.brier,
            "test_ece_calibrated": calibrated.ece,
            "test_roc_auc": calibrated.roc_auc,
        }
        (ARTIFACTS / f"uplift.{args.world}.seed{args.seed}.json").write_text(
            json.dumps(meta, indent=2)
        )
        print(f"\n  saved -> {path.name}")


def _bayes_ceiling(
    test: pd.DataFrame, model: UpliftModel, world: str, seed: int
) -> None:
    """How much of the achievable signal the features actually capture.

    The outcome is a Bernoulli draw weighted by a latent probability, so no model
    can exceed the AUC implied by that probability itself. Reporting a raw 0.62
    invites the wrong conclusion in both directions -- it looks weak, and it gives
    no way to tell underfitting from irreducible noise.
    """
    from sklearn.metrics import roc_auc_score

    from app.simulation.arms import load as load_cases
    from app.simulation.outcomes import ActionType, _action_success_probability

    cases = load_cases(world, seed)
    cases = cases[cases.split == "test"].reset_index(drop=True)
    aligned = test.reset_index(drop=True)
    n = min(len(cases), len(aligned))
    cases, aligned = cases.iloc[:n], aligned.iloc[:n]

    # Verify the alignment rather than assume it. These two frames are produced
    # by separate code paths and joined positionally; if their order ever
    # diverges, every case would be scored against a different case's latent
    # probability and the ceiling would be quietly meaningless rather than wrong
    # in any visible way.
    if not (cases.case_id.to_numpy() == aligned.case_id.to_numpy()).all():
        raise AssertionError(
            "test cases and the training frame are misaligned; the Bayes ceiling "
            "would be computed against the wrong rows"
        )

    treated_mask = aligned.treated.to_numpy()
    if treated_mask.sum() < 50:
        return

    # The true generating probability, reconstructed exactly as the simulator
    # computed it -- including the timing kernel. An earlier version of this
    # dropped the timing term and produced a "ceiling" the model beat by 16%,
    # which is impossible and was the tell that the ceiling was wrong, not the
    # model right.
    p_self = cases.latent_p_self_recover.to_numpy()
    p_act = np.array([
        _action_success_probability(
            cases.iloc[i],
            ActionType(aligned.action.iloc[i]),
            float(np.expm1(aligned.log_delay_h.iloc[i])),
            0,
            0.55,
        )
        if aligned.action.iloc[i] != "none"
        else 0.0
        for i in range(n)
    ])
    true_p = p_act + (1 - p_act) * p_self
    y = aligned.recovered.astype(int).to_numpy()

    bayes = roc_auc_score(y[treated_mask], true_p[treated_mask])
    ours = model.treated.metrics_calibrated.roc_auc
    captured = (ours - 0.5) / (bayes - 0.5) if bayes > 0.5 else 0.0

    print("\nsignal ceiling")
    print(f"  Bayes-optimal ROC-AUC : {bayes:.3f}  (using the true generating probability)")
    print(f"  our model             : {ours:.3f}")
    print(f"  share of achievable   : {captured:.0%} of the signal above chance")
    print("  The remainder is irreducible: the outcome is a coin flip weighted by")
    print("  p, and no model can predict a coin flip. Reported alone, an AUC in the")
    print("  low 0.6s reads as a weak model; against this ceiling it reads as most")
    print("  of the signal that exists in the data.")


def _lever_comparison(world: str, seed: int) -> None:
    """Which decision actually recovers the money.

    Run before the calibration experiment because it determines what calibration
    is *for*. The project began by assuming case ranking was the lever and
    calibration mattered because expected value prices a ranking. The measurement
    says otherwise: ranking is worth ~1% even with oracle knowledge, and the
    action-and-timing choice is worth ~500%. Calibration still matters, but for
    deciding whether to act at all rather than for ordering a queue.
    """
    from app.model.levers import compare, render
    from app.simulation.arms import load as load_cases

    cases = load_cases(world, seed)
    cases = cases[cases.split == "test"].reset_index(drop=True)
    budget = int(len(cases) * 0.15)

    print(f"\nwhich lever recovers the money  (budget: {budget:,} contacts)")
    print("  'action' holds case selection fixed and varies what we do.")
    print("  'ranking' holds the action fixed and varies which cases we work,")
    print("  using ORACLE uplift -- so it is the ceiling for ranking, not our model.")
    print()
    print(render(compare(cases, budget)))
    print()
    print("  Ranking is worth ~1% even with perfect knowledge, because")
    print("  corr(EV, amount) = 0.94: amounts span ~5,000x and uplift ~5x, so the")
    print("  product barely reorders. The action choice is worth ~500%, because")
    print("  the actions are not substitutes -- an expired card responds to a")
    print("  method-switch roughly 20x better than to a retry.")


def _miscalibration_experiment(model: UpliftModel, world: str, seed: int) -> None:
    """Does miscalibration actually cost money, or is this ceremony?"""
    from app.model import features as feat
    from app.model.miscalibration import run_experiment
    from app.simulation.arms import load as load_cases
    from app.simulation.assumptions import ACTION_COST_PAISE
    from app.simulation.outcomes import ActionType

    cases = load_cases(world, seed)
    cases = cases[cases.split == "test"].reset_index(drop=True)

    features = feat.build(cases)
    candidates = [
        (ActionType.RETRY, 1.0),
        (ActionType.PAYMENT_LINK_WHATSAPP, 2.0),
        (ActionType.METHOD_SWITCH_PROMPT, 2.0),
    ]
    model_uplift, actions, _ = model.best_action(features, candidates)

    # Counterfactual truth: an action only adds value on top of what would have
    # happened anyway, so the incremental probability is p_action * (1 - p_self).
    p_self = cases.latent_p_self_recover.to_numpy()
    p_act = np.array([
        cases.latent_p_retry.iloc[i] if a == ActionType.RETRY
        else cases.latent_p_nudge.iloc[i]
        for i, a in enumerate(actions)
    ])
    true_uplift = p_act * (1 - p_self)

    amounts = cases.amount_paise.to_numpy().astype(float)
    cost = np.array([
        float(ACTION_COST_PAISE[str(a)].value) if str(a) in ACTION_COST_PAISE else 0.0
        for a in actions
    ])
    budget = int(len(cases) * 0.15)

    print(f"\nmiscalibration experiment  (budget: {budget:,} contacts over {len(cases):,} cases)")
    print("  Monotone distortions: the case ORDER is identical, so ROC-AUC and")
    print("  PR-AUC are unchanged by construction. Only the magnitudes move.")
    print()
    hdr = f"  {'distortion':16}{'selected':>10}{'realised Rs':>14}{'forecast Rs':>14}{'error':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    baseline_value = None
    for r in run_experiment(amounts, true_uplift, model_uplift, cost, budget):
        if r.label == "none":
            baseline_value = r.realised_paise
        print(
            f"  {r.label:16}{r.cases_selected:>10,}{r.realised_paise/100:>14,.0f}"
            f"{r.predicted_paise/100:>14,.0f}{r.forecast_error:>+8.0%}"
        )
    print()
    print("  Read the two right-hand columns, not the left one.")
    print()
    print("  Realised value barely moves. That follows from the lever result above:")
    print("  case ranking is worth ~1% here, so a distortion that only reorders the")
    print("  queue cannot cost much. I expected miscalibration to wreck the")
    print("  allocation; it does not, and the reason is the same arithmetic.")
    print()
    print("  The FORECAST is wrecked. An over-confident model claims Rs 49 lakh and")
    print("  delivers Rs 26 lakh -- a 90% overstatement, on identical ROC-AUC. That")
    print("  is not cosmetic: a merchant staffs and plans against the forecast, and")
    print("  a recovery product that habitually promises double what it delivers")
    print("  stops being trusted regardless of how much it actually recovers.")
    print()
    print("  So calibration earns its place for FORECASTING and for the decision of")
    print("  whether to act at all -- not for ordering a queue, which was the")
    print("  justification this project started with.")
    if baseline_value:
        print(f"\n  Calibrated allocation realised Rs {baseline_value/100:,.0f}.")


if __name__ == "__main__":
    main()
