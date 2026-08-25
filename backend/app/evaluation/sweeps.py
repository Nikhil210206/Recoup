"""Robustness sweeps: where does this stop working?

Two different failures, deliberately separated.

**Sweep A -- a wrong world.** Re-runs everything under three parameterisations
of customer recoverability. Tests whether a conclusion survives the simulator
being wrong about how recoverable people are.

**Sweep B -- a wrong cause.** Corrupts the classification with increasing
probability and finds where the system stops beating the baselines.

Sweep B is the one that matters, and it took a while to see why. The obvious
robustness test was to degrade the *uplift estimate* -- but the measurements
already say ranking is worth about 1%, so degrading the thing that drives ranking
can only move the result by about 1%. It would have looked like a reassuring
result and meant nothing.

The system's entire claim rests on **action selection from the cause**, worth
roughly +520%. So the question worth asking is: what if the cause is wrong?
Razorpay's error codes are reliable, but an integration can mis-map them, a new
code can be silently mis-classified by the LLM tail, and the day-4 measurement
found a model confidently assigning a fraud block to a retryable decline. This
sweep prices that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.model.levers import cause_aware_plan, true_uplift
from app.simulation.generator import CAUSE_TO_RAZORPAY

#: How badly the cause is known, from perfect to coin-flip.
NOISE_LEVELS = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50)


@dataclass
class SweepPoint:
    label: str
    noise: float
    realised_paise: int
    contacts: int
    baseline_paise: int

    @property
    def lift(self) -> float:
        return (
            (self.realised_paise - self.baseline_paise) / self.baseline_paise
            if self.baseline_paise
            else 0.0
        )


def corrupt_causes(
    cases: pd.DataFrame, noise: float, seed: int = 0
) -> pd.DataFrame:
    """Swap a fraction of error codes for a different cause's code.

    Corrupting the *error code* rather than the cause label matters: the system
    is never handed a cause, it derives one. Rewriting the input is the only
    honest way to simulate a misclassification, because it exercises the same
    lookup the live path uses.
    """
    if noise <= 0:
        return cases

    rng = np.random.default_rng(seed)
    out = cases.copy()
    swappable = [c for c in CAUSE_TO_RAZORPAY if c != "merchant_config"]

    # Only rows that carry an error code can be misclassified. An abandoned
    # checkout has none -- its cause follows from the channel, and no amount of
    # classifier error changes that.
    eligible = out.index[out.error_reason.notna()].to_numpy()
    n_corrupt = int(len(eligible) * noise)
    if n_corrupt == 0:
        return out

    victims = rng.choice(eligible, size=n_corrupt, replace=False)
    for idx in victims:
        true_cause = out.at[idx, "latent_cause"]
        wrong = [c for c in swappable if c != true_cause]
        replacement = CAUSE_TO_RAZORPAY[wrong[int(rng.integers(0, len(wrong)))]]
        out.at[idx, "error_reason"] = replacement[0]
        out.at[idx, "error_source"] = replacement[1]
        out.at[idx, "error_step"] = replacement[2]
    return out


def _realised(cases: pd.DataFrame, planned: pd.DataFrame, budget: int) -> tuple[int, int]:
    """Score a plan made on `planned` against the truth in `cases`.

    The split is the point. The plan is chosen from possibly-corrupted inputs;
    the outcome is computed from the real latent parameters. That is exactly what
    a misclassification costs in production -- you act on what you believed, and
    reality settles on what was true.
    """
    actions, delays = cause_aware_plan(planned)
    uplift = true_uplift(cases, actions, delays)
    amounts = cases.amount_paise.to_numpy().astype(float)

    order = np.argsort(-amounts)
    chosen = np.zeros(len(cases), dtype=bool)
    spent = 0
    from app.simulation.outcomes import CONTACT_ACTIONS

    for i in order:
        action = actions[i]
        if action is None:
            continue
        if action in CONTACT_ACTIONS:
            if spent >= budget:
                continue
            spent += 1
        chosen[i] = True

    return int((amounts[chosen] * uplift[chosen]).sum()), spent


def cause_noise_sweep(
    cases: pd.DataFrame, budget: int, seed: int = 0
) -> tuple[list[SweepPoint], dict[str, int]]:
    """How much classification error the thesis tolerates.

    Measured against **two** baselines, because they answer different questions.

    *Fixed retry at 24h* is Razorpay's documented subscription policy and uses no
    cause at all. Against it the system wins by roughly 600% and keeps winning
    even at 50% classification error -- which is true, and not very informative.
    A wrong cause usually still produces a *contact*, and contacting beats
    retrying for most causes, so the comparison mostly measures "does it retry
    blindly" rather than "does it know the cause".

    *Always send a payment link* is the harder baseline: cause-blind, but not
    stupid. The gap above it is what knowing the cause is actually worth, and it
    is the number that should degrade as classification gets noisier.
    """
    from app.simulation.outcomes import ActionType

    amounts = cases.amount_paise.to_numpy().astype(float)
    order = np.argsort(-amounts)[:budget]

    def _baseline(action: ActionType, delay: float) -> int:
        uplift = true_uplift(cases, [action] * len(cases), np.full(len(cases), delay))
        return int((amounts[order] * uplift[order]).sum())

    baselines = {
        "fixed_retry_24h": _baseline(ActionType.RETRY, 24.0),
        "always_link_24h": _baseline(ActionType.PAYMENT_LINK_WHATSAPP, 24.0),
    }

    points = []
    for noise in NOISE_LEVELS:
        corrupted = corrupt_causes(cases, noise, seed=seed)
        realised, contacts = _realised(cases, corrupted, budget)
        points.append(
            SweepPoint(
                label=f"{noise:.0%} misclassified",
                noise=noise,
                realised_paise=realised,
                contacts=contacts,
                baseline_paise=baselines["always_link_24h"],
            )
        )
    return points, baselines


def crossover(points: list[SweepPoint]) -> float | None:
    """The noise level at which the advantage disappears, interpolated."""
    for previous, current in zip(points, points[1:], strict=False):
        if previous.lift > 0 >= current.lift:
            span = previous.lift - current.lift
            if span == 0:
                return current.noise
            fraction = previous.lift / span
            return previous.noise + fraction * (current.noise - previous.noise)
    return None


def render_cause_sweep(points: list[SweepPoint], baselines: dict[str, int]) -> str:
    lines = [
        "  Baselines, neither of which uses a cause:",
        f"    fixed retry @ 24h (Razorpay T+3) : "
        f"Rs {baselines['fixed_retry_24h'] / 100:>12,.0f}",
        f"    always send a link @ 24h         : "
        f"Rs {baselines['always_link_24h'] / 100:>12,.0f}",
        "",
        f"  {'cause error':22}{'realised Rs':>14}{'vs retry':>11}{'vs always-link':>16}",
        "  " + "-" * 63,
    ]
    retry_base = baselines["fixed_retry_24h"]
    for point in points:
        vs_retry = (
            (point.realised_paise - retry_base) / retry_base if retry_base else 0.0
        )
        lines.append(
            f"  {point.label:22}{point.realised_paise / 100:>14,.0f}"
            f"{vs_retry:>+10.0%}{point.lift:>+15.1%}"
        )

    point = crossover(points)
    lines.append("")
    if point is None:
        lines.append(
            "  The advantage over always-linking survives every level tested."
        )
    else:
        lines.append(
            f"  Crossover at roughly {point:.0%} classification error: past that, "
            "knowing\n  the cause badly is no better than not knowing it and always "
            "sending a link."
        )
    return "\n".join(lines)
