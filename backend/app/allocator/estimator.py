"""Uplift estimation, from the simplest thing that could work upward.

Three estimators, so the question "does the machine learning earn its place?" is
answered with a measurement rather than an assumption:

**`AmountOnly`** -- no uplift estimate at all. Rank by transaction value, act
wherever the taxonomy permits. The null hypothesis.

**`CauseRate`** -- a group-by. Historical recovery rate per (cause, action),
estimated from the training split with shrinkage toward the global mean for
sparse groups. Twenty-odd numbers, computable in SQL, explainable to a merchant
in one sentence.

**`ModelUplift`** -- the fitted per-case gradient-boosted uplift model.

The ordering above is deliberate: each is only worth adopting if it beats the one
before it. The signal in this problem is overwhelmingly *per cause* -- an expired
card and a bank outage recover at different rates for structural reasons -- and a
per-case model has to earn its keep against a group-by that captures exactly that
structure with none of the variance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from app.model import features as feat
from app.model.uplift import UpliftModel
from app.simulation.outcomes import ActionType


class UpliftEstimator(Protocol):
    name: str

    def estimate(self, cases: pd.DataFrame, actions: list[ActionType | None]) -> np.ndarray:
        ...


@dataclass
class AmountOnly:
    """No uplift estimate. Every permitted case is equally worth acting on, so
    ordering falls back to transaction value."""

    name: str = "amount_only"

    def estimate(self, cases: pd.DataFrame, actions) -> np.ndarray:
        return np.where([a is not None for a in actions], 1.0, 0.0)


@dataclass
class CauseRate:
    """Historical uplift per (cause, action), with shrinkage.

    Sparse groups are pulled toward the global mean in proportion to how little
    evidence they carry. Without it, a cause seen four times in training gets a
    rate of 0.00 or 1.00 and dominates the ranking on noise -- the classic
    failure of a naive group-by, and the reason this is not merely a mean.
    """

    name: str = "cause_rate"
    prior_strength: float = 50.0
    rates: dict[tuple[str, str], float] = field(default_factory=dict)
    global_rate: float = 0.0

    def fit(self, frame: pd.DataFrame) -> CauseRate:
        treated = frame[frame.treated]
        control = frame[~frame.treated]

        # Uplift, not raw recovery: the control arm says what would have happened
        # anyway, and crediting that to an action is how recovery numbers inflate.
        baseline_by_cause = control.groupby("cause").recovered.mean()
        self.global_rate = float(
            max(treated.recovered.mean() - control.recovered.mean(), 0.0)
        )

        for (cause, action), group in treated.groupby(["cause", "action"]):
            n = len(group)
            baseline = float(baseline_by_cause.get(cause, control.recovered.mean()))
            observed = float(group.recovered.mean() - baseline)
            # Shrink toward the global uplift by evidence weight.
            weight = n / (n + self.prior_strength)
            self.rates[(str(cause), str(action))] = max(
                weight * observed + (1 - weight) * self.global_rate, 0.0
            )
        return self

    def estimate(self, cases: pd.DataFrame, actions) -> np.ndarray:
        from app.simulation.policies import observable_cause

        out = np.zeros(len(cases))
        for i, action in enumerate(actions):
            if action is None:
                continue
            cause = observable_cause(cases.iloc[i]) or "unknown"
            out[i] = self.rates.get((cause, str(action)), self.global_rate)
        return out


@dataclass
class ModelUplift:
    """The fitted per-case uplift model."""

    model: UpliftModel
    name: str = "model_uplift"

    def estimate(self, cases: pd.DataFrame, actions) -> np.ndarray:
        features = feat.build(cases)
        out = np.zeros(len(cases))
        for action in {a for a in actions if a is not None}:
            mask = np.array([a == action for a in actions])
            out[mask] = self.model.uplift(features[mask], action, 2.0)
        return out
