"""The uplift model: two calibrated probabilities, and their difference.

    uplift(case, action) = P(recover | action) - P(recover | no action)

The subtraction is why calibration is load-bearing rather than tidy. A ranking
error in either model is survivable; a *bias* in either goes straight into the
difference. If the treated model is 10 points optimistic and the baseline is
accurate, every uplift is 10 points too high, every expected value clears its
cost threshold too easily, and the allocator spends its whole budget on cases
that were never worth working. The ranking still looks fine.

Both targets are observable in production -- "did it recover after we acted" and
"did it recover when we did nothing" -- so this is retrainable on real traffic.
That constraint is deliberate. A model that needs the counterfactual can only
ever live in a simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.model import calibration as cal
from app.model import features as feat
from app.simulation.outcomes import ActionType


def _pipeline(kind: str, treated: bool) -> Pipeline:
    categorical, numeric = feat.columns(treated)
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), categorical),
            ("num", StandardScaler(), numeric),
        ],
        remainder="drop",
    )
    if kind == "logistic":
        model = LogisticRegression(max_iter=2000, C=1.0)
    else:
        model = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=40, l2_regularization=1.0, random_state=0,
        )
    return Pipeline([("pre", pre), ("clf", model)])


@dataclass
class ArmModel:
    """One calibrated probability model (control or treatment)."""

    name: str
    treated: bool
    kind: str = "gbm"
    pipeline: Pipeline | None = None
    calibrated: CalibratedClassifierCV | None = None
    #: "raw", "isotonic" or "sigmoid" -- whichever scored best on a slice used
    #: for nothing else.
    chosen_method: str = "raw"
    calibration_scores: dict = field(default_factory=dict)
    metrics_raw: cal.Metrics | None = None
    metrics_calibrated: cal.Metrics | None = None

    def fit(
        self, train: pd.DataFrame, calib: pd.DataFrame, select: pd.DataFrame | None = None
    ) -> ArmModel:
        """Fit, then decide whether calibrating actually helps.

        Calibration is applied *if it earns its place*, not by default. A
        logistic model optimises log loss and is therefore close to calibrated by
        construction; isotonic regression fitted on a few hundred rows can then
        add variance without removing any bias, and make the probabilities worse.

        Measured here: isotonic moved ECE from 0.028 to 0.032 on the treated arm.
        Applying it anyway -- because "calibration is best practice" -- would have
        degraded the exact quantity the allocator depends on, while the write-up
        claimed it was improved.

        So all three options are scored on a slice used for nothing else, and the
        best one wins. `chosen_method` records which, so the audit trail says what
        actually produced a probability.
        """
        cols = self._cols()
        self.pipeline = _pipeline(self.kind, self.treated)
        self.pipeline.fit(train[cols], train.recovered.astype(int))

        # `FrozenEstimator` calibrates an already-fitted model without refitting
        # it. (It replaced `cv="prefit"`, removed in recent scikit-learn.)
        candidates: dict[str, CalibratedClassifierCV | None] = {"raw": None}
        for method in ("isotonic", "sigmoid"):
            try:
                fitted = CalibratedClassifierCV(
                    FrozenEstimator(self.pipeline), method=method
                )
                fitted.fit(calib[cols], calib.recovered.astype(int))
                candidates[method] = fitted
            except ValueError:
                continue

        scoring = select if select is not None and len(select) else calib
        y = scoring.recovered.astype(int)

        best_name, best_score = "raw", None
        for name, fitted in candidates.items():
            probs = (
                self.pipeline.predict_proba(scoring[cols])[:, 1]
                if fitted is None
                else fitted.predict_proba(scoring[cols])[:, 1]
            )
            m = cal.evaluate(y, probs)
            # Brier is proper: it rewards both ranking and honest magnitude, so
            # it will not accept a calibrator that trades one for the other.
            self.calibration_scores[name] = m
            if best_score is None or m.brier < best_score:
                best_name, best_score = name, m.brier

        self.chosen_method = best_name
        self.calibrated = candidates[best_name]
        return self

    def _cols(self) -> list[str]:
        categorical, numeric = feat.columns(self.treated)
        return [*categorical, *numeric]

    def predict_raw(self, frame: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(frame[self._cols()])[:, 1]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Calibrated probability, or the raw one if calibrating made it worse."""
        if self.calibrated is None:
            return self.predict_raw(frame)
        return self.calibrated.predict_proba(frame[self._cols()])[:, 1]

    def evaluate(self, test: pd.DataFrame) -> None:
        y = test.recovered.astype(int)
        self.metrics_raw = cal.evaluate(y, self.predict_raw(test))
        self.metrics_calibrated = cal.evaluate(y, self.predict(test))


@dataclass
class UpliftModel:
    """Baseline and treated models, plus the uplift they imply."""

    kind: str = "gbm"
    baseline: ArmModel = field(default=None)
    treated: ArmModel = field(default=None)

    def fit(
        self, train: pd.DataFrame, calib: pd.DataFrame, select: pd.DataFrame | None = None
    ) -> UpliftModel:
        from app.model.dataset import split_arms

        train_c, train_t = split_arms(train)
        calib_c, calib_t = split_arms(calib)
        select_c, select_t = (
            split_arms(select) if select is not None else (None, None)
        )

        self.baseline = ArmModel("baseline", treated=False, kind=self.kind).fit(
            train_c, calib_c, select_c
        )
        self.treated = ArmModel("treated", treated=True, kind=self.kind).fit(
            train_t, calib_t, select_t
        )
        return self

    def uplift(
        self,
        cases_features: pd.DataFrame,
        action: ActionType,
        delay_h: float,
        prior_contacts: float = 0.0,
        *,
        calibrated: bool = True,
    ) -> np.ndarray:
        """Incremental recovery probability for one action at one delay.

        Clipped at zero. A negative uplift means the model believes acting makes
        recovery *less* likely, which is not a coherent basis for spending money
        -- the correct response is to not act, which zero already expresses.
        """
        n = len(cases_features)
        treated_frame = feat.add_treatment(
            cases_features,
            pd.Series([str(action)] * n, index=cases_features.index),
            pd.Series([delay_h] * n, index=cases_features.index),
            pd.Series([prior_contacts] * n, index=cases_features.index),
        )
        control_frame = feat.add_treatment(
            cases_features,
            pd.Series(["none"] * n, index=cases_features.index),
            pd.Series([0.0] * n, index=cases_features.index),
            pd.Series([prior_contacts] * n, index=cases_features.index),
        )

        predict = (
            (lambda m, f: m.predict(f)) if calibrated else (lambda m, f: m.predict_raw(f))
        )
        p_treated = predict(self.treated, treated_frame)
        p_baseline = predict(self.baseline, control_frame)
        return np.clip(p_treated - p_baseline, 0.0, 1.0)

    def best_action(
        self,
        cases_features: pd.DataFrame,
        candidates: list[tuple[ActionType, float]],
        prior_contacts: float = 0.0,
        *,
        calibrated: bool = True,
    ) -> tuple[np.ndarray, list[ActionType], np.ndarray]:
        """Highest-uplift (action, delay) per case.

        This is the action-success model: not "will this recover" but "which of
        the things we could do recovers it, and when". A cause tells you retrying
        is possible; this tells you whether retrying beats a payment link for
        *this* customer at *this* hour.
        """
        best_u = np.full(len(cases_features), -np.inf)
        best_a: list[ActionType] = [candidates[0][0]] * len(cases_features)
        best_d = np.zeros(len(cases_features))

        for action, delay in candidates:
            u = self.uplift(
                cases_features, action, delay, prior_contacts, calibrated=calibrated
            )
            better = u > best_u
            best_u = np.where(better, u, best_u)
            best_d = np.where(better, delay, best_d)
            for i in np.flatnonzero(better):
                best_a[i] = action
        return best_u, best_a, best_d
