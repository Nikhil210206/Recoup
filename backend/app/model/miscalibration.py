"""What miscalibration costs an allocator that spends money.

The claim this file exists to test: **ranking metrics cannot see the difference
between a model that is safe to spend money on and one that is not.**

The experiment takes the fitted model and applies a monotone distortion to its
probabilities. Monotone means the *order* of cases is untouched -- ROC-AUC,
PR-AUC and every other ranking metric are unchanged by construction. Only the
magnitudes move.

Then both versions allocate the same finite contact budget by expected value,
and the money each recovers is compared.

If the two allocate identically, calibration is a nicety and this project's
emphasis on it is misplaced. If they diverge, then a model can pass every metric
a normal ML review would look at and still be unsafe to spend from, and the
distinction is worth the machinery.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def distort(p: np.ndarray, kind: str, strength: float = 0.35) -> np.ndarray:
    """Monotone distortion. Order preserved exactly; magnitudes moved.

    `overconfident` pushes probabilities toward 1, which is the dangerous
    direction: expected values clear their cost threshold too easily, and the
    allocator spends its budget on cases that were never worth working.
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    if kind == "none":
        return p
    if kind == "overconfident":
        # Power transform with exponent < 1: strictly increasing, pushes up.
        return p ** (1.0 - strength)
    if kind == "underconfident":
        return p ** (1.0 + strength)
    if kind == "compressed":
        # Squashes everything toward the base rate. Ranking intact, spread gone.
        return p.mean() + (p - p.mean()) * (1.0 - strength)
    raise ValueError(kind)


@dataclass
class AllocationResult:
    label: str
    contacts_used: int
    cases_selected: int
    realised_paise: int
    #: What the model *claimed* it would recover. The gap between this and the
    #: realised figure is the practical cost of miscalibration: a merchant plans
    #: against the forecast.
    predicted_paise: int

    @property
    def forecast_error(self) -> float:
        if self.realised_paise == 0:
            return float("inf")
        return (self.predicted_paise - self.realised_paise) / self.realised_paise


def allocate_by_ev(
    amounts: np.ndarray,
    uplift: np.ndarray,
    cost_paise: np.ndarray,
    budget: int,
) -> np.ndarray:
    """Select cases by expected net value until the contact budget is spent.

    A greedy knapsack. Every action here costs one contact, so value-ordering is
    optimal rather than merely a heuristic.
    """
    ev = amounts * uplift - cost_paise
    order = np.argsort(-ev)
    chosen = np.zeros(len(amounts), dtype=bool)
    # Only positive expected value is worth a contact. This is the step that
    # over-confident probabilities break: they make negative-EV cases look
    # positive, so the budget is spent on cases that should have been left alone.
    for i in order[: int(budget)]:
        if ev[i] <= 0:
            break
        chosen[i] = True
    return chosen


def run_experiment(
    amounts: np.ndarray,
    true_uplift: np.ndarray,
    model_uplift: np.ndarray,
    cost_paise: np.ndarray,
    budget: int,
    distortions=("none", "overconfident", "underconfident", "compressed"),
) -> list[AllocationResult]:
    """Allocate the same budget using distorted versions of the same model.

    `true_uplift` is the counterfactual truth, used only to score the outcome --
    never to choose. Realised value is the expected money recovered from the
    cases actually selected.
    """
    results = []
    for kind in distortions:
        u = distort(model_uplift, kind)
        chosen = allocate_by_ev(amounts, u, cost_paise, budget)
        results.append(
            AllocationResult(
                label=kind,
                contacts_used=int(chosen.sum()),
                cases_selected=int(chosen.sum()),
                realised_paise=int((amounts[chosen] * true_uplift[chosen]).sum()),
                predicted_paise=int((amounts[chosen] * u[chosen]).sum()),
            )
        )
    return results
