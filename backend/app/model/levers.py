"""Which decision actually recovers the money?

A recovery system makes three decisions per case:

    1. WHICH cases to work, given a finite contact budget  (ranking)
    2. WHAT to do, and WHEN                                 (action selection)
    3. WHETHER to act at all                                (suppression)

The project originally assumed (1) was the important one -- rank by expected
value rather than by transaction amount, because Rs 10,000 at 90% beats
Rs 50,000 at 5%. That is intuitive and, measured here, worth **under one
percent**.

The reason is arithmetic rather than modelling. `corr(EV, amount) = 0.937`:
amounts span roughly 5,000x while uplift spans about 5x, so multiplying a
hugely-varying quantity by a mildly-varying one barely reorders it. It holds
per-merchant too, and it holds for an *oracle* that knows the true uplift -- so
this is not "the model needs to improve". The ceiling is low.

Decision (2) is worth about 500%. Retrying at a fixed 24 hours recovers a
fraction of what the right action at the cause-appropriate moment recovers on the
identical set of cases, because the actions are not close substitutes: an expired
card responds to a method-switch prompt roughly twenty times better than to a
retry, and no amount of clever case selection fixes having chosen the wrong verb.

This module runs that comparison so the claim is reproducible rather than
asserted, and so it stays honest if the data changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.simulation import assumptions as A
from app.simulation.outcomes import ActionType, _action_success_probability
from app.simulation.policies import observable_cause

#: Actions the allocator may choose between.
CANDIDATE_ACTIONS = (
    ActionType.RETRY,
    ActionType.PAYMENT_LINK_WHATSAPP,
    ActionType.PAYMENT_LINK_SMS,
    ActionType.METHOD_SWITCH_PROMPT,
)


@dataclass
class LeverResult:
    lever: str
    variant: str
    realised_paise: int
    contacts: int
    baseline_paise: int = 0

    @property
    def lift(self) -> float:
        return (
            (self.realised_paise - self.baseline_paise) / self.baseline_paise
            if self.baseline_paise
            else 0.0
        )


def true_uplift(cases: pd.DataFrame, actions, delays) -> np.ndarray:
    """Counterfactual incremental probability, for scoring only.

    An action only adds value on top of what would have happened anyway, so the
    incremental probability is `p_action * (1 - p_self)`. Using `p_action` alone
    would credit the system for recoveries that were already coming.
    """
    p_self = cases.latent_p_self_recover.to_numpy()
    out = np.zeros(len(cases))
    for i in range(len(cases)):
        action = actions[i]
        if action is None:
            continue
        p = _action_success_probability(cases.iloc[i], action, float(delays[i]), 0, 0.55)
        out[i] = p * (1 - p_self[i])
    return out


def cause_aware_plan(cases: pd.DataFrame) -> tuple[list, np.ndarray]:
    """Best action and timing per case, chosen from observable fields only.

    Uses Razorpay's error taxonomy and the loss channel -- never `latent_cause`.
    Returns `None` for cases where no action is permitted, which is the
    suppression decision.
    """
    actions, delays = [], np.zeros(len(cases))
    for i in range(len(cases)):
        row = cases.iloc[i]
        cause_key = observable_cause(row)
        if cause_key is None:
            actions.append(None)
            continue

        from app import taxonomy

        cause = taxonomy.get(cause_key)
        if cause.retry_policy == taxonomy.RetryPolicy.NEVER:
            # A fraud block or a merchant setting. No customer action helps.
            actions.append(None)
            continue

        delay = max(A.RECOVERABILITY[cause_key].best_delay_h, 0.05)
        allowed = [
            a
            for a in CANDIDATE_ACTIONS
            if not (a == ActionType.RETRY and row.channel == "abandoned_checkout")
            and not (a != ActionType.RETRY and not cause.contact_ok)
        ]
        best, best_p = None, -1.0
        for action in allowed:
            p = _action_success_probability(row, action, delay, 0, 0.55)
            if p > best_p:
                best, best_p = action, p
        actions.append(best)
        delays[i] = delay
    return actions, delays


def compare(cases: pd.DataFrame, budget: int) -> list[LeverResult]:
    """Run both levers against the same budget and return the comparison."""
    amounts = cases.amount_paise.to_numpy().astype(float)
    results: list[LeverResult] = []

    def select_by(score: np.ndarray) -> np.ndarray:
        chosen = np.zeros(len(score), dtype=bool)
        chosen[np.argsort(-score)[:budget]] = True
        return chosen

    # --- Lever 2: action selection, holding case selection fixed -------------
    by_amount = select_by(amounts)
    fixed_retry_u = true_uplift(
        cases, [ActionType.RETRY] * len(cases), np.full(len(cases), 24.0)
    )
    baseline = int((amounts[by_amount] * fixed_retry_u[by_amount]).sum())

    variants = [
        ("fixed retry @ 24h", [ActionType.RETRY] * len(cases), np.full(len(cases), 24.0)),
        (
            "fixed link @ 24h",
            [ActionType.PAYMENT_LINK_WHATSAPP] * len(cases),
            np.full(len(cases), 24.0),
        ),
        (
            "retry @ cause-best time",
            [ActionType.RETRY] * len(cases),
            np.array(
                [
                    max(
                        A.RECOVERABILITY[
                            observable_cause(cases.iloc[i]) or "hard_decline"
                        ].best_delay_h,
                        0.05,
                    )
                    for i in range(len(cases))
                ]
            ),
        ),
    ]
    plan_actions, plan_delays = cause_aware_plan(cases)
    variants.append(("cause-aware action + timing", plan_actions, plan_delays))

    for label, actions, delays in variants:
        u = true_uplift(cases, actions, delays)
        results.append(
            LeverResult(
                lever="action",
                variant=label,
                realised_paise=int((amounts[by_amount] * u[by_amount]).sum()),
                contacts=int(by_amount.sum()),
                baseline_paise=baseline,
            )
        )

    # --- Lever 1: ranking, holding the action fixed at the best available ----
    best_u = true_uplift(cases, plan_actions, plan_delays)
    rank_baseline = int((amounts[by_amount] * best_u[by_amount]).sum())
    for label, score in [
        ("amount ranked", amounts),
        ("EV ranked (oracle uplift)", amounts * best_u),
    ]:
        chosen = select_by(score)
        results.append(
            LeverResult(
                lever="ranking",
                variant=label,
                realised_paise=int((amounts[chosen] * best_u[chosen]).sum()),
                contacts=int(chosen.sum()),
                baseline_paise=rank_baseline,
            )
        )
    return results


def render(results: list[LeverResult]) -> str:
    lines = [
        f"  {'lever':10}{'variant':30}{'realised Rs':>14}{'lift':>10}",
        "  " + "-" * 62,
    ]
    for r in results:
        lines.append(
            f"  {r.lever:10}{r.variant:30}{r.realised_paise / 100:>14,.0f}{r.lift:>+9.1%}"
        )
    return "\n".join(lines)
