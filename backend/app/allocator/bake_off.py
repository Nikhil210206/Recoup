"""Equal-budget comparison: given N contacts, who recovers the most?

The earlier arm table let every policy spend as many contacts as it wanted.
Contact-everything used 4,074 and recovered the most in absolute terms, which is
unsurprising and not very interesting -- it is what any policy does when the
constraint it is supposed to respect is not applied.

The allocator exists for the case where contacts are scarce, because that is the
real situation: a customer's tolerance is finite, and a merchant who messages
everyone about everything stops being able to message anyone about anything. So
this compares every arm under an identical contact budget, which is the only
comparison in which the word "allocator" means anything.

Retries are not rationed. They touch the gateway rather than the customer, so an
arm that recovers revenue without spending contacts is free to do so -- and that
asymmetry between channels is most of what there is to arbitrate.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.allocator.evaluate import execute, execute_ladders
from app.allocator.policy import Allocator
from app.simulation import assumptions as A
from app.simulation.arms import ArmResult
from app.simulation.episode import Policy
from app.simulation.outcomes import CONTACT_ACTIONS, ActionType

#: How far a baseline's escalation ladder is followed.
MAX_LADDER_STEPS = 4


@dataclass
class _StubStep:
    """Minimal stand-in for an episode step, so a per-case policy can be walked
    forward without running the simulator."""

    action: ActionType
    at_hours: float


@dataclass
class BudgetedResult:
    result: ArmResult
    budget: int
    contacts_offered: int

    @property
    def within_budget(self) -> bool:
        return self.result.contacts <= self.budget


def _baseline_plan(
    cases: pd.DataFrame, policy: Policy, budget: int
) -> list[list[tuple[ActionType, float]]]:
    """Expand a per-case policy into its full action ladder, under a budget.

    Earlier this executed only the *first* step. Contact-everything's first rung
    is a retry, so it was scored on a single WhatsApp message with its retry
    discarded -- a maximally aggressive multi-touch policy measured as if it were
    a single-touch one. Its whole design is escalation; scoring one rung of it is
    not scoring it.

    Contacts are charged against the shared budget as they are spent, in arrival
    order. That is the honest model of a per-case agent: it has no notion of a
    shared budget, so it simply works its queue until the budget is gone.
    """
    plans: list[list[tuple[ActionType, float]]] = []
    spent = 0
    for i in range(len(cases)):
        row = cases.iloc[i]
        steps: list[tuple[ActionType, float]] = []
        history: list[_StubStep] = []
        for _ in range(MAX_LADDER_STEPS):
            plan = policy.next_action(row, history)
            if plan is None:
                break
            action, delay = plan
            if action in CONTACT_ACTIONS:
                if spent >= budget:
                    break  # budget exhausted; the rest of the ladder is unworked
                spent += 1
            steps.append((action, delay))
            history.append(_StubStep(action, delay))
        plans.append(steps)
    return plans


def ranked(cases: pd.DataFrame) -> pd.DataFrame:
    """Cases in descending value order.

    Used to build the fairest possible control: a per-case policy that is handed
    its queue already sorted. Without this, comparing the allocator against a
    policy working cases in arrival order credits the allocator for an ORDER BY,
    and an ORDER BY is not a product.
    """
    return cases.sort_values("amount_paise", ascending=False).reset_index(drop=True)


def run_bake_off(
    cases: pd.DataFrame,
    allocator: Allocator,
    baselines: dict[str, Policy],
    budget: int,
    fatigue_lambda: float = A.CONTACT_FATIGUE_LAMBDA.value,
) -> list[BudgetedResult]:
    out: list[BudgetedResult] = []

    for name, policy in baselines.items():
        plans = _baseline_plan(cases, policy, budget)
        out.append(
            BudgetedResult(
                execute_ladders(
                    cases, plans, fatigue_lambda=fatigue_lambda, arm_name=name
                ),
                budget,
                sum(1 for steps in plans for a, _ in steps if a in CONTACT_ACTIONS),
            )
        )

    allocator.budget_policy.max_total_contacts = budget
    decisions, _ = allocator.plan(cases)
    out.append(
        BudgetedResult(
            execute(
                cases,
                decisions,
                fatigue_lambda=fatigue_lambda,
                arm_name="RECOUP_allocator",
            ),
            budget,
            sum(1 for d in decisions if d.acted and d.uses_contact),
        )
    )
    return out


def render(results: list[BudgetedResult], budget: int) -> str:
    hdr = (
        f"  {'arm':32}{'incremental Rs':>16}{'contacts':>10}"
        f"{'unnec%':>9}{'Rs/contact':>12}"
    )
    lines = [f"  budget: {budget:,} contacts", "", hdr, "  " + "-" * (len(hdr) - 2)]
    for r in sorted(results, key=lambda x: -x.result.incremental_paise):
        res = r.result
        # `recovered_cases - contacts` was being shown as "free retries", which
        # it is not: it mixes self-recoveries and retries and can go negative.
        # Unnecessary-contact rate is a number that means something.
        rpc = "—" if res.contacts == 0 else f"{res.rupees_per_contact:,.0f}"
        flag = "" if r.within_budget else "  OVER BUDGET"
        lines.append(
            f"  {res.arm:32}{res.incremental_paise / 100:>16,.0f}{res.contacts:>10,}"
            f"{res.unnecessary_rate:>8.0%}{rpc:>12}{flag}"
        )
    return "\n".join(lines)
