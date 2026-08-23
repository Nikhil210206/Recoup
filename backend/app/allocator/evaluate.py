"""Execute an allocation plan against the outcome simulator and score it.

The allocator plans a whole batch at once, which the per-case episode runner
cannot express: its entire premise is comparing cases against each other for a
shared budget. So execution happens here, and the result is shaped like an
`ArmResult` so it drops into the same comparison table as the baselines.

Fatigue is applied in the order the plan was made. A customer's second contact
this week is worth less than their first, and pretending otherwise would flatter
exactly the arm that spends the most contacts.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from app.allocator.policy import Allocator, Decision
from app.simulation import assumptions as A
from app.simulation.arms import ArmResult
from app.simulation.outcomes import simulate_action, simulate_no_action


def execute(
    cases: pd.DataFrame,
    decisions: list[Decision],
    *,
    fatigue_lambda: float = A.CONTACT_FATIGUE_LAMBDA.value,
    arm_name: str = "RECOUP_allocator",
) -> ArmResult:
    """Run each decision and aggregate. Suppressed cases still get their
    counterfactual: money that arrives unprompted is money the merchant keeps,
    and an arm that chose not to act must be credited with it -- and denied
    causal credit for it."""
    by_case = {d.case_id: d for d in decisions}
    prior_contacts: dict[str, int] = defaultdict(int)

    gross = incremental = cost = contacts = unnecessary = 0
    recovered_cases = caused_cases = 0
    per_case_net = np.zeros(len(cases), dtype=np.int64)

    for i in range(len(cases)):
        row = cases.iloc[i]
        decision = by_case.get(str(row.case_id))

        if decision is None or not decision.acted:
            outcome = simulate_no_action(row)
        else:
            seen = prior_contacts[decision.customer_id]
            outcome = simulate_action(
                row,
                decision.action,
                delay_h=decision.delay_h,
                attempt_idx=seen,
                prior_contacts=seen,
                fatigue_lambda=fatigue_lambda,
            )
            if decision.uses_contact:
                prior_contacts[decision.customer_id] += 1

        gross += outcome.amount_recovered_paise
        cost += outcome.cost_paise
        contacts += outcome.contacts_used
        unnecessary += int(outcome.unnecessary_contact)
        recovered_cases += int(outcome.recovered)
        if outcome.caused_by_us:
            incremental += outcome.amount_recovered_paise
            caused_cases += 1
            per_case_net[i] = outcome.amount_recovered_paise - outcome.cost_paise
        else:
            per_case_net[i] = -outcome.cost_paise

    return ArmResult(
        arm=arm_name,
        n_cases=len(cases),
        at_risk_paise=int(cases.amount_paise.sum()),
        gross_paise=gross,
        incremental_paise=incremental,
        cost_paise=cost,
        contacts=contacts,
        unnecessary_contacts=unnecessary,
        recovered_cases=recovered_cases,
        caused_cases=caused_cases,
        per_case_net=per_case_net,
    )


def run(
    cases: pd.DataFrame,
    allocator: Allocator,
    *,
    fatigue_lambda: float = A.CONTACT_FATIGUE_LAMBDA.value,
    arm_name: str = "RECOUP_allocator",
) -> tuple[ArmResult, list[Decision], dict]:
    decisions, ledger = allocator.plan(cases)
    result = execute(
        cases, decisions, fatigue_lambda=fatigue_lambda, arm_name=arm_name
    )
    return result, decisions, ledger.summary()


def execute_ladders(
    cases: pd.DataFrame,
    plans: list[list[tuple]],
    *,
    fatigue_lambda: float = A.CONTACT_FATIGUE_LAMBDA.value,
    arm_name: str = "baseline",
) -> ArmResult:
    """Execute multi-step escalation ladders.

    A per-case policy may retry, wait, message, wait, message again. Scoring only
    its first rung understates both what it recovers and what it costs, and for
    an aggressive policy those errors do not cancel -- it looks cheaper *and*
    less effective than it is.

    The episode stops as soon as money arrives, from any source. A policy that
    keeps escalating after a silent self-recovery is charged for the contacts it
    would have sent, because in production it would have sent them.
    """
    from collections import defaultdict

    from app.simulation.outcomes import CONTACT_ACTIONS

    prior_contacts: dict[str, int] = defaultdict(int)
    gross = incremental = cost = contacts = unnecessary = 0
    recovered_cases = caused_cases = 0
    per_case_net = np.zeros(len(cases), dtype=np.int64)

    for i in range(len(cases)):
        row = cases.iloc[i]
        customer = str(row.customer_id)
        steps = plans[i] if i < len(plans) else []

        settled = None
        retry_idx = 0
        for action, delay in steps:
            is_contact = action in CONTACT_ACTIONS
            seen = prior_contacts[customer]
            outcome = simulate_action(
                row,
                action,
                delay_h=float(delay),
                attempt_idx=seen if is_contact else retry_idx,
                prior_contacts=seen,
                fatigue_lambda=fatigue_lambda,
            )
            if is_contact:
                prior_contacts[customer] += 1
            else:
                retry_idx += 1

            cost += outcome.cost_paise
            contacts += outcome.contacts_used
            unnecessary += int(outcome.unnecessary_contact)
            if outcome.recovered:
                settled = outcome
                break

        if settled is None:
            settled = simulate_no_action(row)

        gross += settled.amount_recovered_paise
        recovered_cases += int(settled.recovered)
        if settled.caused_by_us:
            incremental += settled.amount_recovered_paise
            caused_cases += 1
        per_case_net[i] = (
            settled.amount_recovered_paise if settled.caused_by_us else 0
        ) - settled.cost_paise

    return ArmResult(
        arm=arm_name,
        n_cases=len(cases),
        at_risk_paise=int(cases.amount_paise.sum()),
        gross_paise=gross,
        incremental_paise=incremental,
        cost_paise=cost,
        contacts=contacts,
        unnecessary_contacts=unnecessary,
        recovered_cases=recovered_cases,
        caused_cases=caused_cases,
        per_case_net=per_case_net,
    )
