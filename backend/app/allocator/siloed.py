"""Siloed per-channel agents: the architecture Recoup is actually arguing with.

Razorpay's Agent Studio ships separate agents -- Subscription Recovery,
Abandoned Cart Conversion, and so on. Each owns one loss channel, each works its
own queue, and nothing published describes a layer above them.

Every earlier baseline in this project quietly failed to model that. They pooled
all three channels into a single queue and worked it in value order, which means
they had already solved the collision problem for free. Against that strawman the
allocator's contact cap prevented exactly **one** customer from being
over-messaged and cost 1.3% of revenue for the privilege, and the honest reading
was that the collision guard did nothing.

That reading was an artifact of the baseline. Independent agents do not
coordinate: each ranks its own channel and spends its own budget, and their picks
collide on precisely the customers who are having the worst week -- someone whose
subscription failed *and* who abandoned a cart is high-value in two queues at
once, and gets contacted by both.

This module models that faithfully so the comparison is against the architecture
that exists rather than a more convenient one.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from app.allocator.bake_off import MAX_LADDER_STEPS, _StubStep
from app.allocator.evaluate import execute_ladders
from app.simulation import assumptions as A
from app.simulation.arms import ArmResult
from app.simulation.outcomes import CONTACT_ACTIONS, ActionType


def plan_siloed(
    cases: pd.DataFrame,
    policy_factory,
    total_budget: int,
    *,
    rank_within_channel: bool = True,
) -> tuple[list[list[tuple[ActionType, float]]], dict]:
    """Each channel gets its own agent and its own slice of the budget.

    The budget is split in proportion to each channel's revenue at risk, which is
    the most generous reasonable allocation -- a real deployment would more likely
    give each agent a fixed quota. Nothing coordinates between them.
    """
    at_risk = cases.groupby("channel").amount_paise.sum()
    share = at_risk / at_risk.sum()

    plans: list[list[tuple[ActionType, float]]] = [[] for _ in range(len(cases))]
    stats: dict = {"per_channel": {}}

    for channel, group in cases.groupby("channel"):
        budget = int(round(total_budget * share[channel]))
        policy = policy_factory()

        ordered = (
            group.sort_values("amount_paise", ascending=False)
            if rank_within_channel
            else group
        )

        spent = 0
        for position in ordered.index:
            row = cases.loc[position]
            steps: list[tuple[ActionType, float]] = []
            history: list[_StubStep] = []
            for _ in range(MAX_LADDER_STEPS):
                decision = policy.next_action(row, history)
                if decision is None:
                    break
                action, delay = decision
                if action in CONTACT_ACTIONS:
                    if spent >= budget:
                        break
                    spent += 1
                steps.append((action, delay))
                history.append(_StubStep(action, delay))
            plans[cases.index.get_loc(position)] = steps

        stats["per_channel"][str(channel)] = {"budget": budget, "contacts_spent": spent}

    return plans, stats


def contact_collisions(cases: pd.DataFrame, plans) -> dict:
    """How often one customer is contacted more than once, and by how many
    different channels.

    The cross-channel count is the number that matters. Two contacts from one
    agent is that agent escalating; two contacts from two agents is nobody
    owning the customer.
    """
    per_customer: dict[str, list[str]] = defaultdict(list)
    for i, steps in enumerate(plans):
        row = cases.iloc[i]
        for action, _ in steps:
            if action in CONTACT_ACTIONS:
                per_customer[str(row.customer_id)].append(str(row.channel))

    multi = {k: v for k, v in per_customer.items() if len(v) > 1}
    cross = {k: v for k, v in multi.items() if len(set(v)) > 1}
    return {
        "customers_contacted": len(per_customer),
        "contacted_more_than_once": len(multi),
        "contacted_by_multiple_channels": len(cross),
        "max_contacts_to_one_customer": max(
            (len(v) for v in per_customer.values()), default=0
        ),
    }


def run_siloed(
    cases: pd.DataFrame,
    policy_factory,
    total_budget: int,
    *,
    fatigue_lambda: float = A.CONTACT_FATIGUE_LAMBDA.value,
    arm_name: str = "SILOED_per_channel_agents",
) -> tuple[ArmResult, dict]:
    plans, stats = plan_siloed(cases, policy_factory, total_budget)
    result = execute_ladders(
        cases, plans, fatigue_lambda=fatigue_lambda, arm_name=arm_name
    )
    stats["collisions"] = contact_collisions(cases, plans)
    return result, stats
