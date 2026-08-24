"""The allocator: what to do about each case, whether to do it, and who is allowed to.

Every recovery agent asks "can I recover this?". This layer asks "should I?" --
and the measured answer is more interesting than the one the project set out to
find.

## What is actually worth what

Measured at a 600-contact budget against siloed per-channel agents, which is the
architecture that exists today:

| component | measured value |
|---|---|
| action selection from the taxonomy | **+520%** |
| value-ordering the budget vs arrival order | **+18.5%** |
| everything distinctive to this allocator | **+3.3% to -2.1%** |

The last row is budget-dependent and, honestly, noise. The shared budget, the
contact cap, the expected-value floor and the learned uplift estimator do not pay
for themselves in revenue terms. Only 9.5% of customers here have cases in more
than one channel, so the cross-agent collision this layer exists to prevent
rarely binds: at a 600-contact budget it affects 13 of 458 contacted customers.

That is reported rather than buried, and it changes what this component claims to
be.

## What it is, then

**A governance layer, not a revenue layer.** The revenue comes from choosing the
right action and spending a finite budget in value order -- both of which a
simple per-case policy can do. What this adds, at a cost of roughly 1%:

- a **hard cap on how much of one customer's patience a merchant may spend**,
  enforced across every loss channel at once, which no per-channel agent can do
  because none of them can see the others
- **suppression rules** that refuse to act where no customer action can help --
  a fraud block, a merchant misconfiguration
- **quiet hours**, so nobody is messaged at 3am about a failed payment
- a **decision-level audit trail**: every case carries why it was acted on or
  left alone, which rule fired, and what was predicted

For a payments company those are not garnish. They are the difference between an
automation you can put in front of a compliance review and one you cannot. But
they are worth stating as governance rather than dressed up as revenue.

## Structure

The model proposes; deterministic policy disposes. Anything touching money or a
customer's inbox is a rule that can be read, and the audit trail records which
rule fired rather than only what happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from app import taxonomy
from app.allocator.budget import BudgetLedger, BudgetPolicy
from app.simulation import assumptions as A
from app.simulation.outcomes import CONTACT_ACTIONS, ActionType
from app.simulation.policies import observable_cause

#: Actions the allocator may choose between, with the delays worth considering.
#: Timing is per-cause rather than fixed, so the grid is expressed as multipliers
#: of the cause's best moment.
DELAY_MULTIPLIERS = (0.5, 1.0, 2.0)

CANDIDATE_ACTIONS = (
    ActionType.RETRY,
    ActionType.PAYMENT_LINK_WHATSAPP,
    ActionType.PAYMENT_LINK_SMS,
    ActionType.METHOD_SWITCH_PROMPT,
)


@dataclass
class Decision:
    """One allocation decision, with everything needed to defend it later."""

    case_id: str
    customer_id: str
    action: ActionType | None
    delay_h: float
    predicted_uplift: float
    expected_value_paise: int
    cost_paise: int
    #: Why this outcome. Populated for suppressions too -- "nothing happened" is
    #: a decision and needs a reason as much as an action does.
    reason: str
    rule: str
    channel: str = ""
    cause: str | None = None

    @property
    def acted(self) -> bool:
        return self.action is not None

    @property
    def uses_contact(self) -> bool:
        return self.action in CONTACT_ACTIONS


def _action_cost(action: ActionType) -> int:
    key = str(action)
    return int(A.ACTION_COST_PAISE[key].value) if key in A.ACTION_COST_PAISE else 0


def preferred_action(row, cause: taxonomy.Cause) -> ActionType | None:
    """The action Razorpay's own taxonomy implies for this cause and channel.

    **Deliberately not learned.** The taxonomy already encodes which action fits
    which cause -- an expired card needs a different instrument, a bank outage
    needs a retry and no message at all -- and that is published, documented
    semantics rather than a pattern to rediscover from noisy outcomes.

    Measured: the uplift model, asked to choose the action, picked the true-best
    one on **3 of 11 causes**. It always chose a payment link, because that is the
    correct answer *on average* and the per-cause interaction is a second-order
    effect inside a first-order-noisy Bernoulli signal at ROC 0.62. This rule
    picks the true-best action on 9 of 11.

    Same judgement as the classifier's deterministic tier: a model is the wrong
    tool for a closed set somebody has already written down. The model's job here
    is magnitude -- how much is this worth, and is it worth anything -- which is
    exactly what a taxonomy cannot tell you.
    """
    channel = str(row.get("channel") or "")

    if cause.retry_policy == taxonomy.RetryPolicy.NEVER:
        # A merchant-configuration failure has a correct action -- it is just not
        # a customer-facing one. Telling the merchant that international cards
        # are disabled is the whole recoverable value here, and the rule was
        # named `merchant_alert_only` while emitting no alert at all.
        if cause.who_can_fix == taxonomy.WhoCanFix.MERCHANT:
            return ActionType.MERCHANT_ALERT
        # A risk block is different: there is no correct action. Working around a
        # fraud control is not a recovery.
        return None

    # No payment exists to re-attempt, so the only route is the customer.
    if channel == "abandoned_checkout":
        return ActionType.PAYMENT_LINK_WHATSAPP if cause.contact_ok else None

    # The instrument itself is the problem; retrying it repeats the failure.
    if cause.retry_policy == taxonomy.RetryPolicy.DIFFERENT_INSTRUMENT:
        return ActionType.METHOD_SWITCH_PROMPT if cause.contact_ok else None

    # Nobody downstream can act -- a bank outage resolves on its own. Retry, and
    # do not message the customer about their bank's downtime.
    if not cause.contact_ok:
        return ActionType.RETRY

    return ActionType.RETRY


def _feasible_actions(row, cause: taxonomy.Cause) -> list[ActionType]:
    """Actions that are physically possible and permitted for this case.

    Not a preference ordering -- a feasibility filter. A retry against an
    abandoned checkout is not a weak option, it is an impossible one: there is
    no payment to re-attempt.
    """
    allowed = []
    for action in CANDIDATE_ACTIONS:
        if action == ActionType.RETRY:
            if str(row.get("channel") or "") == "abandoned_checkout":
                continue
            if cause.retry_policy == taxonomy.RetryPolicy.DIFFERENT_INSTRUMENT:
                continue  # the same instrument will fail identically
        elif not cause.contact_ok:
            continue  # messaging about a bank outage or a merchant setting
        allowed.append(action)
    return allowed


@dataclass
class Allocator:
    """Plans a batch of cases under a shared budget.

    The uplift estimator is pluggable, and which one to use was settled by
    measurement rather than preference. Ranking 1,080 contactable cases under a
    300-contact budget:

    | estimator | realised | corr with true uplift |
    |---|---|---|
    | amount only (no estimate) | Rs 19.85L | -- |
    | **cause-rate group-by** | Rs 19.85L | **+0.362** |
    | per-case GBM uplift model | Rs 18.55L | +0.275 |
    | oracle (true uplift) | Rs 19.92L | 1.000 |

    Two things follow. The group-by tracks true uplift *better* than the fitted
    model -- the signal in this problem is structural and per-cause, and a
    per-case model adds variance without adding signal. And the oracle itself
    buys only +0.3%, so no estimator can rescue ranking; it is not a lever.

    The per-case model is therefore not in the decision path. It is retained for
    forecasting, where calibration measurably matters and a per-case estimate is
    what a merchant actually plans against.
    """

    estimator: object
    budget_policy: BudgetPolicy = field(default_factory=BudgetPolicy)
    #: Minimum expected net value, in paise, before a contact is worth spending.
    #: Not zero: a marginal case consumes a customer's finite tolerance that a
    #: better case could have used, and the cheap message is not the real cost.
    min_ev_paise: int = 5_000

    def plan(
        self, cases: pd.DataFrame, now: datetime | None = None
    ) -> tuple[list[Decision], BudgetLedger]:
        now = now or datetime.now(UTC)
        ledger = BudgetLedger(policy=self.budget_policy)

        proposals = self._propose(cases)

        # Ranking. Worth ~1% on its own; the ordering matters here mainly because
        # the contact budget is finite and something has to go first.
        proposals.sort(key=lambda d: -d.expected_value_paise)

        decisions: list[Decision] = []
        for proposal in proposals:
            decisions.append(self._gate(proposal, ledger, now))
        return decisions, ledger

    # -- proposal -----------------------------------------------------------

    def _propose(self, cases: pd.DataFrame) -> list[Decision]:
        """Taxonomy chooses the action; the model prices it.

        The split matters. Action choice is a closed, documented question and a
        rule answers it better than a model does. Magnitude -- is this worth a
        contact at all, and which cases get the scarce ones -- is an open
        question no taxonomy can answer, and that is the model's job.
        """
        amounts = cases.amount_paise.to_numpy().astype(float)
        causes = [observable_cause(cases.iloc[i]) for i in range(len(cases))]

        actions: list[ActionType | None] = []
        delays = np.zeros(len(cases))
        for i, cause_key in enumerate(causes):
            if cause_key is None:
                actions.append(None)
                continue
            row = cases.iloc[i]
            cause = taxonomy.get(cause_key)
            chosen = preferred_action(row, cause)
            # The feasibility filter covers customer-facing actions only. A
            # merchant alert is not one of them -- it goes to the merchant, costs
            # no contact, and is the correct output for a misconfiguration -- so
            # filtering it against the customer-action list silently discarded it
            # and the case fell through to the fraud-suppression branch.
            if (
                chosen is not None
                and chosen != ActionType.MERCHANT_ALERT
                and chosen not in _feasible_actions(row, cause)
            ):
                chosen = None
            actions.append(chosen)
            delays[i] = max(A.RECOVERABILITY[cause_key].best_delay_h, 0.05)

        # Price each distinct action once across the batch rather than per row:
        # a per-case call into a fitted pipeline is dominated by overhead, and
        # this runs on every tick.
        uplift = np.zeros(len(cases))
        for action in {a for a in actions if a is not None}:
            mask = np.array([a == action for a in actions])
            if not mask.any():
                continue
            subset_cases = cases[mask]
            subset_actions = [a for a, m in zip(actions, mask, strict=True) if m]
            uplift[mask] = self.estimator.estimate(subset_cases, subset_actions)

        proposals = []
        for i in range(len(cases)):
            row = cases.iloc[i]
            action = actions[i]
            cost = _action_cost(action) if action else 0
            ev = int(amounts[i] * uplift[i] - cost) if action else 0
            proposals.append(
                Decision(
                    case_id=str(row.case_id),
                    customer_id=str(row.customer_id),
                    action=action,
                    delay_h=float(delays[i]),
                    predicted_uplift=float(max(uplift[i], 0.0)),
                    expected_value_paise=ev,
                    cost_paise=cost,
                    reason="",
                    rule="",
                    channel=str(row.get("channel") or ""),
                    cause=causes[i],
                )
            )
        return proposals

    # -- gating -------------------------------------------------------------

    def _gate(self, d: Decision, ledger: BudgetLedger, now: datetime) -> Decision:
        """Deterministic rules, applied in order. The model cannot override any."""
        if d.cause is None:
            return self._suppress(d, ledger, "unclassified", "exception_list")

        cause = taxonomy.get(d.cause)
        if cause.retry_policy == taxonomy.RetryPolicy.NEVER:
            # No customer action clears either of these, so every customer-facing
            # option has expected value zero and non-zero cost. But a merchant
            # misconfiguration still has a correct output: tell the merchant.
            if d.action == ActionType.MERCHANT_ALERT:
                d.reason = f"cause_{d.cause}_is_the_merchants_to_fix"
                d.rule = "merchant_alert_only"
                return d
            return self._suppress(
                d, ledger, f"cause_{d.cause}_is_not_recoverable", "risk_suppression"
            )

        if d.action is None:
            return self._suppress(d, ledger, "no_feasible_action", "feasibility")

        if not d.uses_contact:
            # A retry touches the gateway, not the customer, so it consumes none
            # of the scarce resource. Any positive expected uplift is worth
            # taking; there is nothing to trade it off against.
            #
            # The EV floor used to be applied here too, which suppressed hundreds
            # of *free* retries for failing to clear a threshold that exists to
            # ration contacts. It cost the allocator more value than every other
            # gate combined, and it looked like prudence.
            if d.predicted_uplift <= 0:
                return self._suppress(d, ledger, "no_expected_uplift", "zero_uplift")
            d.reason = "retry_no_contact_required"
            d.rule = "unrationed"
            return d

        # From here down the action costs a contact, and a contact is the scarce
        # thing. A marginal case consumes a customer's finite tolerance that a
        # better case could have used.
        if d.expected_value_paise < self.min_ev_paise:
            return self._suppress(
                d, ledger, f"ev_below_threshold_{self.min_ev_paise}", "ev_floor"
            )

        allowed, why = ledger.can_contact(d.customer_id, now)
        if not allowed:
            return self._suppress(d, ledger, why, "contact_budget")

        send_at = now + timedelta(hours=d.delay_h)
        if self.budget_policy.is_quiet_at(send_at):
            # Deferred, not dropped. The case is still worth recovering; 3am is
            # simply not the moment to ask.
            allowed_at = self.budget_policy.next_allowed_time(send_at)
            shift_h = (allowed_at - send_at).total_seconds() / 3600.0
            d.delay_h += shift_h
            d.reason = f"deferred_{shift_h:.1f}h_out_of_quiet_hours_ist"
            d.rule = "quiet_hours"
        else:
            d.reason = why
            d.rule = "contact_budget"

        ledger.spend(d.customer_id, now)
        return d

    @staticmethod
    def _suppress(d: Decision, ledger: BudgetLedger, reason: str, rule: str) -> Decision:
        ledger.refuse(rule)
        d.action = None
        d.reason = reason
        d.rule = rule
        return d


def summarise(decisions: list[Decision]) -> dict:
    acted = [d for d in decisions if d.acted]
    contacts = [d for d in acted if d.uses_contact]
    suppressed = [d for d in decisions if not d.acted]
    by_rule: dict[str, int] = {}
    for d in suppressed:
        by_rule[d.rule] = by_rule.get(d.rule, 0) + 1
    return {
        "cases": len(decisions),
        "acted": len(acted),
        "contacts": len(contacts),
        "free_retries": len(acted) - len(contacts),
        "suppressed": len(suppressed),
        "suppressed_by_rule": by_rule,
    }
