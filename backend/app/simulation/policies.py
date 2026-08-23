"""Per-case baseline policies.

These are the comparators. Two of them are deliberately not strawmen:

`RazorpayT3` is the policy a merchant on Razorpay Subscriptions actually gets
today -- retry the next day, three times, over a T+3 cycle, regardless of cause,
issuer, or customer. It is documented, it is in production, and it is the number
Recoup has to beat to claim anything.

`ContactEverything` is what an unconstrained recovery agent does when its only
objective is recovery rate. It will recover more gross revenue than anything
else here. Whether that is a *good* policy is the question the evaluation exists
to answer, and the answer lives in the cost and unnecessary-contact columns.

The budget-constrained and calibrated-EV arms are not here: they are allocator
concerns and cannot be expressed as a per-case policy, because their whole
premise is comparing cases against each other for a shared budget.
"""

from __future__ import annotations

from app import taxonomy
from app.simulation import assumptions as A
from app.simulation.episode import Step
from app.simulation.outcomes import (
    CONTACT_ACTIONS,
    ActionType,
    _action_success_probability,
)


class DoNothing:
    """B0. The floor. Whatever this recovers, the merchant gets for free."""

    name = "B0_do_nothing"

    def next_action(self, case, history: list[Step]) -> tuple[ActionType, float] | None:
        return None


class RazorpayT3:
    """B1. Razorpay's documented subscription retry: next day, three times.

    One schedule for every failure. An authentication failure -- where the
    customer is still at the checkout and the retry window is minutes -- gets the
    same 24-hour wait as insufficient funds, where 24 hours is too *early* to
    help. That mismatch is not a strawman; it is what a fixed schedule means.
    """

    name = "B1_razorpay_t3"

    def __init__(self, retries: int = 3, interval_hours: float = 24.0):
        self.retries = retries
        self.interval_hours = interval_hours

    def next_action(self, case, history: list[Step]) -> tuple[ActionType, float] | None:
        if len(history) >= self.retries:
            return None
        return ActionType.RETRY, self.interval_hours * (len(history) + 1)


class ContactEverything:
    """B2. Maximal intervention: message every case, escalate, never stop.

    Recovers the most gross revenue in the run. It also pays for every contact,
    burns fatigue on customers who were going to pay anyway, and messages the
    cases where no customer action can possibly help.
    """

    name = "B2_contact_everything"

    LADDER = [
        (ActionType.RETRY, 1.0),
        (ActionType.PAYMENT_LINK_WHATSAPP, 2.0),
        (ActionType.PAYMENT_LINK_SMS, 24.0),
        (ActionType.PAYMENT_LINK_SMS, 72.0),
    ]

    def next_action(self, case, history: list[Step]) -> tuple[ActionType, float] | None:
        if len(history) >= len(self.LADDER):
            return None
        return self.LADDER[len(history)]


class CauseAware:
    """Cause-aware single action, with no budget and no probability model.

    Deliberately positioned between the baselines and the full system. It knows
    only what Razorpay's error taxonomy already tells you -- who can fix this,
    does retrying help, is contact appropriate -- and acts at the cause's best
    moment.

    Its purpose in the ablation is to separate two claims that are easy to
    conflate: how much comes from *knowing the cause*, and how much comes from
    *calibrated allocation under a budget*. If this arm captured most of the
    gain, the modelling would not be earning its place.
    """

    name = "B4_cause_aware"

    def next_action(self, case, history: list[Step]) -> tuple[ActionType, float] | None:
        if history:
            return None  # single shot

        # Classify from what a live system can see -- Razorpay's error fields --
        # never from `latent_cause`. Reading the ground truth here would make
        # this arm unbeatable for a reason that has nothing to do with policy,
        # and any reviewer grepping for `latent_` in a policy would be right to
        # throw out every number in the evaluation.
        classified = taxonomy.classify(
            case.get("error_reason"), case.get("error_source"), case.get("error_step")
        )
        if classified.cause is None:
            return None  # unmapped tail: hand to the exception list, do not guess

        cause = taxonomy.get(classified.cause)
        # `best_delay_h` is a published constant per cause, not a per-case latent.
        delay = max(A.RECOVERABILITY[classified.cause].best_delay_h, 0.05)

        # Nothing the customer can do fixes a merchant setting, and a risk block
        # must not be worked around. Both suppress contact entirely.
        if cause.retry_policy == taxonomy.RetryPolicy.NEVER:
            if cause.who_can_fix == taxonomy.WhoCanFix.MERCHANT:
                return ActionType.MERCHANT_ALERT, 0.0
            return None

        if cause.retry_policy == taxonomy.RetryPolicy.DIFFERENT_INSTRUMENT:
            return (ActionType.METHOD_SWITCH_PROMPT, delay) if cause.contact_ok else None

        # A bank outage resolves on its own. Retry; do not message the customer
        # about their bank's downtime.
        if not cause.contact_ok:
            return ActionType.RETRY, delay

        return ActionType.RETRY, delay


class Oracle:
    """The realised ceiling. **Not a competitor -- an upper bound.**

    This policy deliberately reads latent fields and, through the shared draws,
    effectively knows which action would have worked for each case. No real
    system can do this. It exists to answer one question the other arms cannot:
    *of the revenue that was recoverable at all, how much did we get?*

    Without it, "33% of revenue at risk" is uninterpretable. Much of that risk is
    structurally unrecoverable -- risk blocks, merchant configuration, hard
    declines -- or would have self-recovered anyway. The oracle separates
    "our policy is weak" from "there was nothing left to take".

    It is excluded from every headline comparison and reported only as headroom.
    """

    name = "ORACLE_ceiling"
    reads_ground_truth = True

    #: Every action worth trying, evaluated at the cause's best moment.
    CANDIDATES = [
        ActionType.RETRY,
        ActionType.PAYMENT_LINK_WHATSAPP,
        ActionType.METHOD_SWITCH_PROMPT,
    ]

    def __init__(self, max_steps: int = 4):
        # The ceiling is defined *under an action budget*. With unlimited
        # contacts and no fatigue the ceiling would be meaningless -- keep
        # messaging until they pay. Matching the most aggressive baseline's
        # budget makes "% of ceiling" a fair statement rather than a trick.
        self.max_steps = max_steps

    def next_action(self, case, history: list[Step]) -> tuple[ActionType, float] | None:
        if len(history) >= self.max_steps:
            return None

        cause_key = str(case["latent_cause"])
        cause = taxonomy.get(cause_key)
        if cause.retry_policy == taxonomy.RetryPolicy.NEVER:
            return None

        base_delay = max(A.RECOVERABILITY[cause_key].best_delay_h, 0.05)
        # Stagger slightly so attempts are ordered in time without drifting far
        # from the cause's optimal moment.
        delay = base_delay * (1.0 + 0.15 * len(history))

        prior_contacts = sum(1 for h in history if h.action in CONTACT_ACTIONS)
        best, best_p = None, 0.0
        for action in self.CANDIDATES:
            p = _action_success_probability(case, action, delay, prior_contacts, 0.55)
            if p > best_p:
                best, best_p = action, p
        return (best, delay) if best else None


BASELINES = [DoNothing(), RazorpayT3(), ContactEverything(), CauseAware()]

#: Reported alongside the baselines as headroom, never ranked against them.
ORACLE = Oracle()
