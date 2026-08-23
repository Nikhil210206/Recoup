"""Outcome simulator: what actually happens to a case, with or without us.

This is the counterfactual engine. It answers three questions that no real
system can answer about its own traffic:

    1. Would this customer have paid anyway, unprompted?
    2. Did our intervention work?
    3. Given both, did we actually *cause* anything?

Question 3 is the one that matters, and it is the reason this file exists. A
recovery that would have happened regardless is indistinguishable, in production,
from one the system caused. Every recovery metric silently counts both. Here the
counterfactual is known, so "money recovered" can be split into money we caused
and money we merely observed.

## Common random numbers

Every case draws its randomness **once**, deterministically, from its own id.
Every arm -- do-nothing, T+3, contact-everything, amount-ranked, Recoup -- then
evaluates against those same draws.

This is deliberate and load-bearing. If each arm rolled its own dice, a
difference between two arms would mix a real effect with sampling noise, and the
noise does not shrink just because the dataset is large: it is a difference of
two random variables. With common random numbers the comparison becomes paired
-- the same customer, with the same latent willingness, faced by two different
policies -- so a reported difference is the policy, not the dice.

Practically: `Recoup recovered Rs X more than fixed T+3` means, case by case,
this policy did better on the same people, not that it got a luckier draw.
"""

from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass

import numpy as np

from app import taxonomy
from app.simulation import assumptions as A

# How long after the failure a recovery still counts. Beyond this the customer
# has either bought elsewhere or forgotten; attributing a later payment to a
# recovery action would be generous to the point of dishonesty.
RECOVERY_WINDOW_HOURS = 7 * 24


class ActionType(enum.StrEnum):
    """The allowlist. The allocator may propose nothing outside this set."""

    NONE = "none"
    RETRY = "retry"
    PAYMENT_LINK_SMS = "payment_link_sms"
    PAYMENT_LINK_WHATSAPP = "payment_link_whatsapp"
    PAYMENT_LINK_EMAIL = "payment_link_email"
    METHOD_SWITCH_PROMPT = "method_switch_prompt"
    MERCHANT_ALERT = "merchant_alert"
    HUMAN_REVIEW = "human_review"


#: Actions that reach the customer. These consume the shared contact budget and
#: incur fatigue; retries and merchant alerts do not.
CONTACT_ACTIONS = frozenset(
    {
        ActionType.PAYMENT_LINK_SMS,
        ActionType.PAYMENT_LINK_WHATSAPP,
        ActionType.PAYMENT_LINK_EMAIL,
        ActionType.METHOD_SWITCH_PROMPT,
        ActionType.HUMAN_REVIEW,
    }
)

#: Relative conversion by channel, applied on top of the cause's `p_nudge`.
#: WhatsApp outperforms SMS in India and costs more; email is cheap and weak.
CHANNEL_EFFECTIVENESS: dict[ActionType, float] = {
    ActionType.PAYMENT_LINK_WHATSAPP: 1.15,
    ActionType.PAYMENT_LINK_SMS: 1.00,
    ActionType.PAYMENT_LINK_EMAIL: 0.55,
    ActionType.METHOD_SWITCH_PROMPT: 1.00,
    ActionType.HUMAN_REVIEW: 1.60,
}


@dataclass(frozen=True)
class Outcome:
    """What happened to one case under one policy."""

    recovered: bool
    recovered_at_hours: float | None
    amount_recovered_paise: int

    # Attribution. `caused_by_us` is the honest numerator: it excludes recoveries
    # that the counterfactual says would have happened without any action.
    caused_by_us: bool
    would_have_self_recovered: bool

    # Cost side.
    contacts_used: int
    cost_paise: int

    #: Contacted a customer who was going to pay anyway. Invisible to any metric
    #: that only counts recoveries, and a real cost in money and goodwill.
    unnecessary_contact: bool

    reason: str = ""


def case_rng(case_id: str, salt: str = "") -> np.random.Generator:
    """Deterministic per-case generator, independent of evaluation order.

    Seeding from the case id rather than a global stream means arms can be run
    in any order, in parallel, or one at a time, and still see identical draws.
    """
    digest = hashlib.sha256(f"{case_id}|{salt}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


@dataclass(frozen=True)
class CaseDraw:
    """The fixed randomness for one case. Drawn once, shared by every arm."""

    u_self: float             # decides whether they self-recover at all
    self_recover_at_h: float  # when, if they do
    u_retry: tuple[float, ...]   # one per retry attempt
    u_nudge: tuple[float, ...]   # one per contact attempt

    @classmethod
    def for_case(cls, case_id: str, max_attempts: int = 8) -> CaseDraw:
        rng = case_rng(case_id)
        return cls(
            u_self=float(rng.random()),
            # Self-recovery is heavily front-loaded: a customer who was going to
            # retry unaided usually does it soon, not on day six.
            self_recover_at_h=float(np.clip(rng.lognormal(1.6, 1.3), 0.05, RECOVERY_WINDOW_HOURS)),
            u_retry=tuple(float(x) for x in rng.random(max_attempts)),
            u_nudge=tuple(float(x) for x in rng.random(max_attempts)),
        )


def _timing_factor(delay_h: float, best_delay_h: float) -> float:
    """How much of a retry's potential survives acting at the wrong moment.

    A log-normal kernel around the cause's best delay. Acting at the right time
    keeps ~all of it; acting an order of magnitude early or late keeps little.

    The width is what makes timing matter differently per cause. An
    authentication failure decays in minutes -- the customer is still at the
    checkout -- while insufficient funds is gated on a balance changing, so a
    retry an hour later is near-worthless and one after payday is not. A fixed
    schedule cannot express that, which is precisely where T+3 loses.
    """
    if best_delay_h <= 0:
        return 1.0 if delay_h <= 1.0 else 0.2
    delay_h = max(delay_h, 1e-3)
    ratio = np.log(delay_h / best_delay_h)
    return float(np.exp(-0.5 * (ratio / 1.25) ** 2))


def _self_recovery(case, draw: CaseDraw) -> tuple[bool, float | None]:
    """Would this customer have paid without any intervention, and when."""
    if draw.u_self < case["latent_p_self_recover"]:
        return True, draw.self_recover_at_h
    return False, None


def simulate_no_action(case) -> Outcome:
    """The do-nothing baseline. Everything else is measured against this."""
    draw = CaseDraw.for_case(case["case_id"])
    self_rec, at_h = _self_recovery(case, draw)
    return Outcome(
        recovered=self_rec,
        recovered_at_hours=at_h,
        amount_recovered_paise=int(case["amount_paise"]) if self_rec else 0,
        caused_by_us=False,
        would_have_self_recovered=self_rec,
        contacts_used=0,
        cost_paise=0,
        unnecessary_contact=False,
        reason="no_action",
    )


def _action_success_probability(
    case, action: ActionType, delay_h: float, prior_contacts: int, fatigue_lambda: float
) -> float:
    """Probability this specific action recovers this specific case."""
    cause = taxonomy.get(case["latent_cause"])

    if action in (ActionType.NONE, ActionType.MERCHANT_ALERT):
        return 0.0

    if action == ActionType.RETRY:
        # Retrying an instrument that structurally cannot work does not become
        # more likely with better timing.
        return float(case["latent_p_retry"]) * _timing_factor(
            delay_h, float(case["latent_best_delay_h"])
        )

    # Contact actions. Fatigue compounds per prior contact to this customer --
    # this is the term the shared contact budget exists to defend.
    p = float(case["latent_p_nudge"]) * CHANNEL_EFFECTIVENESS.get(action, 1.0)
    p *= fatigue_lambda ** prior_contacts

    # A prompt to switch instrument is the only thing that helps when the
    # instrument itself is the problem, and it is wasted effort when it is not.
    if action == ActionType.METHOD_SWITCH_PROMPT:
        if cause.retry_policy == taxonomy.RetryPolicy.DIFFERENT_INSTRUMENT:
            p *= 1.45
        else:
            p *= 0.70

    return float(np.clip(p, 0.0, 0.95))


def simulate_action(
    case,
    action: ActionType,
    *,
    delay_h: float,
    attempt_idx: int = 0,
    prior_contacts: int = 0,
    fatigue_lambda: float = A.CONTACT_FATIGUE_LAMBDA.value,
) -> Outcome:
    """Simulate one intervention against a case.

    Attribution rules, in order:

      - If the customer would have self-recovered *before* we acted, the money
        was never ours to claim. If we acted anyway and it reached them, that is
        an unnecessary contact: real cost, zero incremental revenue.
      - If our action succeeds and no self-recovery was coming, we caused it.
      - If our action succeeds but a self-recovery was also coming later, the
        revenue is real but not incremental. Counting it would inflate the
        headline, so it is recorded and excluded from the causal numerator.
    """
    draw = CaseDraw.for_case(case["case_id"])
    self_rec, self_at = _self_recovery(case, draw)
    amount = int(case["amount_paise"])
    is_contact = action in CONTACT_ACTIONS
    cost = A.ACTION_COST_PAISE[str(action)].value if str(action) in A.ACTION_COST_PAISE else 0
    cost = int(cost)

    # 1. Already paid before this action was due.
    #
    # The action never fires: a real recovery system reads payment status before
    # sending anything, and a paid order stops the workflow. So no contact is
    # consumed and no cost is incurred.
    #
    # This is deliberately *not* counted as an unnecessary contact either. An
    # unnecessary contact means reaching someone who was going to pay anyway, at
    # a moment when that was unknowable. Messaging someone who has already paid
    # is a different thing -- a status-checking bug, not a policy decision -- and
    # conflating them charges contact-heavy arms for messages they would never
    # actually send.
    if self_rec and self_at is not None and self_at <= delay_h:
        return Outcome(
            recovered=True,
            recovered_at_hours=self_at,
            amount_recovered_paise=amount,
            caused_by_us=False,
            would_have_self_recovered=True,
            contacts_used=0,
            cost_paise=0,
            unnecessary_contact=False,
            reason="self_recovered_before_action",
        )

    if delay_h > RECOVERY_WINDOW_HOURS:
        return Outcome(
            False, None, 0, False, self_rec, 0, 0, False, reason="acted_outside_window"
        )

    p = _action_success_probability(case, action, delay_h, prior_contacts, fatigue_lambda)
    stream = draw.u_nudge if is_contact else draw.u_retry
    u = stream[min(attempt_idx, len(stream) - 1)]

    # 2. The action worked.
    if u < p:
        return Outcome(
            recovered=True,
            recovered_at_hours=delay_h,
            amount_recovered_paise=amount,
            # Incremental only if nothing was coming anyway.
            caused_by_us=not self_rec,
            would_have_self_recovered=self_rec,
            contacts_used=1 if is_contact else 0,
            cost_paise=cost,
            # Reaching someone who was going to pay later still burns a contact
            # and a little goodwill, even though it "worked".
            unnecessary_contact=is_contact and self_rec,
            reason="recovered_by_action",
        )

    # 3. The action failed.
    #
    # A pending self-recovery is deliberately *not* resolved here. This function
    # reports only what this action did; the episode runner owns the timeline and
    # applies any self-recovery once the policy has finished acting.
    #
    # Returning "recovered" here was a real bug: an early failed retry with a
    # self-recovery pending days later ended the episode, so every subsequent
    # contact the policy would have sent never fired. Contact-everything was
    # scored as cheaper and far less spammy than it actually is.
    return Outcome(
        recovered=False,
        recovered_at_hours=None,
        amount_recovered_paise=0,
        caused_by_us=False,
        would_have_self_recovered=self_rec,
        contacts_used=1 if is_contact else 0,
        cost_paise=cost,
        # Reaching someone who is going to pay anyway is a wasted contact
        # whether or not this particular message converted.
        unnecessary_contact=is_contact and self_rec,
        reason="action_failed",
    )
