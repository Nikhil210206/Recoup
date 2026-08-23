"""Every assumption behind the synthetic data, in one file, each one labelled.

This module exists so that no number in the evaluation is unattributed. Each
parameter carries a `basis` describing where it came from, using three levels:

    SOURCED   -- taken from published figures (Razorpay docs, NPCI/UPI reporting)
    ANCHORED  -- derived from a sourced figure plus a stated inference
    ESTIMATE  -- my judgement. Not sourced. Sensitivity-tested rather than trusted.

Most of the behavioural parameters are ESTIMATE, and that is the honest position:
nobody publishes conditional recovery probabilities by failure cause, because
that is precisely the proprietary knowledge a payments company accumulates. The
response is not to dress estimates up as facts. It is to (a) label them, (b) hold
them fixed and reproducible, and (c) re-run the whole evaluation under different
parameterisations to see which conclusions survive -- see the world sweep.

Read `data/ASSUMPTIONS.md` for the same content in prose.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class Basis(enum.StrEnum):
    SOURCED = "sourced"
    ANCHORED = "anchored"
    ESTIMATE = "estimate"


@dataclass(frozen=True)
class Param:
    value: float
    basis: Basis
    note: str


# ---------------------------------------------------------------------------
# Volume and failure rates
# ---------------------------------------------------------------------------

# Blended merchant-level success rates in India sit around 92-96%, with below
# 90% treated as a serious business problem. We model a merchant at the weaker
# end of healthy, because a merchant with nothing to recover is not a merchant
# who would deploy this.
BLENDED_SUCCESS_RATE = Param(
    0.935,
    Basis.SOURCED,
    "Published Indian merchant blended success rates cluster at 92-96%.",
)

# System-wide UPI technical declines fell to roughly 0.7-0.8% by 2025. This is
# the floor of unavoidable infrastructure failure, distinct from business
# declines (wrong PIN, insufficient funds) which dominate the remainder.
TECHNICAL_DECLINE_RATE = Param(
    0.008,
    Basis.SOURCED,
    "UPI technical decline rate ~0.8% (NPCI reporting, 2024-25).",
)

# Razorpay states that more than half of payment failures come from customer
# errors or network issues.
CUSTOMER_SIDE_SHARE = Param(
    0.55,
    Basis.SOURCED,
    "Razorpay: '>50% payment failures are due to customer errors or network issues'.",
)

# Success dips during peak hours when issuers are overloaded; reported dips to
# 80-85% are common. Modelled as a multiplier on the base success rate.
PEAK_HOUR_SUCCESS_MULTIPLIER = Param(
    0.90,
    Basis.ANCHORED,
    "Derived from reported peak dips to 80-85% against a 93.5% base.",
)
PEAK_HOURS_IST: tuple[int, ...] = (12, 13, 19, 20, 21)

# ---------------------------------------------------------------------------
# Payment method mix
# ---------------------------------------------------------------------------

# UPI dominates Indian online payments by volume; cards are a distant second and
# skew to higher ticket sizes. Exact split varies enormously by merchant
# category, so this is a judgement call, not a published figure.
METHOD_MIX: dict[str, Param] = {
    "upi": Param(0.62, Basis.ESTIMATE, "UPI-dominant mix typical of Indian D2C."),
    "card": Param(0.22, Basis.ESTIMATE, "Skews to higher ticket sizes."),
    "netbanking": Param(0.09, Basis.ESTIMATE, "Higher ticket, lower volume."),
    "wallet": Param(0.07, Basis.ESTIMATE, "Small-ticket tail."),
}

# ---------------------------------------------------------------------------
# Cause distribution, conditional on method
# ---------------------------------------------------------------------------
# Anchored on two sourced constraints -- customer-side causes carry ~55% of the
# mass, and bank/gateway technical declines are ~1% -- with the split *within*
# those bands estimated. The failure profile genuinely differs by method: UPI
# fails at collect and cancellation, cards fail at authentication and decline.

CAUSE_MIX_BY_METHOD: dict[str, dict[str, float]] = {
    "upi": {
        "collect_expired": 0.26,
        "customer_abandoned": 0.22,
        "insufficient_funds": 0.16,
        "authentication_failed": 0.10,
        "transient_bank_downtime": 0.09,
        "invalid_instrument": 0.07,
        "hard_decline": 0.05,
        "limit_exceeded": 0.03,
        "account_mismatch": 0.01,
        "risk_blocked": 0.01,
    },
    "card": {
        "authentication_failed": 0.27,
        "hard_decline": 0.18,
        "insufficient_funds": 0.13,
        "card_disabled_online": 0.10,
        "customer_abandoned": 0.08,
        "transient_bank_downtime": 0.07,
        "card_expired": 0.05,
        # Merchant-configuration failures (e.g. international cards disabled).
        # Low frequency, high asymmetry: every customer-facing recovery action
        # is wrong, so it disproportionately punishes agents that treat all
        # failures as "chase the customer". Found on a real payload, day 0.
        "merchant_config": 0.04,
        "limit_exceeded": 0.03,
        "risk_blocked": 0.03,
        "card_blocked": 0.02,
    },
    "netbanking": {
        "transient_bank_downtime": 0.30,
        "customer_abandoned": 0.25,
        "authentication_failed": 0.18,
        "insufficient_funds": 0.15,
        "hard_decline": 0.09,
        "risk_blocked": 0.03,
    },
    "wallet": {
        "insufficient_funds": 0.38,
        "customer_abandoned": 0.24,
        "transient_bank_downtime": 0.16,
        "authentication_failed": 0.12,
        "hard_decline": 0.07,
        "risk_blocked": 0.03,
    },
}

# ---------------------------------------------------------------------------
# Loss channels
# ---------------------------------------------------------------------------
# The allocator's whole premise is arbitrating *between* these under one shared
# contact budget, so they have to differ in ways that actually pull against each
# other:
#
#   failed_payment       a payment attempt failed. Retry is possible and cheap.
#   failed_subscription  a mandate exists, so a retry needs no customer contact
#                        at all -- recovery can cost zero contacts.
#   abandoned_checkout   nothing failed; the customer left. There is no payment
#                        to retry, so the ONLY route to the money is a contact.
#
# That asymmetry is the trade-off nothing in Agent Studio currently arbitrates:
# subscription retries are nearly free, abandoned carts are contact-only, and
# both compete for the same finite tolerance of one customer.

CHANNEL_MIX: dict[str, Param] = {
    "failed_payment": Param(0.62, Basis.ESTIMATE, "Highest volume for a typical D2C merchant."),
    "abandoned_checkout": Param(
        0.30, Basis.ESTIMATE, "Drop-off is high volume relative to payment failure."
    ),
    "failed_subscription": Param(0.08, Basis.ESTIMATE, "Only merchants running recurring plans."),
}

#: Abandoned checkouts carry no Razorpay error code -- nothing failed. The cause
#: is structural, and `p_retry` is zero because there is no payment to re-attempt.
ABANDONED_CHECKOUT_CAUSE = "customer_abandoned"

#: Subscription charge failures skew to funds and instrument problems rather than
#: authentication: no one is at a checkout typing an OTP, a mandate is firing.
SUBSCRIPTION_CAUSE_MIX: dict[str, float] = {
    "insufficient_funds": 0.38,
    "card_expired": 0.16,
    "hard_decline": 0.14,
    "limit_exceeded": 0.10,
    "transient_bank_downtime": 0.09,
    "card_disabled_online": 0.06,
    "card_blocked": 0.04,
    "risk_blocked": 0.03,
}

# ---------------------------------------------------------------------------
# Latent recoverability -- the counterfactual ground truth
# ---------------------------------------------------------------------------
# These are the parameters the agent never observes. They define, per cause:
#
#   p_self_recover  probability the customer pays anyway, unprompted, inside the
#                   window. This is the denominator that makes "unnecessary
#                   contact" measurable: contacting someone who would have paid
#                   regardless is a cost with no matching benefit, and it is
#                   invisible to any metric that only counts recoveries.
#
#   p_retry         probability a well-timed retry of the same instrument works.
#   p_nudge         probability a message converts, given no self-recovery.
#   best_delay_h    when the retry is most likely to land.
#
# All ESTIMATE. Their *relative ordering* is the load-bearing claim, not their
# absolute values: expired collect requests recover far more readily than hard
# declines, and no plausible parameterisation reverses that.


@dataclass(frozen=True)
class Recoverability:
    p_self_recover: float
    p_retry: float
    p_nudge: float
    best_delay_h: float
    basis: Basis = Basis.ESTIMATE
    note: str = ""


RECOVERABILITY: dict[str, Recoverability] = {
    "collect_expired": Recoverability(
        0.18,
        0.55,
        0.48,
        0.5,
        note="Customer often never saw the request; re-sending converts well.",
    ),
    "authentication_failed": Recoverability(
        0.34,
        0.62,
        0.40,
        0.05,
        note="Highest intent, fastest decay. Many retry unaided within minutes.",
    ),
    "customer_abandoned": Recoverability(
        0.22,
        0.30,
        0.34,
        2.0,
        note="Genuine intent and a change of mind are indistinguishable here.",
    ),
    "insufficient_funds": Recoverability(
        0.16,
        0.28,
        0.31,
        48.0,
        note="Recovery is gated on the balance changing, not on persuasion.",
    ),
    "transient_bank_downtime": Recoverability(
        0.30,
        0.68,
        0.10,
        3.0,
        note="Resolves on its own once the issuer recovers. Retry, do not contact.",
    ),
    "limit_exceeded": Recoverability(
        0.14,
        0.45,
        0.22,
        26.0,
        note="Daily limits reset; a next-day retry is materially better.",
    ),
    "account_mismatch": Recoverability(
        0.12,
        0.35,
        0.44,
        1.0,
        note="Needs an instruction, not a retry.",
    ),
    "invalid_instrument": Recoverability(
        0.08,
        0.04,
        0.30,
        1.0,
        note="Same VPA fails identically; only a method switch helps.",
    ),
    "card_expired": Recoverability(
        0.06,
        0.02,
        0.35,
        1.0,
        note="Deterministically dead on this card, fine on another.",
    ),
    "card_disabled_online": Recoverability(
        0.07,
        0.05,
        0.33,
        1.0,
        note="Fixable, but only by a customer who is told what to fix.",
    ),
    "card_blocked": Recoverability(
        0.05,
        0.03,
        0.18,
        2.0,
        note="Often follows a fraud report.",
    ),
    "hard_decline": Recoverability(
        0.09,
        0.12,
        0.16,
        6.0,
        note="Issuer refused without an actionable reason.",
    ),
    "risk_blocked": Recoverability(
        0.01,
        0.01,
        0.01,
        0.0,
        note="Treated as unrecoverable by policy, not by probability.",
    ),
    "merchant_config": Recoverability(
        0.02,
        0.00,
        0.00,
        0.0,
        note=(
            "Zero by construction. No customer action can clear a merchant "
            "setting, so every customer-directed recovery has expected value zero "
            "and non-zero cost."
        ),
    ),
}

# ---------------------------------------------------------------------------
# Contact fatigue
# ---------------------------------------------------------------------------
# Each additional contact to the same customer multiplies the response
# probability by this factor. This is what turns "contact everyone repeatedly"
# into a losing strategy, and it is the term the allocator's shared budget is
# defending. ESTIMATE, and one of the most important parameters to sweep.
CONTACT_FATIGUE_LAMBDA = Param(
    0.55,
    Basis.ESTIMATE,
    "Each prior contact scales response probability by 0.55. Swept in world sweep.",
)

# Roughly 60% of customers report abandoning a brand after a failed payment.
# Used to bound how aggressive contact can be before it costs future revenue.
BRAND_ABANDONMENT_AFTER_FAILURE = Param(
    0.60,
    Basis.SOURCED,
    "~60% of customers abandon a brand after a failed payment experience.",
)

# ---------------------------------------------------------------------------
# Intervention costs (paise)
# ---------------------------------------------------------------------------
# Real per-message costs for Indian channels, order-of-magnitude. These make EV
# a net figure rather than a gross one.
ACTION_COST_PAISE: dict[str, Param] = {
    "retry": Param(0, Basis.ANCHORED, "Gateway retry has no per-attempt customer cost."),
    "payment_link_sms": Param(25, Basis.ESTIMATE, "~INR 0.25 per transactional SMS."),
    "payment_link_whatsapp": Param(80, Basis.ESTIMATE, "~INR 0.80 per WhatsApp utility message."),
    "payment_link_email": Param(2, Basis.ESTIMATE, "Negligible per-email cost."),
    "method_switch_prompt": Param(25, Basis.ESTIMATE, "Delivered over the same channels."),
    "merchant_alert": Param(0, Basis.ANCHORED, "Internal notification, no external cost."),
    "human_review": Param(4000, Basis.ESTIMATE, "~INR 40 of an agent's time per case."),
}

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
# Razorpay's documented subscription retry policy: retry the next day, three
# times over a T+3 cycle, regardless of cause, issuer, or customer. This is the
# honest comparator -- the policy a merchant on Razorpay actually gets today.
RAZORPAY_T_PLUS_3 = {
    "retries": 3,
    "interval_hours": 24,
    "basis": Basis.SOURCED,
    "note": "Razorpay Subscriptions: automatic retry next day, thrice over T+3.",
}

# Razorpay's Failed Payment Recovery product claims up to 20% of failed payments
# recovered via multichannel retargeting. Used as an external sanity band: a
# simulation reporting 70% recovery would be modelling a world that does not exist.
PUBLISHED_RECOVERY_CEILING = Param(
    0.20,
    Basis.SOURCED,
    "Razorpay Failed Payment Recovery: 'recover up to 20% of failed payments'.",
)

# ---------------------------------------------------------------------------
# Dataset shape
# ---------------------------------------------------------------------------

DEFAULT_SEED = 42
DEFAULT_DAYS = 90  # long enough for a clean time-based split
TARGET_CASES = 10_000  # >= 8,000 required; >= 2,000 reserved for calibration
CALIBRATION_HOLDOUT = 0.20
TEST_HOLDOUT = 0.20  # final 20% of the timeline, never trained on


def summary() -> list[tuple[str, str, str]]:
    """(parameter, basis, note) for every declared assumption, for the docs."""
    rows: list[tuple[str, str, str]] = []
    for name, obj in sorted(globals().items()):
        if isinstance(obj, Param):
            rows.append((name, obj.basis.value, obj.note))
    for method, p in METHOD_MIX.items():
        rows.append((f"METHOD_MIX[{method}]", p.basis.value, p.note))
    for action, p in ACTION_COST_PAISE.items():
        rows.append((f"ACTION_COST_PAISE[{action}]", p.basis.value, p.note))
    return rows
