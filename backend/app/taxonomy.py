"""Canonical cause taxonomy, mapped from Razorpay's own error reasons.

Every `error_reason` string below is taken from Razorpay's published error code
tables for cards and UPI. Nothing here is invented, which matters: the same map
classifies a real webhook payload and a synthetic one, so the classifier is
never tested against a taxonomy built to flatter it.

**This is deliberately not an LLM.** `error_reason` is a finite, documented enum.
A lookup is faster, free, reproducible, and strictly more accurate than a model
guessing at a closed set. The LLM is reserved for reasons that fall outside this
map -- see `classify()`, which routes the unmapped tail to an exception path.

The recovery semantics on each cause are the point of the file. "Payment failed"
is not a decision; "the customer cannot fix this and contacting them is harmful"
is. Three axes drive every downstream action:

    who_can_fix   -- customer, merchant, bank, or nobody
    retry_policy  -- is retrying the same instrument ever going to work
    contact_ok    -- would reaching out to the customer be useful or insulting
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class WhoCanFix(enum.StrEnum):
    CUSTOMER = "customer"  # the payer can act: retry, top up, use another card
    MERCHANT = "merchant"  # the merchant's own configuration caused this
    BANK = "bank"  # issuer or network side; nobody downstream can act
    NOBODY = "nobody"  # structurally unrecoverable on this instrument


class RetryPolicy(enum.StrEnum):
    IMMEDIATE = "immediate"  # a prompt retry has a real chance
    DELAYED = "delayed"  # retry later; the blocker is time-dependent
    DIFFERENT_INSTRUMENT = "different_instrument"  # same instrument will fail again
    NEVER = "never"  # retrying is pointless, or abusive


@dataclass(frozen=True)
class Cause:
    """A canonical failure cause and what it implies for recovery."""

    key: str
    label: str
    who_can_fix: WhoCanFix
    retry_policy: RetryPolicy
    contact_ok: bool
    note: str


# --- Canonical causes ------------------------------------------------------

CAUSES: dict[str, Cause] = {
    c.key: c
    for c in [
        Cause(
            "transient_bank_downtime",
            "Bank or gateway downtime",
            WhoCanFix.BANK,
            RetryPolicy.DELAYED,
            contact_ok=False,
            note=(
                "Nothing the customer did caused this and nothing they can do fixes "
                "it. Retry once the issuer recovers. Messaging them invites a second "
                "failure and reads as blaming them for a bank outage."
            ),
        ),
        Cause(
            "insufficient_funds",
            "Insufficient funds",
            WhoCanFix.CUSTOMER,
            RetryPolicy.DELAYED,
            contact_ok=True,
            note=(
                "Genuinely recoverable, but only after the balance changes. Timing "
                "dominates: a retry an hour later is near-worthless, one after "
                "payday is not."
            ),
        ),
        Cause(
            "authentication_failed",
            "Authentication failed (OTP or CVV)",
            WhoCanFix.CUSTOMER,
            RetryPolicy.IMMEDIATE,
            contact_ok=True,
            note=(
                "Highest-intent failure class. The customer was actively paying and "
                "mistyped. Recovery value decays fast -- minutes, not days."
            ),
        ),
        Cause(
            "customer_abandoned",
            "Customer cancelled or went back",
            WhoCanFix.CUSTOMER,
            RetryPolicy.IMMEDIATE,
            contact_ok=True,
            note=(
                "Intent is ambiguous: a change of mind and a mis-tap look identical "
                "in the payload. Treat as recoverable but low-confidence."
            ),
        ),
        Cause(
            "collect_expired",
            "Collect request expired or timed out",
            WhoCanFix.CUSTOMER,
            RetryPolicy.IMMEDIATE,
            contact_ok=True,
            note=(
                "The customer usually never saw the request. Re-sending is cheap and "
                "often works, which makes this one of the best EV cases."
            ),
        ),
        Cause(
            "invalid_instrument",
            "Invalid or unresolvable VPA",
            WhoCanFix.CUSTOMER,
            RetryPolicy.DIFFERENT_INSTRUMENT,
            contact_ok=True,
            note="The UPI ID does not resolve. Retrying it repeats the failure exactly.",
        ),
        Cause(
            "card_expired",
            "Card expired",
            WhoCanFix.CUSTOMER,
            RetryPolicy.DIFFERENT_INSTRUMENT,
            contact_ok=True,
            note=(
                "Deterministically unrecoverable on this card and deterministically "
                "recoverable on another. A card-update prompt, never a plain retry."
            ),
        ),
        Cause(
            "card_disabled_online",
            "Card not enabled for online payments",
            WhoCanFix.CUSTOMER,
            RetryPolicy.DIFFERENT_INSTRUMENT,
            contact_ok=True,
            note=(
                "Fixable by the customer, but only if told what to fix. A generic "
                "'your payment failed, try again' wastes the contact entirely."
            ),
        ),
        Cause(
            "card_blocked",
            "Card blocked by customer or bank",
            WhoCanFix.BANK,
            RetryPolicy.DIFFERENT_INSTRUMENT,
            contact_ok=True,
            note="Often follows a fraud report. Suggest an alternative, never a retry.",
        ),
        Cause(
            "hard_decline",
            "Declined by the issuer",
            WhoCanFix.BANK,
            RetryPolicy.DIFFERENT_INSTRUMENT,
            contact_ok=True,
            note=(
                "The issuer refused without an actionable reason. Repeated retries "
                "against a hard decline can itself look like card testing."
            ),
        ),
        Cause(
            "risk_blocked",
            "Blocked by a risk or fraud check",
            WhoCanFix.NOBODY,
            RetryPolicy.NEVER,
            contact_ok=False,
            note=(
                "Never retry and never contact. If the block is correct, recovery "
                "means recovering a fraudulent payment; if it is a false positive, "
                "nudging the customer cannot clear it. Suppress and move on."
            ),
        ),
        Cause(
            "limit_exceeded",
            "Transaction limit exceeded",
            WhoCanFix.CUSTOMER,
            RetryPolicy.DELAYED,
            contact_ok=True,
            note="Daily limits reset. A next-day retry is materially better than an hourly one.",
        ),
        Cause(
            "merchant_config",
            "Blocked by the merchant's own configuration",
            WhoCanFix.MERCHANT,
            RetryPolicy.NEVER,
            contact_ok=False,
            note=(
                "The merchant is the cause. Every customer-facing recovery action is "
                "wrong here: retrying fails identically forever, and a nudge blames "
                "the customer for a setting only the merchant controls. The correct "
                "output is a merchant alert and zero customer contact. Found this on "
                "day 0 from a real payload -- see INCIDENTS.md."
            ),
        ),
        Cause(
            "account_mismatch",
            "Different bank account than the one registered",
            WhoCanFix.CUSTOMER,
            RetryPolicy.IMMEDIATE,
            contact_ok=True,
            note="Recoverable immediately, but only with an instruction attached.",
        ),
    ]
}


# --- Razorpay error_reason -> canonical cause ------------------------------
# Sourced from Razorpay's published cards and UPI error code tables.

REASON_TO_CAUSE: dict[str, str] = {
    # Bank / gateway side
    "bank_technical_error": "transient_bank_downtime",
    "gateway_technical_error": "transient_bank_downtime",
    "server_error": "transient_bank_downtime",
    # Funds
    "insufficient_funds": "insufficient_funds",
    # Authentication
    "authentication_failed": "authentication_failed",
    "incorrect_cvv": "authentication_failed",
    "invalid_otp": "authentication_failed",
    "incorrect_otp": "authentication_failed",
    # Abandonment
    "payment_cancelled": "customer_abandoned",
    # Expiry / timeout
    "payment_collect_request_expired": "collect_expired",
    "payment_timed_out": "collect_expired",
    # Instrument validity
    "invalid_vpa": "invalid_instrument",
    "vpa_resolution_failed": "invalid_instrument",
    "card_expired": "card_expired",
    "card_not_enrolled": "card_disabled_online",
    "card_disabled_for_online_payments": "card_disabled_online",
    "debit_instrument_inactive": "card_disabled_online",
    "debit_instrument_blocked": "card_blocked",
    # Declines
    "card_declined": "hard_decline",
    "payment_declined": "hard_decline",
    "payment_failed": "hard_decline",
    "credit_failed": "hard_decline",
    # Risk
    "payment_risk_check_failed": "risk_blocked",
    # Limits
    "transaction_limit_exceeded": "limit_exceeded",
    # Merchant configuration
    "international_transaction_not_allowed": "merchant_config",
    # Registration mismatch
    "payer_account_mismatch": "account_mismatch",
}


def _as_text(value: object) -> str | None:
    """Normalise a possibly-missing field to a string or None.

    Webhooks omit fields; DataFrames represent the same absence as NaN. Both must
    behave identically here, because the same classifier serves both paths.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    if isinstance(value, float) and value != value:  # NaN
        return None
    return str(value)


@dataclass(frozen=True)
class Classification:
    cause: str | None
    confidence: float
    method: str  # "deterministic" | "unmapped"
    matched_on: str | None


def classify(
    error_reason: str | None,
    error_source: str | None = None,
    error_step: str | None = None,
) -> Classification:
    """Deterministic classification from Razorpay's error fields.

    Returns `cause=None` with `method="unmapped"` when the reason is not in the
    published tables. That is not a failure -- it is the hand-off point to the
    LLM tail and, failing that, to the exception list. Guessing a cause here
    would be worse than admitting we do not know one.
    """
    # Coerce defensively. A missing field arrives as None from a webhook and as
    # NaN from a DataFrame, and both used to reach `.strip()` and raise.
    error_reason = _as_text(error_reason)
    error_source = _as_text(error_source)

    if error_reason:
        key = error_reason.strip().lower()
        cause = REASON_TO_CAUSE.get(key)
        if cause:
            return Classification(cause, 1.0, "deterministic", f"error_reason={key}")

    # `error_source` alone is a coarse but honest fallback. A business-source
    # failure is the merchant's configuration whatever the specific reason, and
    # that alone is enough to suppress customer contact.
    if (error_source or "").lower() == "business":
        return Classification("merchant_config", 0.6, "deterministic", "error_source=business")

    return Classification(None, 0.0, "unmapped", None)


def get(cause_key: str) -> Cause:
    return CAUSES[cause_key]


def all_causes() -> list[Cause]:
    return list(CAUSES.values())
