"""Feature extraction. Observable fields only.

Every feature here is something a live system would hold at decision time: the
Razorpay payload, the loss channel, and the customer's own history up to this
failure. Nothing derived from `latent_*` may appear, and a test walks this
module's AST to enforce that rather than relying on care.

The `customer_*` history fields are safe specifically because the generator
builds them forward in time -- a two-phase pass added after the first version
accumulated history in processing order and let 41% of customers carry "prior"
events that happened later in wall-clock time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.simulation.policies import observable_cause

#: Categorical features, one-hot encoded.
CATEGORICAL = [
    "channel",
    "cause",
    "method",
    "issuer",
    "customer_segment",
    "merchant_category",
]

#: Numeric features, used as-is.
NUMERIC = [
    "log_amount",
    "hour_ist",
    "is_peak_hour",
    "is_weekend",
    "customer_prior_attempts",
    "customer_prior_failures",
    "customer_observed_success_rate",
    "has_prior_history",
    "log_hours_since_last_failure",
    "has_prior_failure",
    "log_customer_ltv",
]

#: Added only to the treated model: what we did and when.
TREATMENT = ["action", "log_delay_h", "prior_contacts"]


def build(cases: pd.DataFrame) -> pd.DataFrame:
    """Observable feature frame for a set of cases."""
    out = pd.DataFrame(index=cases.index)

    # Money spans several orders of magnitude; the log is what carries signal.
    out["log_amount"] = np.log1p(cases["amount_paise"].astype(float))
    out["log_customer_ltv"] = np.log1p(cases["customer_ltv_paise"].astype(float))

    out["hour_ist"] = cases["hour_ist"].astype(float)
    out["is_peak_hour"] = cases["is_peak_hour"].astype(float)
    out["is_weekend"] = cases["is_weekend"].astype(float)

    out["customer_prior_attempts"] = cases["customer_prior_attempts"].astype(float)
    out["customer_prior_failures"] = cases["customer_prior_failures"].astype(float)
    out["has_prior_history"] = cases["has_prior_history"].astype(float)
    out["has_prior_failure"] = cases["has_prior_failure"].astype(float)

    # Missing history is genuinely missing, not zero. A customer with no prior
    # attempts has an undefined success rate, and encoding that as 0.0 would tell
    # the model they always fail. Median-fill with an explicit presence flag.
    rate = cases["customer_observed_success_rate"].astype(float)
    out["customer_observed_success_rate"] = rate.fillna(rate.median())

    gap = cases["hours_since_last_failure"].astype(float)
    out["log_hours_since_last_failure"] = np.log1p(gap.fillna(0.0).clip(lower=0))

    # The cause a live system could determine: Razorpay's error fields, or the
    # channel where no error exists. Never `latent_cause`.
    out["cause"] = cases.apply(observable_cause, axis=1).fillna("unknown")

    for col in ("channel", "method", "issuer", "customer_segment", "merchant_category"):
        out[col] = cases[col].astype(str)

    return out


def add_treatment(
    features: pd.DataFrame,
    actions: pd.Series,
    delays_h: pd.Series,
    prior_contacts: pd.Series,
) -> pd.DataFrame:
    """Attach what was done, for the treated model.

    Timing is logged because the effect is multiplicative and spans minutes to
    days -- an authentication failure decays in minutes, insufficient funds is
    gated on a balance changing.
    """
    out = features.copy()
    out["action"] = actions.astype(str).to_numpy()
    out["log_delay_h"] = np.log1p(delays_h.astype(float).clip(lower=0)).to_numpy()
    out["prior_contacts"] = prior_contacts.astype(float).to_numpy()
    return out


def columns(treated: bool) -> tuple[list[str], list[str]]:
    """(categorical, numeric) column names for the requested model."""
    cat = [*CATEGORICAL, "action"] if treated else list(CATEGORICAL)
    num = [*NUMERIC, "log_delay_h", "prior_contacts"] if treated else list(NUMERIC)
    return cat, num
