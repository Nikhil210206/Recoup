"""Reproducible synthetic dataset of revenue-at-risk cases.

What this generator is, and is not:

It is a **testbed for the allocator**. It produces cases whose failure reasons
are Razorpay's real published codes, whose distribution is anchored on published
Indian success-rate benchmarks, and whose latent recoverability is hidden from
everything downstream.

It is **not evidence that the recovery model is accurate**. The generating
process is known to me, so a model trained on this data and scored on this data
is measuring gradient descent, not payments. That circularity is real and is
addressed by attacking the allocator two separate ways -- degrading the
probability oracle, and re-running under different world parameterisations --
rather than by pretending the simulator validates the model. See EVALUATION.md.

Every case carries two disjoint sets of fields:

    observable  what a real system would see at decision time
    latent      the counterfactual truth (would they have paid anyway?)

Nothing outside the outcome simulator may read a `latent_` field. That boundary
is what makes "unnecessary contact" measurable at all: without a counterfactual,
a recovery that would have happened regardless is indistinguishable from one the
system caused, and every recovery metric silently inflates.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from app.simulation import assumptions as A

# Razorpay error fields, reversed out of the canonical cause so that synthetic
# payloads are shaped exactly like real ones and hit the same classifier.
CAUSE_TO_RAZORPAY: dict[str, tuple[str, str, str]] = {
    # cause -> (error_reason, error_source, error_step)
    "collect_expired": ("payment_collect_request_expired", "customer", "payment_authorization"),
    "authentication_failed": ("authentication_failed", "customer", "payment_authentication"),
    "customer_abandoned": ("payment_cancelled", "customer", "payment_authentication"),
    "insufficient_funds": ("insufficient_funds", "customer", "payment_authorization"),
    "transient_bank_downtime": ("bank_technical_error", "bank", "payment_authorization"),
    "limit_exceeded": ("transaction_limit_exceeded", "customer", "payment_authorization"),
    "account_mismatch": ("payer_account_mismatch", "customer", "payment_authorization"),
    "invalid_instrument": ("invalid_vpa", "customer", "payment_initiation"),
    "card_expired": ("card_expired", "customer", "payment_initiation"),
    "card_disabled_online": ("card_disabled_for_online_payments", "customer", "payment_initiation"),
    "card_blocked": ("debit_instrument_blocked", "issuer", "payment_initiation"),
    "hard_decline": ("card_declined", "issuer", "payment_authorization"),
    "risk_blocked": ("payment_risk_check_failed", "issuer", "payment_authorization"),
    "merchant_config": ("international_transaction_not_allowed", "business", "payment_initiation"),
}

ISSUERS = ["HDFC", "ICICI", "SBI", "AXIS", "KOTAK", "PAYTM", "YESBANK", "IDFC"]

# Issuer share is deliberately uneven. Concentration is what makes an issuer-level
# outage show up as a correlated spike rather than uniform noise, which is the
# pattern the merchant-facing narrative has to detect.
ISSUER_WEIGHTS = np.array([0.24, 0.19, 0.17, 0.12, 0.09, 0.08, 0.06, 0.05])


@dataclass(frozen=True)
class MerchantProfile:
    merchant_id: str
    name: str
    category: str
    daily_txns: int
    avg_ticket_paise: int
    ticket_spread: float
    method_bias: dict[str, float] = field(default_factory=dict)


MERCHANTS = [
    MerchantProfile(
        "acc_KIRANA01", "Daily Basket", "grocery", 420, 68_000, 0.55, {"upi": 1.25, "card": 0.65}
    ),
    MerchantProfile(
        "acc_FASHION01", "Thread & Co", "fashion", 180, 249_000, 0.80, {"card": 1.30, "upi": 0.90}
    ),
    MerchantProfile(
        "acc_EDTECH01",
        "LearnLoop",
        "education",
        60,
        1_499_000,
        0.45,
        {"card": 1.45, "netbanking": 1.60, "upi": 0.60},
    ),
    MerchantProfile(
        "acc_TRAVEL01",
        "Voyage Desk",
        "travel",
        45,
        3_200_000,
        1.10,
        {"card": 1.55, "netbanking": 1.40, "upi": 0.55},
    ),
]


def _norm(d: dict[str, float]) -> dict[str, float]:
    total = sum(d.values())
    return {k: v / total for k, v in d.items()}


def _stable_id(prefix: str, *parts: object) -> str:
    """Deterministic Razorpay-shaped id, so reruns with the same seed match."""
    digest = hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()
    return f"{prefix}{digest[:14].upper()}"


@dataclass
class GeneratorConfig:
    seed: int = A.DEFAULT_SEED
    days: int = A.DEFAULT_DAYS
    target_cases: int = A.TARGET_CASES
    end_date: datetime = field(default_factory=lambda: datetime(2026, 8, 22, tzinfo=UTC))
    # World parameterisation. The evaluation re-runs under several of these to
    # test whether conclusions survive a different world, not just a worse model.
    world: str = "base"
    recoverability_scale: float = 1.0
    fatigue_lambda: float = A.CONTACT_FATIGUE_LAMBDA.value
    # Scales merchant transaction volume to reach `target_cases`. The merchant
    # profiles below describe realistic *shapes* (ticket size, method mix,
    # category seasonality); this knob sets the scale so the dataset is large
    # enough to time-split and still leave >=2,000 cases for calibration.
    volume_scale: float = 1.6


WORLDS: dict[str, dict[str, float]] = {
    "base": {"recoverability_scale": 1.00, "fatigue_lambda": 0.55},
    "pessimistic": {"recoverability_scale": 0.65, "fatigue_lambda": 0.40},
    "optimistic": {"recoverability_scale": 1.35, "fatigue_lambda": 0.75},
}


def build_customers(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Customer pool with latent behavioural traits.

    `latent_reliability` is the customer-level propensity to complete a payment
    once they have started one. It is never exposed to the model; what the model
    gets is the *observed* history it produces, which is exactly the relationship
    a real system faces.
    """
    segments = rng.choice(
        ["new", "occasional", "regular", "loyal"], size=n, p=[0.28, 0.34, 0.26, 0.12]
    )
    seg_reliability = {"new": 0.55, "occasional": 0.66, "regular": 0.78, "loyal": 0.88}
    base = np.array([seg_reliability[s] for s in segments])
    reliability = np.clip(rng.normal(base, 0.12), 0.05, 0.99)

    return pd.DataFrame(
        {
            "customer_id": [_stable_id("cust_", "c", i) for i in range(n)],
            "segment": segments,
            "latent_reliability": reliability,
            # Willingness to act on a message, independent of ability to pay.
            "latent_responsiveness": np.clip(rng.beta(2.2, 3.0, n), 0.02, 0.95),
            "lifetime_value_paise": (rng.lognormal(12.4, 0.9, n)).astype(np.int64),
            "preferred_issuer": rng.choice(ISSUERS, size=n, p=ISSUER_WEIGHTS),
        }
    )


def _pick_method(rng, merchant: MerchantProfile) -> str:
    weights = {m: p.value * merchant.method_bias.get(m, 1.0) for m, p in A.METHOD_MIX.items()}
    weights = _norm(weights)
    return rng.choice(list(weights), p=list(weights.values()))


def _pick_cause(rng, method: str) -> str:
    mix = A.CAUSE_MIX_BY_METHOD[method]
    return rng.choice(list(mix), p=list(mix.values()))


def _success_probability(hour_ist: int, reliability: float, outage: float) -> float:
    """Probability this attempt succeeds.

    Combines the merchant-level base rate, the customer's own reliability, a
    peak-hour penalty, and any issuer outage active at that moment.
    """
    p = A.BLENDED_SUCCESS_RATE.value
    p *= 0.75 + 0.25 * (reliability / 0.75)  # customer effect, bounded
    if hour_ist in A.PEAK_HOURS_IST:
        p *= A.PEAK_HOUR_SUCCESS_MULTIPLIER.value
    p *= 1.0 - outage
    return float(np.clip(p, 0.35, 0.995))


def generate(config: GeneratorConfig | None = None) -> dict[str, pd.DataFrame]:
    """Generate the dataset. Deterministic for a given seed and world.

    Runs in two phases, and the split matters. Phase 1 lays out every payment
    attempt across all merchants and days; phase 2 sorts them chronologically and
    walks them in time order, accumulating each customer's history as it goes.

    The two phases exist because the obvious single-loop version is wrong. Looping
    day -> merchant -> transaction accumulates history in *processing* order, so a
    customer who pays at two merchants on the same day can have "prior" history
    containing events that happen later in wall-clock time. 41% of customers here
    are active at more than one merchant, so that is not an edge case -- it is
    look-ahead contamination in exactly the features the model trains on.
    """
    config = config or GeneratorConfig()
    if config.world in WORLDS:
        w = WORLDS[config.world]
        config.recoverability_scale = w["recoverability_scale"]
        config.fatigue_lambda = w["fatigue_lambda"]

    rng = np.random.default_rng(config.seed)

    n_customers = max(2_000, int(config.target_cases * config.volume_scale // 2))
    customers = build_customers(rng, n_customers)

    reliability = customers["latent_reliability"].to_numpy()
    responsiveness = customers["latent_responsiveness"].to_numpy()
    segments = customers["segment"].to_numpy()
    ltv = customers["lifetime_value_paise"].to_numpy()
    pref_issuer = customers["preferred_issuer"].to_numpy()

    # Issuer outage windows: correlated, short, concentrated on one issuer. This
    # is what a real "UPI failures tripled in two hours" incident looks like.
    outages: list[tuple[str, datetime, datetime, float]] = []
    for _ in range(rng.integers(6, 12)):
        issuer = str(rng.choice(ISSUERS, p=ISSUER_WEIGHTS))
        start_day = int(rng.integers(0, config.days))
        start_hour = int(rng.integers(0, 24))
        start_ts = config.end_date - timedelta(days=config.days - start_day)
        start_ts = start_ts.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        outages.append(
            (
                issuer,
                start_ts,
                start_ts + timedelta(hours=int(rng.integers(1, 6))),
                float(rng.uniform(0.25, 0.70)),
            )
        )

    def outage_for(issuer: str, ts: datetime) -> float:
        for iss, s_ts, e_ts, severity in outages:
            if iss == issuer and s_ts <= ts < e_ts:
                return severity
        return 0.0

    start_date = config.end_date - timedelta(days=config.days)

    # --- Phase 1: lay out every attempt ------------------------------------
    attempts: list[tuple[datetime, int, int, str, int]] = []
    for day in range(config.days):
        day_start = start_date + timedelta(days=day)
        # Python weekday(): Monday=0 ... Sunday=6. The weekend is Saturday and
        # Sunday, i.e. 5 and 6. (An earlier version rotated this by one and
        # quietly applied the weekend uplift to Friday and Saturday.)
        weekday = day_start.weekday()
        is_weekend = weekday in (5, 6)

        for m_idx, merchant in enumerate(MERCHANTS):
            is_retail = merchant.category in ("grocery", "fashion")
            season = 1.18 if is_weekend and is_retail else 1.0
            n_txns = int(rng.poisson(merchant.daily_txns * season * config.volume_scale))

            for _ in range(n_txns):
                ci = int(rng.integers(0, n_customers))
                hour = int(np.clip(rng.normal(15, 4.5), 0, 23))
                ts = day_start.replace(hour=hour, minute=int(rng.integers(0, 60)))
                method = _pick_method(rng, merchant)
                amount = int(
                    np.clip(
                        rng.lognormal(np.log(merchant.avg_ticket_paise), merchant.ticket_spread),
                        1_000,
                        50_000_000,
                    )
                )
                attempts.append((ts, m_idx, ci, method, amount))

    # --- Phase 2: walk in time order ---------------------------------------
    # Ties broken deterministically so a given seed always yields one ordering.
    attempts.sort(key=lambda a: (a[0], a[1], a[2]))

    n_attempts = np.zeros(n_customers, dtype=np.int32)
    n_successes = np.zeros(n_customers, dtype=np.int32)
    n_failures = np.zeros(n_customers, dtype=np.int32)
    last_failure_at: dict[int, datetime] = {}

    rows: list[dict] = []
    for ts, m_idx, ci, method, amount in attempts:
        merchant = MERCHANTS[m_idx]
        issuer = str(pref_issuer[ci])
        hour = ts.hour

        # Snapshot history *before* this attempt. Every one of these is strictly
        # what was knowable at decision time.
        prior_attempts = int(n_attempts[ci])
        prior_failures = int(n_failures[ci])
        prior_success_rate = float(n_successes[ci]) / prior_attempts if prior_attempts else np.nan
        prior_failure = last_failure_at.get(ci)
        hours_since = (ts - prior_failure).total_seconds() / 3600.0 if prior_failure else np.nan

        outage = outage_for(issuer, ts)
        p_success = _success_probability(hour, reliability[ci], outage)

        n_attempts[ci] += 1
        if rng.random() < p_success:
            n_successes[ci] += 1
            continue

        # --- failed: this becomes a case -----------------------------------
        # An active outage forces the cause toward bank downtime, producing a
        # correlated spike rather than uniform noise.
        if outage > 0 and rng.random() < outage:
            cause = "transient_bank_downtime"
        else:
            cause = _pick_cause(rng, method)

        reason, source, step = CAUSE_TO_RAZORPAY[cause]
        rec = A.RECOVERABILITY[cause]
        scale = config.recoverability_scale

        # Latent truth. Customer reliability modulates whether they would have
        # paid anyway; the cause dominates the ceiling.
        p_self = float(
            np.clip(rec.p_self_recover * scale * (0.6 + 0.8 * reliability[ci]), 0.0, 0.95)
        )
        p_retry = float(np.clip(rec.p_retry * scale, 0.0, 0.97))
        p_nudge = float(np.clip(rec.p_nudge * scale * (0.5 + responsiveness[ci]), 0.0, 0.95))

        n_failures[ci] += 1
        last_failure_at[ci] = ts

        payment_id = _stable_id("pay_", merchant.merchant_id, ci, ts, amount)
        rows.append(
            {
                # ---- identity ----
                "case_id": _stable_id("case_", payment_id),
                "payment_id": payment_id,
                "order_id": _stable_id("order_", payment_id, "o"),
                "merchant_id": merchant.merchant_id,
                "merchant_category": merchant.category,
                "customer_id": customers.at[ci, "customer_id"],
                "failed_at": ts,
                # ---- observable at decision time ----
                "amount_paise": amount,
                "method": method,
                "issuer": issuer,
                "error_reason": reason,
                "error_source": source,
                "error_step": step,
                "hour_ist": hour,
                "weekday": ts.weekday(),
                "is_weekend": ts.weekday() in (5, 6),
                "is_peak_hour": hour in A.PEAK_HOURS_IST,
                "customer_segment": str(segments[ci]),
                "customer_prior_attempts": prior_attempts,
                "customer_prior_failures": prior_failures,
                # NaN, not a sentinel: a model reading -1 as a duration treats
                # "never failed before" as "failed an hour in the future".
                "customer_observed_success_rate": prior_success_rate,
                "has_prior_history": prior_attempts > 0,
                "hours_since_last_failure": hours_since,
                "has_prior_failure": prior_failure is not None,
                "customer_ltv_paise": int(ltv[ci]),
                # ---- ground truth, never exposed to the model ----
                "latent_cause": cause,
                "latent_p_self_recover": p_self,
                "latent_p_retry": p_retry,
                "latent_p_nudge": p_nudge,
                "latent_best_delay_h": rec.best_delay_h,
                "latent_issuer_outage": outage,
                "latent_customer_responsiveness": float(responsiveness[ci]),
            }
        )

    cases = pd.DataFrame(rows).reset_index(drop=True)

    # Time-based split. Never random: a random split lets a customer's later
    # behaviour inform a prediction about their earlier failure, which inflates
    # every metric and would not survive production.
    #
    # Boundaries are placed on a *change of timestamp*, so no single instant can
    # appear on both sides of a split.
    n = len(cases)
    cases["split"] = "train"
    train_end = _split_boundary(cases, int(n * (1 - A.CALIBRATION_HOLDOUT - A.TEST_HOLDOUT)))
    calib_end = _split_boundary(cases, int(n * (1 - A.TEST_HOLDOUT)))
    cases.loc[train_end : calib_end - 1, "split"] = "calibration"
    cases.loc[calib_end:, "split"] = "test"

    return {"cases": cases, "customers": customers}


def _split_boundary(cases: pd.DataFrame, idx: int) -> int:
    """Move `idx` forward until the timestamp changes.

    Without this, a batch of cases sharing one timestamp can straddle a split
    boundary, which is a small but real leak across an ostensibly clean
    chronological division.
    """
    if idx <= 0 or idx >= len(cases):
        return idx
    ts = cases.at[idx, "failed_at"]
    while idx < len(cases) and cases.at[idx, "failed_at"] == ts:
        idx += 1
    return idx
