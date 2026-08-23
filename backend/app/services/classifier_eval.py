"""Measured evaluation of the LLM tail, and of model choice.

    python -m app.services.classifier_eval --models claude-opus-5,claude-haiku-4-5

The tail has an obvious measurement problem: by definition it handles error codes
we have never seen, so there are no labels for it. Waiting for Razorpay to ship
new codes is not an evaluation strategy.

So: hold out reason codes that *are* in the deterministic table, hide them from
the lookup, and force them down the LLM path. Their true cause is known, which
gives real ground truth for exactly the operation the tail performs -- reading a
code and description it cannot look up, and mapping it onto the canonical set.

This is a proxy. Held-out codes are ones I already chose canonical causes for, so
they are cleaner than a genuinely novel code would be, and the accuracy here is
an optimistic bound on tail accuracy. Stated, not hidden.

The model comparison exists because "use the best model everywhere" is not
engineering judgement, it is the absence of it. If a cheaper model classifies the
tail as accurately, the expensive one is spending money for nothing; if it does
not, the cost is justified and can be defended with a number.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from app import taxonomy
from app.config import get_settings
from app.services.classifier import CONFIDENCE_THRESHOLD, CauseClassifier

#: Held out from the deterministic table. Chosen to span the decision space:
#: a bank-side transient, a customer funds issue, an instrument problem, a hard
#: decline, a risk block, and the merchant-configuration case where every
#: customer-facing action is wrong.
HELD_OUT: dict[str, str] = {
    "bank_technical_error": "The payment failed due to a downtime on the UPI provider.",
    "insufficient_funds": (
        "The payment did not go through because the customer's bank account did "
        "not have enough funds."
    ),
    "card_expired": "The payment could not be completed because the customer's card is expired.",
    "payment_risk_check_failed": (
        "The payment was declined by the bank as it was suspected to be fraudulent."
    ),
    "card_declined": (
        "The payment was declined by the customer's bank, resulting in the "
        "transaction being unsuccessful."
    ),
    "international_transaction_not_allowed": (
        "Your payment could not be completed as this business accepts domestic "
        "(Indian) card payments only. Try another payment method."
    ),
    "payment_collect_request_expired": (
        "The payment could not be completed as the customer exceeded the time "
        "limit for payment processing."
    ),
    "transaction_limit_exceeded": "The customer has reached their daily transaction limit.",
}

#: The error_source/step each held-out code really carries, so the model sees a
#: realistic payload rather than a bare string.
CONTEXT: dict[str, tuple[str, str, str]] = {
    "bank_technical_error": ("bank", "payment_authorization", "upi"),
    "insufficient_funds": ("customer", "payment_authorization", "upi"),
    "card_expired": ("customer", "payment_initiation", "card"),
    "payment_risk_check_failed": ("issuer", "payment_authorization", "card"),
    "card_declined": ("issuer", "payment_authorization", "card"),
    "international_transaction_not_allowed": ("business", "payment_initiation", "card"),
    "payment_collect_request_expired": ("customer", "payment_authorization", "upi"),
    "transaction_limit_exceeded": ("customer", "payment_authorization", "card"),
}


@dataclass
class ModelReport:
    model: str
    n: int = 0
    correct: int = 0
    wrong: int = 0
    abstained: int = 0
    below_threshold: int = 0
    total_latency_ms: int = 0
    errors: int = 0
    rows: list[tuple] = None

    def __post_init__(self):
        self.rows = self.rows or []

    @property
    def accuracy(self) -> float:
        """Accuracy over cases where the model committed to an answer."""
        answered = self.correct + self.wrong
        return self.correct / answered if answered else 0.0

    @property
    def coverage(self) -> float:
        return (self.n - self.abstained) / self.n if self.n else 0.0

    @property
    def dangerous(self) -> int:
        """Confidently wrong: above the acting threshold and incorrect.

        The only failure mode that actually costs money. A wrong answer below the
        threshold goes to a human; a wrong answer above it drives a real action.
        """
        return sum(1 for r in self.rows if not r[2] and r[3] >= CONFIDENCE_THRESHOLD)


def evaluate_model(spec: str) -> ModelReport:
    """`spec` is "provider:model", e.g. "ollama:llama3.2:3b" or
    "anthropic:claude-opus-5"."""
    provider, _, model = spec.partition(":")
    clf = CauseClassifier(
        model=model, provider=provider, blocked_reasons=frozenset(HELD_OUT)
    )
    rep = ModelReport(model=spec)

    for reason, description in HELD_OUT.items():
        source, step, method = CONTEXT[reason]
        truth = taxonomy.REASON_TO_CAUSE[reason]

        started = time.perf_counter()
        got = clf.classify(
            reason, source, step, description=description, payment_method=method
        )
        rep.total_latency_ms += int((time.perf_counter() - started) * 1000)
        rep.n += 1

        if got.method == "llm_error":
            rep.errors += 1
            continue
        if got.cause is None:
            rep.abstained += 1
            rep.rows.append((reason, truth, False, got.confidence, None))
            continue

        ok = got.cause == truth
        rep.correct += int(ok)
        rep.wrong += int(not ok)
        if got.confidence < CONFIDENCE_THRESHOLD:
            rep.below_threshold += 1
        rep.rows.append((reason, truth, ok, got.confidence, got.cause))

    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the LLM classification tail.")
    ap.add_argument("--models", default="ollama:llama3.2:3b,ollama:qwen2.5:7b")
    args = ap.parse_args()

    specs = [m.strip() for m in args.models.split(",") if m.strip()]
    if any(s.startswith("anthropic") for s in specs) and not get_settings().anthropic_api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set; drop the anthropic model from --models.")

    print("LLM tail evaluation -- held-out reason codes")
    print(f"  held out        : {len(HELD_OUT)} codes hidden from the lookup table")
    print(f"  acting threshold: {CONFIDENCE_THRESHOLD}")
    print("  NOTE: held-out codes are ones already in the table, so this is an")
    print("        optimistic bound on accuracy for genuinely novel codes.")
    print()

    reports = [evaluate_model(s) for s in specs]

    hdr = (
        f"{'model':26}{'answered':>10}{'correct':>9}{'wrong':>7}"
        f"{'abstain':>9}{'CONF-WRONG':>12}{'ms/call':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in reports:
        print(
            f"{r.model:26}{r.correct + r.wrong:>10}{r.correct:>9}{r.wrong:>7}"
            f"{r.abstained:>9}{r.dangerous:>12}{r.total_latency_ms // max(r.n, 1):>9}"
        )

    print("\nper-code detail:")
    for r in reports:
        print(f"\n  {r.model}")
        for reason, truth, ok, conf, got in r.rows:
            mark = "ok " if ok else "MISS"
            got_s = got or "abstained"
            print(f"    {mark} {reason:38} truth={truth:24} got={got_s:24} conf={conf:.2f}")

    if len(reports) > 1:
        best, cheap = reports[0], reports[-1]
        print(
            f"\nverdict: {best.model} accuracy {best.accuracy:.0%} vs "
            f"{cheap.model} {cheap.accuracy:.0%}"
        )
        print(
            "  Confidently-wrong count is the number that should decide this, not "
            "raw accuracy:\n"
            f"  {best.model}: {best.dangerous}   {cheap.model}: {cheap.dangerous}"
        )


if __name__ == "__main__":
    main()
