"""Regenerate data/ASSUMPTIONS.md from the assumptions module.

Generated rather than hand-written so the document cannot drift from the code it
describes. A stale assumptions file is worse than none: it makes the evaluation
look documented while describing parameters that are no longer in use.

    python -m app.simulation.docgen
"""

from __future__ import annotations

from pathlib import Path

from app import taxonomy
from app.simulation import assumptions as A
from app.simulation.generator import MERCHANTS, WORLDS

OUT = Path(__file__).resolve().parents[3] / "data" / "ASSUMPTIONS.md"

INTRO = """# Declared assumptions

Every parameter behind the synthetic dataset, with an honest label for where it
came from. **Generated from `backend/app/simulation/assumptions.py` — do not edit
by hand.** Regenerate with `make assumptions`.

| Basis | Meaning |
|---|---|
| `sourced` | Taken from published figures (Razorpay documentation, NPCI/UPI reporting). |
| `anchored` | Derived from a sourced figure plus a stated inference. |
| `estimate` | My judgement. Not sourced. Sensitivity-tested rather than trusted. |

## Why so many estimates

Nobody publishes conditional recovery probabilities by failure cause, because
that is exactly the proprietary knowledge a payments company accumulates from
production traffic. Pretending otherwise would be the dishonest move.

The response is not to dress estimates up as facts. It is to label them, hold
them fixed and reproducible, and re-run the entire evaluation under different
parameterisations to see which conclusions survive. For the recoverability
parameters in particular, the load-bearing claim is their **relative ordering** —
expired collect requests recover far more readily than hard declines — not their
absolute values. No plausible parameterisation reverses that ordering.

## What this dataset can and cannot support

**It can** test whether the allocator makes better decisions than the baselines,
given a probability oracle of a stated quality.

**It cannot** demonstrate that the recovery model is accurate. The generating
process is known, so a model trained and scored here measures gradient descent,
not payments. That circularity is addressed by degrading the oracle and varying
the world — not by pretending the simulator validates the model.
"""


def _table(rows: list[tuple[str, str, str]]) -> str:
    out = ["| Parameter | Value | Basis | Note |", "|---|---|---|---|"]
    out.extend(f"| `{n}` | {v} | `{b}` | {note} |" for n, v, b, note in rows)
    return "\n".join(out)


def build() -> str:
    parts = [INTRO, "\n## Rates and volumes\n"]

    rows = []
    for name in [
        "BLENDED_SUCCESS_RATE",
        "TECHNICAL_DECLINE_RATE",
        "CUSTOMER_SIDE_SHARE",
        "PEAK_HOUR_SUCCESS_MULTIPLIER",
        "CONTACT_FATIGUE_LAMBDA",
        "BRAND_ABANDONMENT_AFTER_FAILURE",
        "PUBLISHED_RECOVERY_CEILING",
    ]:
        p = getattr(A, name)
        rows.append((name, f"{p.value:g}", p.basis.value, p.note))
    parts.append(_table(rows))

    parts.append(f"\nPeak hours (IST): {', '.join(str(h) for h in A.PEAK_HOURS_IST)}\n")

    parts.append("\n## Payment method mix\n")
    parts.append(
        _table(
            [
                (f"METHOD_MIX[{m}]", f"{p.value:.0%}", p.basis.value, p.note)
                for m, p in A.METHOD_MIX.items()
            ]
        )
    )

    parts.append("\n## Cause mix by method\n")
    parts.append(
        "Anchored on two sourced constraints — customer-side causes carry ~55% of "
        "the mass, and bank/gateway technical declines are ~1% — with the split "
        "*within* those bands estimated. The profile genuinely differs by method: "
        "UPI fails at collect and cancellation, cards fail at authentication and "
        "decline.\n"
    )
    methods = list(A.CAUSE_MIX_BY_METHOD)
    all_causes = sorted({c for mix in A.CAUSE_MIX_BY_METHOD.values() for c in mix})
    header = "| Cause | " + " | ".join(methods) + " |"
    parts.append(header)
    parts.append("|---" * (len(methods) + 1) + "|")
    for cause in all_causes:
        cells = [
            f"{A.CAUSE_MIX_BY_METHOD[m][cause]:.0%}" if A.CAUSE_MIX_BY_METHOD[m].get(cause) else "—"
            for m in methods
        ]
        parts.append(f"| `{cause}` | " + " | ".join(cells) + " |")

    parts.append("\n## Latent recoverability (counterfactual ground truth)\n")
    parts.append(
        "These are the parameters the agent never observes. `p_self_recover` is "
        "the denominator that makes *unnecessary contact* measurable: contacting "
        "someone who would have paid regardless is a cost with no matching "
        "benefit, and it is invisible to any metric that only counts recoveries.\n"
    )
    parts.append("| Cause | p_self_recover | p_retry | p_nudge | best delay (h) | Note |")
    parts.append("|---|---|---|---|---|---|")
    for key, r in A.RECOVERABILITY.items():
        parts.append(
            f"| `{key}` | {r.p_self_recover:.2f} | {r.p_retry:.2f} | "
            f"{r.p_nudge:.2f} | {r.best_delay_h:g} | {r.note} |"
        )

    parts.append("\n## Recovery semantics per cause\n")
    parts.append(
        "Derived from Razorpay's published error tables. Three axes drive every "
        "downstream action.\n"
    )
    parts.append("| Cause | Who can fix | Retry policy | Contact OK |")
    parts.append("|---|---|---|---|")
    for c in taxonomy.all_causes():
        parts.append(
            f"| `{c.key}` | {c.who_can_fix.value} | {c.retry_policy.value} | "
            f"{'yes' if c.contact_ok else '**no**'} |"
        )

    parts.append("\n## Intervention costs\n")
    parts.append(
        _table(
            [
                (f"ACTION_COST_PAISE[{a}]", f"₹{p.value / 100:.2f}", p.basis.value, p.note)
                for a, p in A.ACTION_COST_PAISE.items()
            ]
        )
    )

    parts.append("\n## Merchant profiles\n")
    parts.append("| Merchant | Category | Txns/day | Avg ticket |")
    parts.append("|---|---|---|---|")
    for m in MERCHANTS:
        parts.append(
            f"| {m.name} | {m.category} | {m.daily_txns} | ₹{m.avg_ticket_paise / 100:,.0f} |"
        )

    parts.append("\n## World parameterisations\n")
    parts.append(
        "The world sweep re-runs the full evaluation under each of these. Worlds "
        "hold the *same failures* and vary only how recoverable they are, so any "
        "difference in results is attributable to the world and nothing else.\n"
    )
    parts.append("| World | Recoverability scale | Contact fatigue λ |")
    parts.append("|---|---|---|")
    for w, p in WORLDS.items():
        parts.append(f"| `{w}` | {p['recoverability_scale']:.2f} | {p['fatigue_lambda']:.2f} |")

    parts.append("\n## Dataset shape\n")
    parts.append(
        f"- Default seed: `{A.DEFAULT_SEED}`\n"
        f"- Window: {A.DEFAULT_DAYS} days\n"
        f"- Target cases: {A.TARGET_CASES:,} (minimum 8,000)\n"
        f"- Calibration holdout: {A.CALIBRATION_HOLDOUT:.0%} (minimum 2,000 cases)\n"
        f"- Test holdout: {A.TEST_HOLDOUT:.0%}, taken as the final slice of the "
        f"timeline — never a random sample\n"
    )
    return "\n".join(parts) + "\n"


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build())
    print(f"wrote {OUT} ({len(build().splitlines())} lines)")
