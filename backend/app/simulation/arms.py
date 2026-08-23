"""Run every arm against a dataset slice and report comparable metrics.

    python -m app.simulation.arms --world base --split test

Prints its own configuration before any numbers. Three separate times in this
project a check has been green while exercising something other than what I
believed -- the wrong database, a cause with no probability mass, an episode
terminating early. A harness that does not state what it just ran is a harness
you are trusting rather than checking.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from app.simulation.episode import Policy, run_episode
from app.simulation.generator import WORLDS
from app.simulation.policies import BASELINES, ORACLE

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "generated"


@dataclass
class ArmResult:
    arm: str
    n_cases: int
    at_risk_paise: int
    gross_paise: int
    incremental_paise: int
    cost_paise: int
    contacts: int
    unnecessary_contacts: int
    recovered_cases: int
    caused_cases: int

    #: Per-case net incremental paise, in dataset order. Retained because common
    #: random numbers make arms directly comparable case by case, which turns a
    #: two-sample comparison into a paired one.
    per_case_net: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def net_paise(self) -> int:
        return self.incremental_paise - self.cost_paise

    @property
    def incremental_pct(self) -> float:
        return self.incremental_paise / self.at_risk_paise if self.at_risk_paise else 0.0

    @property
    def rupees_per_contact(self) -> float:
        """Incremental rupees recovered per customer contact spent.

        The efficiency metric, and the one that separates the arms most sharply.
        Gross recovery can always be increased by contacting more people; this
        cannot, because the denominator grows with the numerator.
        """
        return (self.incremental_paise / 100) / self.contacts if self.contacts else float("inf")

    @property
    def unnecessary_rate(self) -> float:
        return self.unnecessary_contacts / self.contacts if self.contacts else 0.0


def evaluate(cases: pd.DataFrame, policy: Policy, fatigue_lambda: float) -> ArmResult:
    eps = [run_episode(r, policy, fatigue_lambda=fatigue_lambda) for _, r in cases.iterrows()]
    return ArmResult(
        arm=policy.name,
        n_cases=len(cases),
        at_risk_paise=int(cases.amount_paise.sum()),
        gross_paise=sum(e.amount_recovered_paise for e in eps),
        incremental_paise=sum(e.incremental_paise for e in eps),
        cost_paise=sum(e.cost_paise for e in eps),
        contacts=sum(e.contacts_used for e in eps),
        unnecessary_contacts=sum(e.unnecessary_contacts for e in eps),
        recovered_cases=sum(e.recovered for e in eps),
        caused_cases=sum(e.caused_by_us for e in eps),
        per_case_net=np.array([e.net_incremental_paise for e in eps], dtype=np.int64),
    )


def paired_bootstrap(
    a: ArmResult,
    b: ArmResult,
    *,
    n_boot: int = 5_000,
    seed: int = 0,
    material_rs: float = 30.0,
) -> dict:
    """Paired bootstrap on the per-case difference between two arms.

    Valid here specifically *because* of common random numbers. Both arms faced
    the same customers with the same latent willingness and the same draws, so
    the per-case difference isolates the policy. Treating the arms as two
    independent samples would throw that away and give a far wider interval for
    no reason.

    Reports the mean difference in net incremental rupees per case, a 95%
    percentile interval, and the share of cases where each arm did better.
    """
    diff = (a.per_case_net - b.per_case_net) / 100.0  # rupees
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    # Raw win/loss counts are misleading here. A contact-heavy arm shows as
    # "worse" on any case both arms recovered, purely because it paid Rs 0.25 for
    # a message. Counting that as a loss alongside a genuine Rs 5,000 miss makes
    # the arm look far worse than it is, so differences below one or two
    # contacts' cost are reported as immaterial rather than as losses.
    material = np.abs(diff) >= material_rs
    return {
        "mean_diff_rs": float(diff.mean()),
        "ci_low_rs": float(lo),
        "ci_high_rs": float(hi),
        "significant": bool(lo > 0 or hi < 0),
        "a_better_pct": float((diff >= material_rs).mean()),
        "b_better_pct": float((diff <= -material_rs).mean()),
        "immaterial_pct": float((~material).mean()),
        "material_rs": material_rs,
    }


def load(world: str, seed: int) -> pd.DataFrame:
    name = f"cases.seed{seed}.parquet" if world == "base" else f"cases.seed{seed}.{world}.parquet"
    path = DATA_DIR / name
    if not path.exists():
        # A plain exception, not SystemExit. `load` is called from tests and
        # from other modules; SystemExit escapes normal exception handling and
        # tears down the caller, which is hostile behaviour for a library.
        raise FileNotFoundError(f"missing {path}. Run: make data-worlds")
    return pd.read_parquet(path)


def report(results: list[ArmResult]) -> str:
    hdr = (
        f"{'arm':26}{'incremental':>14}{'cost':>9}{'NET':>14}"
        f"{'incr%':>8}{'contacts':>10}{'unnec%':>8}{'Rs/contact':>12}"
    )
    lines = [hdr, "-" * len(hdr)]
    for r in results:
        rpc = "—" if r.contacts == 0 else f"{r.rupees_per_contact:,.0f}"
        lines.append(
            f"{r.arm:26}{r.incremental_paise/100:>14,.0f}{r.cost_paise/100:>9,.0f}"
            f"{r.net_paise/100:>14,.0f}{r.incremental_pct:>7.1%}{r.contacts:>10,}"
            f"{r.unnecessary_rate:>7.0%}{rpc:>12}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare recovery arms.")
    ap.add_argument("--world", choices=sorted(WORLDS), default="base")
    ap.add_argument("--split", choices=["train", "calibration", "test", "all"], default="test")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cases = load(args.world, args.seed)
    if args.split != "all":
        cases = cases[cases.split == args.split]

    lam = WORLDS[args.world]["fatigue_lambda"]

    print("Recoup arm comparison")
    print(f"  world        : {args.world}  {WORLDS[args.world]}")
    print(f"  split        : {args.split}")
    print(f"  seed         : {args.seed}")
    print(f"  cases        : {len(cases):,}")
    print(f"  at risk      : Rs {cases.amount_paise.sum()/100:,.0f}")
    print(f"  fatigue λ    : {lam}")
    print("  SIMULATION BENCHMARK -- not production Razorpay data")
    print()

    results = [evaluate(cases, pol, lam) for pol in BASELINES]
    ceiling = evaluate(cases, ORACLE, lam)
    print(report(results))

    floor = next(r for r in results if r.arm == "B0_do_nothing")
    print(f"\n  do-nothing floor: Rs {floor.gross_paise/100:,.0f} recovered with zero action")
    print("  (every other arm is measured as incremental ON TOP of this)")

    print(f"\n  ORACLE ceiling  : Rs {ceiling.incremental_paise/100:,.0f} incremental")
    print("  (reads ground truth; an upper bound under the same 4-action budget,")
    print("   never ranked against the arms -- it says how much was left to take)")
    for r in results:
        if r.arm == "B0_do_nothing":
            continue
        pct = 100 * r.incremental_paise / max(ceiling.incremental_paise, 1)
        print(f"      {r.arm:26} {pct:>5.1f}% of ceiling")

    print("\n  Paired comparison vs Razorpay T+3 (common random numbers -> paired):")
    t3 = next(r for r in results if r.arm == "B1_razorpay_t3")
    for r in results:
        if r.arm in ("B0_do_nothing", "B1_razorpay_t3"):
            continue
        st = paired_bootstrap(r, t3)
        mark = "significant" if st["significant"] else "NOT significant"
        print(
            f"      {r.arm:26} +Rs {st['mean_diff_rs']:>8,.0f}/case  "
            f"95% CI [{st['ci_low_rs']:,.0f}, {st['ci_high_rs']:,.0f}]  {mark}"
        )
        print(
            f"      {'':26}   materially better on {st['a_better_pct']:.0%}, "
            f"worse on {st['b_better_pct']:.0%}, "
            f"within Rs {st['material_rs']:.0f} on {st['immaterial_pct']:.0%}"
        )


if __name__ == "__main__":
    main()
