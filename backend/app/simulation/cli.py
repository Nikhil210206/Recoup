"""Dataset generation CLI.

    python -m app.simulation.cli --seed 42 --world base

Prints its own configuration before writing anything. That is deliberate: on
day 0 a green test suite turned out to be running against the wrong database,
and the lesson generalises -- a harness that does not state what it just ran is
a harness you are trusting rather than checking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.simulation import assumptions as A
from app.simulation.generator import WORLDS, GeneratorConfig, generate

OUT_DIR = Path(__file__).resolve().parents[3] / "data" / "generated"


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the Recoup synthetic dataset.")
    ap.add_argument("--seed", type=int, default=A.DEFAULT_SEED)
    ap.add_argument("--world", choices=sorted(WORLDS), default="base")
    ap.add_argument("--days", type=int, default=A.DEFAULT_DAYS)
    ap.add_argument("--target-cases", type=int, default=A.TARGET_CASES)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    cfg = GeneratorConfig(
        seed=args.seed, world=args.world, days=args.days, target_cases=args.target_cases
    )

    print("Recoup dataset generator")
    print(f"  seed        : {cfg.seed}")
    print(f"  world       : {cfg.world}  {WORLDS[cfg.world]}")
    print(f"  days        : {cfg.days}")
    print(f"  target cases: {cfg.target_cases:,}")
    print()

    data = generate(cfg)
    cases, customers = data["cases"], data["customers"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.world == "base" else f".{args.world}"
    cases_path = args.out_dir / f"cases.seed{cfg.seed}{suffix}.parquet"
    cust_path = args.out_dir / f"customers.seed{cfg.seed}{suffix}.parquet"
    cases.to_parquet(cases_path, index=False)
    customers.to_parquet(cust_path, index=False)

    manifest = {
        "seed": cfg.seed,
        "world": cfg.world,
        "world_params": WORLDS[cfg.world],
        "days": cfg.days,
        "volume_scale": cfg.volume_scale,
        "n_cases": int(len(cases)),
        "n_customers": int(len(customers)),
        "date_range": [str(cases.failed_at.min()), str(cases.failed_at.max())],
        "splits": {k: int(v) for k, v in cases.split.value_counts().items()},
        "cause_mix": {
            k: float(v) for k, v in cases.latent_cause.value_counts(normalize=True).items()
        },
        "at_risk_paise": int(cases.amount_paise.sum()),
    }
    (args.out_dir / f"manifest.seed{cfg.seed}{suffix}.json").write_text(
        json.dumps(manifest, indent=2)
    )

    print(f"  cases       : {len(cases):,}  -> {cases_path.name}")
    print(f"  customers   : {len(customers):,}  -> {cust_path.name}")
    print(f"  at risk     : Rs {cases.amount_paise.sum() / 100:,.0f}")
    print(f"  splits      : {dict(cases.split.value_counts())}")


if __name__ == "__main__":
    main()
