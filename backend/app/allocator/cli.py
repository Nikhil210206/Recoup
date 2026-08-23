"""Allocator bake-off.

    python -m app.allocator.cli --world base --budget 600

Every arm gets the same contact budget, which is the only comparison in which
the word "allocator" means anything. Prints its configuration first, and the
component ablation last -- including the components that contribute nothing.
"""

from __future__ import annotations

import argparse

from app.allocator.bake_off import BudgetedResult, ranked, render, run_bake_off
from app.allocator.budget import BudgetPolicy
from app.allocator.estimator import CauseRate
from app.allocator.policy import Allocator, summarise
from app.allocator.siloed import run_siloed
from app.model.train import build_frames
from app.simulation.arms import load
from app.simulation.generator import WORLDS
from app.simulation.policies import CauseAware, ContactEverything, RazorpayT3


def main() -> None:
    ap = argparse.ArgumentParser(description="Allocator bake-off at equal budget.")
    ap.add_argument("--world", choices=sorted(WORLDS), default="base")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget", type=int, default=600)
    args = ap.parse_args()

    cases = load(args.world, args.seed)
    train_frame = build_frames(args.world, args.seed)["train"]
    test = cases[cases.split == "test"].reset_index(drop=True)
    lam = WORLDS[args.world]["fatigue_lambda"]

    print("Recoup allocator bake-off")
    print(f"  world   : {args.world}  {WORLDS[args.world]}")
    print(f"  seed    : {args.seed}")
    print(f"  cases   : {len(test):,}")
    print(f"  at risk : Rs {test.amount_paise.sum() / 100:,.0f}")
    print(f"  budget  : {args.budget:,} contacts (identical for every arm)")
    print("  Retries are not rationed: they touch the gateway, not the customer.")
    print("  SIMULATION BENCHMARK -- not production Razorpay data")
    print()

    estimator = CauseRate().fit(train_frame)
    allocator = Allocator(
        estimator=estimator, budget_policy=BudgetPolicy(max_contacts_per_customer=2)
    )
    baselines = {
        "B1_razorpay_t3": RazorpayT3(),
        "B2_contact_everything": ContactEverything(),
        "B4_cause_aware (arrival order)": CauseAware(),
    }
    results = run_bake_off(test, allocator, baselines, args.budget, lam)

    # The two controls that make the comparison honest.
    #
    # SILOED is the architecture that exists: one agent per loss channel, each
    # with its own queue and its own slice of the budget, none of them aware of
    # the others.
    #
    # POOLED+RANKED is an idealised single agent handed an already-sorted queue.
    # It is stronger than anything shipping, and it exists to show how much of
    # the allocator's advantage is really just an ORDER BY.
    siloed, silo_stats = run_siloed(test, CauseAware, args.budget, fatigue_lambda=lam)
    pooled = [
        r.result
        for r in run_bake_off(
            ranked(test),
            Allocator(
                estimator=estimator,
                budget_policy=BudgetPolicy(max_contacts_per_customer=99),
            ),
            {"B5_pooled + value ranking": CauseAware()},
            args.budget,
            lam,
        )
        if r.result.arm.startswith("B5")
    ][0]
    results.append(BudgetedResult(siloed, args.budget, siloed.contacts))
    results.append(BudgetedResult(pooled, args.budget, pooled.contacts))

    print(render(results, args.budget))

    decisions, ledger = Allocator(
        estimator=estimator,
        budget_policy=BudgetPolicy(
            max_contacts_per_customer=2, max_total_contacts=args.budget
        ),
    ).plan(test)

    print("\n  allocation")
    for key, value in summarise(decisions).items():
        print(f"    {key:22} {value}")
    print("\n  budget ledger")
    for key, value in ledger.summary().items():
        print(f"    {key:22} {value}")

    print("\n  what the collision guard prevents")
    for key, value in silo_stats["collisions"].items():
        print(f"    {key:34} {value}")

    alloc_result = next(r.result for r in results if r.result.arm == "RECOUP_allocator")
    arrival = next(
        r.result for r in results if r.result.arm.startswith("B4_cause_aware")
    )
    print("\n  honest decomposition")
    rank_lift = (
        pooled.incremental_paise - arrival.incremental_paise
    ) / arrival.incremental_paise
    print(f"    value ordering  (arrival -> ranked) : {rank_lift:+.1%}")
    vs_silo = (
        alloc_result.incremental_paise - siloed.incremental_paise
    ) / siloed.incremental_paise
    print(f"    allocator over siloed agents        : {vs_silo:+.1%}")
    vs_pooled = (
        alloc_result.incremental_paise - pooled.incremental_paise
    ) / pooled.incremental_paise
    print(f"    allocator over pooled+ranked        : {vs_pooled:+.1%}")
    print()
    print("    Recoup's revenue comes from action selection and value ordering.")
    print("    The budget, the contact cap and the audit trail are a GOVERNANCE")
    print("    layer: they cost about 1% and buy a hard ceiling on how much of one")
    print("    customer's patience a merchant may spend across every channel at")
    print("    once -- which no per-channel agent can enforce, because none of")
    print("    them can see the others. Reported as governance, not as revenue.")


if __name__ == "__main__":
    main()
