"""Populate the database so the console has something to show.

    make seed        (needs `make db` and `make data`)

Loads cases from the generated benchmark into Postgres and then runs the **real**
paths over them: the deterministic classifier, the allocator, the stopping rules.
Nothing here fabricates a decision for display -- the queue, the rule counts and
the ledger are all produced by the same code the live path uses.

Two things are worth being explicit about.

**These cases are simulated.** They come from `make data`, which generates them
from declared assumptions in `data/ASSUMPTIONS.md`. The one case in this database
that went through real Razorpay is left untouched, and the console labels it.

**Their timestamps are shifted forward.** The generator lays cases out across a
90-day window ending in the past, and a stopping rule closes anything older than
seven days. Seeding them at their original timestamps would produce a database
where every case is correctly, uselessly, `past_recovery_window`. So detection
times are re-based into the last few days. That is a property of the seed, not of
the system, which is why it is said here rather than left to be inferred.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select

from app.allocator.budget import BudgetPolicy
from app.allocator.estimator import CauseRate
from app.allocator.policy import Allocator
from app.db import SessionLocal, init_db
from app.models import Case, CaseStatus, LossChannel
from app.services import ledger, live_allocator
from app.services.ingest import diagnose

#: Marks a row as seeded, so a re-run replaces its own cases and never touches
#: one that arrived from Razorpay.
SEED_MERCHANT_PREFIX = "sim_"


def _channel(value: str) -> LossChannel:
    return LossChannel(value)


def seed(n: int, world: str, seed_no: int, *, budget: int) -> dict:
    from app.simulation.arms import load

    frame = load(world, seed_no)
    frame = frame[frame.split == "test"].head(n).reset_index(drop=True)

    init_db()
    db = SessionLocal()
    try:
        # Re-running the seed replaces only what the seed created. The filter is
        # on the merchant prefix, so a case that arrived from Razorpay is not
        # reachable from here even by accident -- which matters, because this
        # deletes from the append-only ledger, the one table nothing else may
        # touch. Seeded rows are build output; a real case never is.
        ids = list(
            db.scalars(
                select(Case.id).where(Case.merchant_id.startswith(SEED_MERCHANT_PREFIX))
            ).all()
        )
        if ids:
            from app.models import Action, CaseEvent

            db.query(CaseEvent).filter(CaseEvent.case_id.in_(ids)).delete(
                synchronize_session=False
            )
            db.query(Action).filter(Action.case_id.in_(ids)).delete(synchronize_session=False)
            db.query(Case).filter(Case.id.in_(ids)).delete(synchronize_session=False)
            db.commit()
            print(f"  replaced {len(ids)} previously seeded cases")

        now = datetime.now(UTC)
        # Spread detection across the last five days: inside the seven-day
        # recovery window, but old enough that the age-based rules can fire.
        span_h = 5 * 24
        made = []
        for i, row in frame.iterrows():
            detected = now - timedelta(hours=(i * span_h / max(len(frame) - 1, 1)))
            case = Case(
                external_ref=str(row.payment_id),
                order_ref=str(row.order_id),
                channel=_channel(str(row.channel)),
                merchant_id=SEED_MERCHANT_PREFIX + str(row.merchant_id),
                customer_id=str(row.customer_id),
                amount_paise=int(row.amount_paise),
                detected_at=detected,
                status=CaseStatus.OPEN,
                error_code=None if _isna(row.error_reason) else "BAD_REQUEST_ERROR",
                error_source=None if _isna(row.error_source) else str(row.error_source),
                error_step=None if _isna(row.error_step) else str(row.error_step),
                error_reason=None if _isna(row.error_reason) else str(row.error_reason),
                payment_method=str(row.method),
                issuer=str(row.issuer),
                customer_contact="+910000000000",
                customer_email=None,
            )
            db.add(case)
            db.flush()
            ledger.append(
                db,
                case_id=case.id,
                event_type="case.opened",
                actor="seed",
                payload={
                    "channel": case.channel.value,
                    "amount_paise": case.amount_paise,
                    "error_reason": case.error_reason,
                    "error_source": case.error_source,
                    "error_step": case.error_step,
                    "simulated": True,
                },
            )
            made.append(case)
        db.commit()

        for case in made:
            diagnose(db, case)
        db.commit()

        estimator = _estimator(world, seed_no)
        allocator = Allocator(
            estimator=estimator,
            budget_policy=BudgetPolicy(max_contacts_per_customer=2, max_total_contacts=budget),
        )
        # Two passes. The first plans and executes; the second is what produces
        # the `action_already_pending` and `max_actions_reached` stops that make
        # the rules panel show refusals rather than a single empty category.
        first = live_allocator.allocate_and_execute(db, allocator, limit=n, live=False)
        second = live_allocator.allocate_and_execute(db, allocator, limit=n, live=False)

        by_status = dict(
            db.execute(select(Case.status, func.count()).group_by(Case.status)).all()
        )
        return {
            "seeded": len(made),
            "pass_1": first,
            "pass_2": second,
            "by_status": {k.value: v for k, v in by_status.items()},
        }
    finally:
        db.close()


def _isna(value: object) -> bool:
    import pandas as pd

    return value is None or (not isinstance(value, str) and pd.isna(value))


def _estimator(world: str, seed_no: int):
    from app.model.train import build_frames

    try:
        return CauseRate().fit(build_frames(world, seed_no)["train"])
    except FileNotFoundError:
        from app.allocator.estimator import AmountOnly

        return AmountOnly()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", type=int, default=400, help="cases to seed")
    parser.add_argument("--world", default="base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=120)
    args = parser.parse_args()

    print(f"seeding {args.n} SIMULATED cases from world={args.world} seed={args.seed}")
    result = seed(args.n, args.world, args.seed, budget=args.budget)
    print(f"  seeded            {result['seeded']}")
    print(f"  pass 1            {result['pass_1']}")
    print(f"  pass 2            {result['pass_2']}")
    print("  cases by status:")
    for status, n in sorted(result["by_status"].items()):
        print(f"    {status:24}{n:>6}")


if __name__ == "__main__":
    main()
