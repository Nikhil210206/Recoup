"""The bridge from live cases to executed actions.

Until now the allocator planned over parquet files. Nothing connected a case that
arrived on a real webhook to an action anybody took, which meant the two halves
of this project had never met: the ingestion path was live and tested, the
decision path was measured and reproducible, and there was no code between them.

This module is that code. It loads open cases from Postgres, shapes them into
what the allocator expects, plans, applies the stopping rules, and hands each
surviving decision to the action layer.

## Stopping rules live here, not in the allocator

The allocator reasons about a batch under a budget. Stopping rules are about a
case's own history -- has it already recovered, have we already tried three times,
did this customer ask us to stop -- and that history lives in the database. Asking
the allocator to know about it would mean giving a batch optimiser a database
handle, and the reason it is testable is that it has neither.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.allocator.budget import IST, BudgetPolicy
from app.allocator.policy import Allocator, Decision
from app.models import Action, ActionStatus, Case, CaseEvent, CaseStatus
from app.services import actions as action_tools
from app.services import ledger
from app.simulation.outcomes import CONTACT_ACTIONS, ActionType

#: A case gets at most this many actions, ever. Beyond it the case is closed
#: rather than worked further: a fourth attempt on the same failure is not
#: persistence, it is harassment with extra steps.
MAX_ACTIONS_PER_CASE = 3

#: How long a case stays worth working. Past this the customer has bought
#: elsewhere or forgotten, and attributing a later payment to us would be
#: generous to the point of dishonesty.
CASE_TTL_HOURS = 7 * 24

#: Statuses that mean the case is finished, whatever the allocator thinks.
TERMINAL = {
    CaseStatus.RECOVERED,
    CaseStatus.CLOSED_UNRECOVERED,
    CaseStatus.SUPPRESSED,
}


@dataclass
class StopRule:
    name: str
    reason: str


class Outcome(enum.StrEnum):
    """What happened to a claimed action.

    `DEFERRED` exists because it is neither of the other two. Folding it into
    EXECUTED would report a contact that has not happened; folding it into
    FAILED would report an error where the system did exactly what its own
    policy asked. The distinction is the whole point of a deferral.
    """

    EXECUTED = "executed"
    DEFERRED = "deferred"
    FAILED = "failed"


def check_stopping_rules(db: Session, case: Case, now: datetime) -> StopRule | None:
    """Reasons to leave a case alone that have nothing to do with its economics.

    Checked before the allocator is consulted, because none of these are
    trade-offs. A recovered case is not a cheap opportunity, it is a finished one.
    """
    if case.status in TERMINAL:
        return StopRule("terminal_status", f"case is {case.status}")

    if case.status == CaseStatus.EXCEPTION:
        return StopRule("on_exception_list", "cause unresolved; awaiting a human")

    if case.status == CaseStatus.PENDING_DIAGNOSIS:
        return StopRule("awaiting_diagnosis", "classification deferred; not yet actionable")

    age_h = (now - case.detected_at).total_seconds() / 3600.0
    if age_h > CASE_TTL_HOURS:
        return StopRule("past_recovery_window", f"case is {age_h / 24:.1f} days old")

    # A case already awaiting a decision must not be re-planned.
    #
    # Idempotency is keyed on (case_id, action_type, attempt_no), and each pass
    # increments the attempt. So without this rule every allocation pass created a
    # *new* action for a case still sitting in the approval queue: the same
    # Rs 99,000 case appeared three times after three passes, and a reviewer
    # could have approved it three times and sent three payment links. The
    # idempotency key was working exactly as designed and protecting nothing,
    # because the thing being repeated was the decision, not the attempt.
    # A human said no. That decision is terminal for the case, not just for the
    # action row.
    #
    # Without this, rejection was silently discarded: the action became terminal,
    # the case stayed DIAGNOSED, and the next tick proposed a fresh action at
    # attempt_no+1. A reviewer who rejected a case got it back every tick,
    # forever -- three cycles produced three rejected actions for one case. The
    # queue was asking the same question until it got the answer it wanted.
    rejected = db.scalar(
        select(func.count())
        .select_from(Action)
        .where(Action.case_id == case.id)
        .where(Action.status == ActionStatus.REJECTED)
    )
    if rejected:
        return StopRule(
            "human_rejected",
            f"a reviewer refused this action {rejected} time(s); not re-proposing",
        )

    awaiting = db.scalar(
        select(func.count())
        .select_from(Action)
        .where(Action.case_id == case.id)
        .where(Action.status.in_([ActionStatus.PENDING_APPROVAL, ActionStatus.PROPOSED]))
    )
    if awaiting:
        return StopRule(
            "action_already_pending",
            f"{awaiting} action(s) awaiting approval or execution",
        )

    taken = db.scalar(
        select(func.count())
        .select_from(Action)
        .where(Action.case_id == case.id)
        .where(Action.status == ActionStatus.EXECUTED)
    )
    if (taken or 0) >= MAX_ACTIONS_PER_CASE:
        return StopRule("max_actions_reached", f"{taken} actions already executed")

    return None


def _already_stopped_for(db: Session, case: Case, rule: StopRule) -> bool:
    """Whether the most recent ledger event is already this same stop."""
    latest = db.scalar(
        select(CaseEvent)
        .where(CaseEvent.case_id == case.id)
        .order_by(CaseEvent.seq.desc())
        .limit(1)
    )
    return (
        latest is not None
        and latest.event_type == "case.stopped"
        and latest.payload.get("rule") == rule.name
    )


def open_cases(db: Session, limit: int = 500) -> list[Case]:
    """Cases eligible for allocation, oldest first.

    Only `DIAGNOSED` cases qualify. An undiagnosed case has no cause, and every
    downstream decision -- which action, whether to contact at all -- is a
    function of the cause.
    """
    return list(
        db.scalars(
            select(Case)
            .where(Case.status == CaseStatus.DIAGNOSED)
            .order_by(Case.detected_at)
            .limit(limit)
        )
    )


def customer_history(db: Session, customer_ids: list[str]) -> dict[str, dict]:
    """Prior failure counts per customer, from the case store.

    The live path used to hand the feature builder zeros and `"unknown"` for
    every history field. Nothing broke, because the estimator in use is a
    group-by over cause and action that ignores them -- but the model was
    *measured* on real history and would have *run* on placeholders, and that
    kind of gap does not announce itself. It shows up as a model that performed
    well in evaluation and mysteriously does not in production.

    So the history is computed for real. It is thinner than the simulator's --
    a live deployment sees only failures, not the successful payments in
    between -- and that difference is stated rather than papered over.
    """
    if not customer_ids:
        return {}

    rows = db.execute(
        select(
            Case.customer_id,
            func.count().label("failures"),
            func.max(Case.detected_at).label("last_failure"),
        )
        .where(Case.customer_id.in_(customer_ids))
        .group_by(Case.customer_id)
    ).all()
    return {
        row.customer_id: {
            "failures": int(row.failures),
            "last_failure": row.last_failure,
        }
        for row in rows
    }


def to_frame(cases: list[Case], db: Session | None = None) -> pd.DataFrame:
    """Shape live cases into the frame the allocator was measured on.

    The columns are the allocator's contract. Building it explicitly here, rather
    than letting the allocator read the ORM, is what keeps the same planning code
    exercised by both the reproducible evaluation and the live path -- if they
    diverged, every measured number would stop describing what actually runs.
    """
    now = datetime.now(UTC)
    history = customer_history(db, [c.customer_id for c in cases]) if db is not None else {}

    rows = []
    for case in cases:
        detected = case.detected_at
        if detected.tzinfo is None:
            detected = detected.replace(tzinfo=UTC)
        hours_since = (now - detected).total_seconds() / 3600.0

        prior = history.get(case.customer_id, {})
        # This case is in the store too, so exclude it from its own history.
        prior_failures = max(int(prior.get("failures", 0)) - 1, 0)
        last_failure = prior.get("last_failure")
        if last_failure is not None and last_failure.tzinfo is None:
            last_failure = last_failure.replace(tzinfo=UTC)
        gap_h = (
            (detected - last_failure).total_seconds() / 3600.0
            if last_failure is not None and last_failure < detected
            else None
        )

        rows.append(
            {
                "case_id": case.id,
                "customer_id": case.customer_id,
                "merchant_id": case.merchant_id,
                "merchant_category": "unknown",
                "channel": case.channel.value,
                "amount_paise": case.amount_paise,
                "method": case.payment_method or "unknown",
                "issuer": case.issuer or "unknown",
                "error_reason": case.error_reason,
                "error_source": case.error_source,
                "error_step": case.error_step,
                # Hour of day in IST: the feature means "what time was it where
                # the customer is", and the store holds UTC.
                "hour_ist": detected.astimezone(IST).hour,
                "weekday": detected.astimezone(IST).weekday(),
                "is_weekend": detected.astimezone(IST).weekday() in (5, 6),
                "is_peak_hour": detected.astimezone(IST).hour in (12, 13, 19, 20, 21),
                "customer_segment": "unknown",
                "customer_prior_attempts": prior_failures,
                "customer_prior_failures": prior_failures,
                # A live system sees failures, not the successes between them, so
                # an observed success rate cannot be computed. NaN says "unknown",
                # which is true; zero would say "always fails", which is not.
                "customer_observed_success_rate": None,
                "has_prior_history": prior_failures > 0,
                "hours_since_last_failure": gap_h if gap_h is not None else hours_since,
                "has_prior_failure": gap_h is not None,
                "customer_ltv_paise": case.amount_paise,
            }
        )
    return pd.DataFrame(rows)


def allocate_and_execute(
    db: Session,
    allocator: Allocator,
    *,
    limit: int = 500,
    live: bool = False,
    now: datetime | None = None,
) -> dict:
    """Plan over the open cases and execute what survives every gate.

    `live=False` records actions without calling Razorpay, which is the default.
    An allocation pass that creates real payment links every time it runs is one
    nobody can safely run twice.

    `now` is injectable so behaviour that depends on the clock can be tested at a
    chosen instant. Quiet hours mean this path genuinely behaves differently at
    23:00 IST than at 14:00, and a test that reads the wall clock passes all
    afternoon and fails at night.
    """
    now = now or datetime.now(UTC)
    candidates = open_cases(db, limit=limit)
    if not candidates:
        return {"considered": 0, "planned": 0, "executed": 0, "stopped": {}, "queued": 0}

    stopped: dict[str, int] = {}
    eligible: list[Case] = []
    for case in candidates:
        rule = check_stopping_rules(db, case, now)
        if rule is None:
            eligible.append(case)
            continue
        stopped[rule.name] = stopped.get(rule.name, 0) + 1
        # Only record a stop that is new information. A case waiting in the
        # approval queue is examined on every tick, and logging "still pending"
        # each time buried the events that mattered under hundreds of identical
        # rows -- an append-only ledger is only useful if what gets appended is
        # a change.
        if not _already_stopped_for(db, case, rule):
            ledger.append(
                db,
                case_id=case.id,
                event_type="case.stopped",
                actor="stopping_rules",
                payload={"rule": rule.name, "reason": rule.reason},
            )
    db.commit()

    if not eligible:
        return {
            "considered": len(candidates),
            "planned": 0,
            "executed": 0,
            "stopped": stopped,
            "queued": 0,
        }

    by_id = {c.id: c for c in eligible}
    decisions, ledger_state = allocator.plan(to_frame(eligible, db), now=now)

    executed = queued = suppressed = failed = deferred = 0
    for decision in decisions:
        case = by_id.get(decision.case_id)
        if case is None:
            continue
        if not decision.acted:
            suppressed += 1
            _record_suppression(db, case, decision)
            continue

        attempt = _next_attempt_no(db, case)
        claimed = action_tools.claim(
            db,
            case,
            decision.action,
            attempt_no=attempt,
            rule=decision.rule,
            predicted_uplift=decision.predicted_uplift,
            expected_value_paise=decision.expected_value_paise,
            cost_paise=decision.cost_paise,
        )
        if claimed.duplicate:
            continue
        if claimed.action.status == ActionStatus.PENDING_APPROVAL:
            queued += 1
            continue

        outcome = _execute(db, claimed.action, case, decision, live=live, now=now)
        if outcome is Outcome.EXECUTED:
            executed += 1
        elif outcome is Outcome.DEFERRED:
            deferred += 1
        else:
            failed += 1

    return {
        "considered": len(candidates),
        "planned": len(decisions),
        "executed": executed,
        "deferred": deferred,
        "failed_to_execute": failed,
        "queued_for_approval": queued,
        "suppressed": suppressed,
        "stopped": stopped,
        "budget": ledger_state.summary(),
    }


def _next_attempt_no(db: Session, case: Case) -> int:
    taken = db.scalar(select(func.count()).select_from(Action).where(Action.case_id == case.id))
    return int(taken or 0) + 1


def _record_suppression(db: Session, case: Case, decision: Decision) -> None:
    """A deliberate decision not to act, recorded as one.

    Left implicit, an unworked case is indistinguishable from a case the system
    never saw -- and "we chose not to, for this reason" is the answer an audit
    needs.
    """
    case.status = CaseStatus.SUPPRESSED
    ledger.append(
        db,
        case_id=case.id,
        event_type="case.suppressed",
        actor="allocator",
        payload={
            "rule": decision.rule,
            "reason": decision.reason,
            "cause": decision.cause,
            "predicted_uplift": decision.predicted_uplift,
            "expected_value_paise": decision.expected_value_paise,
        },
    )
    db.commit()


def contact_due_at(case: Case, delay_h: float, now: datetime) -> datetime:
    """When a customer contact for this case may actually be sent.

    One rule, used by both the executor and the deferred-work tick, because two
    places deciding this independently is how they drift apart.

    Two things had been computed and then discarded on the live path:

    **The per-cause delay.** The allocator picks a delay from the cause -- 48h
    for insufficient funds so it lands after payday, 26h for a spent limit, 6h
    for a hard decline. The executor sent everything immediately. Timing is half
    of the "cause-aware action *and timing*" result the evaluation reports, so
    the deployed path was not running the system that was measured.

    **Quiet hours.** The allocator shifts the send time out of 21:00-09:00 IST
    and records rule `quiet_hours`, and only ever does so for actions that
    contact a customer -- retries return earlier, unrationed. The executor
    ignored the shift, so quiet hours were honoured on exactly the actions that
    touch a gateway and ignored on the ones that touch a person.

    The delay runs from **detection**, not from now: "48 hours after the payment
    failed" is the claim the evaluation scores, and a case picked up three days
    late is already due rather than owed another 48 hours.
    """
    due = case.detected_at + timedelta(hours=max(delay_h, 0.0))
    if due < now:
        due = now

    policy = BudgetPolicy()
    if policy.is_quiet_at(due):
        due = policy.next_allowed_time(due)
    return due


def _execute(
    db: Session,
    action: Action,
    case: Case,
    decision: Decision,
    *,
    live: bool,
    now: datetime,
) -> Outcome:
    """Perform the decided action, or record why it is not being performed yet.

    **The case only advances if the action succeeded.** This used to set `ACTED`
    unconditionally, and the consequence was the worst bug in the project: a
    payment link failed to create, the case was marked as acted on anyway, the
    customer later paid through a different route, and `resolve_recovery` saw an
    `ACTED` case and credited the recovery to us.

    The system claimed to have recovered money it had done nothing about. Every
    guard elsewhere in this codebase exists to stop exactly that -- the
    counterfactual attribution, the self-recovery distinction, the unnecessary-
    contact metric -- and one unconditional status assignment on the live path
    undid all of them.
    """
    due = contact_due_at(case, decision.delay_h, now)
    if decision.action in CONTACT_ACTIONS and due > now:
        action.scheduled_for = due
        ledger.append(
            db,
            case_id=case.id,
            event_type="case.action_deferred",
            actor="executor",
            payload={
                "action_id": action.id,
                "action_type": str(decision.action),
                "rule": decision.rule,
                "reason": decision.reason,
                "due_at": due.isoformat(),
                "delay_h": round(decision.delay_h, 2),
                "waiting_h": round((due - now).total_seconds() / 3600.0, 2),
            },
        )
        db.commit()
        return Outcome.DEFERRED

    if decision.action == ActionType.RETRY:
        result = action_tools.schedule_retry(db, action, case, delay_h=decision.delay_h)
    elif decision.action == ActionType.MERCHANT_ALERT:
        result = action_tools.alert_merchant(
            db,
            action,
            case,
            message=f"{case.cause} is blocking payments; only you can change it",
        )
    elif decision.action in CONTACT_ACTIONS:
        channel = str(decision.action).replace("payment_link_", "")
        result = action_tools.send_payment_link(db, action, case, channel=channel, live=live)
    else:
        result = action_tools.schedule_retry(db, action, case, delay_h=decision.delay_h)

    if not result.performed:
        # The case stays where it was. It is still open, still unworked, and a
        # later payment on it is a self-recovery rather than one we caused.
        ledger.append(
            db,
            case_id=case.id,
            event_type="case.action_did_not_execute",
            actor="executor",
            payload={
                "action_id": action.id,
                "action_type": str(decision.action),
                "detail": result.detail,
                "note": "case not advanced; nothing was done to it",
            },
        )
        db.commit()
        return Outcome.FAILED

    case.status = CaseStatus.ACTED
    action.scheduled_for = None
    db.commit()
    return Outcome.EXECUTED


def execute_due(
    db: Session,
    *,
    limit: int = 100,
    live: bool = False,
    force: bool = False,
    now: datetime | None = None,
) -> dict:
    """Carry out contacts whose deferral has expired.

    The other half of the quiet-hours fix. Deferring a contact is only correct
    if something later sends it; without this the rule would not be a deferral
    at all, it would be a silent drop, which is the failure mode the ledger
    event `case.action_deferred` would otherwise be quietly documenting.

    Runs on an explicit tick, like every other background job here.

    `force` sends contacts whose cause-implied delay has **not** yet elapsed. It
    exists because that delay is measured in hours -- six for a hard decline,
    forty-eight for insufficient funds -- and a demonstration cannot wait. An
    operator pushing a recovery early is also a real thing merchants ask for.
    Each override is written to the ledger as a human decision, because an
    override nobody can see afterwards is indistinguishable from the bug this
    scheduling was added to fix.

    **`force` does not bypass quiet hours.** The cause-implied delay is an
    optimisation and overriding it costs a little expected value. Quiet hours
    are about not waking someone at 3am, and there is no demo worth a hole in
    that.
    """
    now = now or datetime.now(UTC)
    query = (
        select(Action)
        .where(Action.status == ActionStatus.PROPOSED)
        .where(Action.scheduled_for.is_not(None))
    )
    if not force:
        query = query.where(Action.scheduled_for <= now)
    due = db.scalars(query.order_by(Action.scheduled_for).limit(limit)).all()

    sent = failed = 0
    still_quiet = 0
    policy = BudgetPolicy()
    for action in due:
        case = db.scalar(select(Case).where(Case.id == action.case_id))
        if case is None:
            continue

        # Re-check rather than trust the earlier arithmetic. A tick can run late,
        # and "it was going to be fine when we planned it" is not a reason to
        # message someone at 3am.
        if policy.is_quiet_at(now):
            action.scheduled_for = policy.next_allowed_time(now)
            still_quiet += 1
            continue

        if force and action.scheduled_for > now:
            ledger.append(
                db,
                case_id=case.id,
                event_type="case.schedule_overridden",
                actor="human",
                payload={
                    "action_id": action.id,
                    "action_type": action.action_type,
                    "was_due_at": action.scheduled_for.isoformat(),
                    "sent_early_by_h": round(
                        (action.scheduled_for - now).total_seconds() / 3600.0, 2
                    ),
                    "note": "operator sent ahead of the cause-implied delay",
                },
            )

        channel = str(action.action_type).replace("payment_link_", "")
        result = action_tools.send_payment_link(db, action, case, channel=channel, live=live)
        if result.performed:
            action.scheduled_for = None
            case.status = CaseStatus.ACTED
            sent += 1
        else:
            failed += 1
    db.commit()

    waiting = db.scalar(
        select(func.count())
        .select_from(Action)
        .where(Action.status == ActionStatus.PROPOSED)
        .where(Action.scheduled_for.is_not(None))
    )
    return {
        "due": len(due),
        "forced": bool(force),
        "sent": sent,
        "failed": failed,
        "still_in_quiet_hours": still_quiet,
        "waiting": int(waiting or 0),
    }
