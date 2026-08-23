"""Running a policy against a case over time.

A single action is rarely the whole story: a policy may retry, wait, then send a
link, then stop. `run_episode` executes that sequence against the outcome
simulator and returns one consolidated result per case.

Two invariants hold here, and both exist so that arm comparisons stay honest:

**Attempt streams are indexed separately.** Retries consume `u_retry`, contacts
consume `u_nudge`. Two policies whose first action is the same therefore see the
same draw for it, and only diverge where they actually behave differently.

**Episodes terminate on the truth, not on the policy's belief.** Once the
customer has paid -- whether we caused it or not -- the episode is over. A policy
that keeps sending links after a silent self-recovery is charged for those
contacts, because in production it would have sent them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.simulation import assumptions as A
from app.simulation.outcomes import (
    RECOVERY_WINDOW_HOURS,
    ActionType,
    CaseDraw,
    Outcome,
    simulate_action,
    simulate_no_action,
)


@dataclass(frozen=True)
class Step:
    action: ActionType
    at_hours: float
    outcome: Outcome


@dataclass
class Episode:
    """One case, one policy, start to finish."""

    case_id: str
    policy: str
    steps: list[Step] = field(default_factory=list)

    recovered: bool = False
    recovered_at_hours: float | None = None
    amount_recovered_paise: int = 0

    #: Recovered *because of* this policy. Excludes money that the counterfactual
    #: says would have arrived anyway. This is the honest numerator.
    caused_by_us: bool = False
    would_have_self_recovered: bool = False

    contacts_used: int = 0
    cost_paise: int = 0
    unnecessary_contacts: int = 0
    stopped_reason: str = ""

    @property
    def incremental_paise(self) -> int:
        """Revenue that exists only because this policy acted."""
        return self.amount_recovered_paise if self.caused_by_us else 0

    @property
    def net_incremental_paise(self) -> int:
        """Incremental revenue minus what it cost to get it.

        The quantity an allocator should actually maximise. Gross recovery can
        be increased indefinitely by contacting everyone; this cannot.
        """
        return self.incremental_paise - self.cost_paise


class Policy(Protocol):
    """Decides what to do next for one case.

    Returns `(action, at_hours)` for the next step, or `None` to stop. `history`
    holds the steps already taken, so a policy can escalate, back off, or give up.
    """

    name: str

    def next_action(self, case, history: list[Step]) -> tuple[ActionType, float] | None:
        ...


def run_episode(
    case,
    policy: Policy,
    *,
    max_steps: int = 6,
    fatigue_lambda: float = A.CONTACT_FATIGUE_LAMBDA.value,
    prior_contacts: int = 0,
) -> Episode:
    """Execute `policy` against `case` until it recovers, stops, or times out.

    `prior_contacts` carries fatigue in from earlier cases involving the same
    customer. It is how the shared contact budget makes itself felt: a customer
    already messaged twice this week converts worse, whichever case is asking.
    """
    ep = Episode(case_id=str(case["case_id"]), policy=policy.name)
    from app.simulation.outcomes import CONTACT_ACTIONS

    retry_idx = 0
    contact_idx = 0
    contacts_this_episode = 0

    for _ in range(max_steps):
        decision = policy.next_action(case, ep.steps)
        if decision is None:
            ep.stopped_reason = ep.stopped_reason or "policy_stopped"
            break

        action, at_hours = decision
        if action == ActionType.NONE:
            ep.stopped_reason = "policy_chose_no_action"
            break
        if at_hours > RECOVERY_WINDOW_HOURS:
            ep.stopped_reason = "scheduled_outside_window"
            break

        is_contact = action in CONTACT_ACTIONS
        outcome = simulate_action(
            case,
            action,
            delay_h=at_hours,
            attempt_idx=contact_idx if is_contact else retry_idx,
            prior_contacts=prior_contacts + contacts_this_episode,
            fatigue_lambda=fatigue_lambda,
        )
        if is_contact:
            contact_idx += 1
            contacts_this_episode += 1
        else:
            retry_idx += 1

        ep.steps.append(Step(action, at_hours, outcome))
        ep.contacts_used += outcome.contacts_used
        ep.cost_paise += outcome.cost_paise
        ep.unnecessary_contacts += int(outcome.unnecessary_contact)

        if outcome.recovered:
            ep.recovered = True
            ep.recovered_at_hours = outcome.recovered_at_hours
            ep.amount_recovered_paise = outcome.amount_recovered_paise
            ep.caused_by_us = outcome.caused_by_us
            ep.would_have_self_recovered = outcome.would_have_self_recovered
            ep.stopped_reason = outcome.reason
            return ep
    else:
        ep.stopped_reason = "max_steps"

    # The policy stopped without recovering. A self-recovery still lands: money
    # arriving on its own is money the merchant keeps, and a policy that did
    # nothing must be credited with it (and denied causal credit for it).
    baseline = simulate_no_action(case)
    if baseline.recovered:
        ep.recovered = True
        ep.recovered_at_hours = baseline.recovered_at_hours
        ep.amount_recovered_paise = baseline.amount_recovered_paise
        ep.would_have_self_recovered = True
        ep.caused_by_us = False
        if not ep.steps:
            ep.stopped_reason = "self_recovered_no_action"
    return ep


def self_recovery_time(case) -> float | None:
    """When this customer would pay unprompted, if they would at all.

    Exposed for the evaluation harness only. Reading this inside a policy would
    be cheating: it is the counterfactual the policy is being scored against.
    """
    draw = CaseDraw.for_case(str(case["case_id"]))
    if draw.u_self < float(case["latent_p_self_recover"]):
        return draw.self_recover_at_h
    return None
