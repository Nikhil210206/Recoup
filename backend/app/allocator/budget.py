"""The shared contact budget, and the collision guard that defends it.

## The problem this exists for

Razorpay's Agent Studio ships several recovery agents -- Subscription Recovery,
Abandoned Cart Conversion -- and each optimises its own queue. Nothing published
describes a layer above them. So when one customer has a failed subscription
charge on Monday and abandons a cart on Wednesday, two agents each correctly
decide to message them, and the customer receives two messages from one merchant
in one week. Neither agent is wrong. Nobody owns the total.

## Why the total is not simply additive

Contact fatigue compounds. With a decay of 0.55, a second message to the same
customer is worth 55% of the first and a third 30%. Three contacts spent on one
customer buy 1.85 contacts' worth of effect; the same three spread over three
customers buy 3.00. The budget is not a cost constraint -- messages are cheap --
it is a constraint on a customer's finite tolerance, and it is consumed whether
or not the message works.

## What does and does not consume it

Retries do not. A retry touches the gateway, not the customer, and a failed
subscription charge can often be recovered with no contact at all. An abandoned
checkout has no payment to re-attempt, so a contact is the only route. That
asymmetry is the whole reason there is something to arbitrate.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class BudgetPolicy:
    """Limits the allocator may not exceed. Enforced in code, never inferred."""

    #: A customer's tolerance over a rolling window. The single most important
    #: number here: it is what one merchant is allowed to spend of one person's
    #: patience, across every loss channel at once.
    max_contacts_per_customer: int = 2
    window_days: int = 7

    #: Total contacts available for the batch. Models finite send capacity or a
    #: deliberate cap while a merchant builds trust in the system.
    max_total_contacts: int | None = None

    #: No customer contact inside these hours, IST. Messaging someone at 3am
    #: about a failed payment is a way to lose them, not recover them.
    quiet_hours_start: int = 21
    quiet_hours_end: int = 9

    def in_quiet_hours(self, hour_ist: int) -> bool:
        if self.quiet_hours_start > self.quiet_hours_end:  # wraps midnight
            return hour_ist >= self.quiet_hours_start or hour_ist < self.quiet_hours_end
        return self.quiet_hours_start <= hour_ist < self.quiet_hours_end

    def next_allowed_hour(self, hour_ist: int) -> int:
        """When a contact deferred out of quiet hours may be sent."""
        return self.quiet_hours_end if self.in_quiet_hours(hour_ist) else hour_ist


@dataclass
class BudgetLedger:
    """Tracks what has been spent, and refuses what would exceed it.

    Deliberately not a model. Every method here is arithmetic a person can check.
    A cap a model can be talked out of is not a cap, and this is the component
    that stands between an optimiser and a customer's inbox.
    """

    policy: BudgetPolicy = field(default_factory=BudgetPolicy)
    _contacts: dict[str, list[datetime]] = field(default_factory=lambda: defaultdict(list))
    total_spent: int = 0

    #: Every refusal, counted by reason. This is the evidence the budget is
    #: load-bearing: "zero violations" is true by construction and proves
    #: nothing, whereas "N blocked, by this rule" shows the gate doing work.
    blocked: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def contacts_in_window(self, customer_id: str, at: datetime) -> int:
        cutoff = at - timedelta(days=self.policy.window_days)
        return sum(1 for ts in self._contacts[customer_id] if ts > cutoff)

    def can_contact(self, customer_id: str, at: datetime) -> tuple[bool, str]:
        """(allowed, reason). The reason is recorded even on success, so the
        audit trail says which constraint was closest to binding."""
        if (
            self.policy.max_total_contacts is not None
            and self.total_spent >= self.policy.max_total_contacts
        ):
            return False, "global_budget_exhausted"

        used = self.contacts_in_window(customer_id, at)
        if used >= self.policy.max_contacts_per_customer:
            # The collision guard. This is where two agents chasing the same
            # person in one week gets stopped, and neither of them would have
            # stopped it alone.
            return False, "customer_contact_cap"

        return True, f"ok_{used}_of_{self.policy.max_contacts_per_customer}"

    def spend(self, customer_id: str, at: datetime) -> None:
        self._contacts[customer_id].append(at)
        self.total_spent += 1

    def refuse(self, reason: str) -> None:
        self.blocked[reason] += 1

    @property
    def customers_contacted(self) -> int:
        return sum(1 for v in self._contacts.values() if v)

    def summary(self) -> dict:
        per_customer = [len(v) for v in self._contacts.values() if v]
        return {
            "total_contacts": self.total_spent,
            "customers_contacted": len(per_customer),
            "max_contacts_to_one_customer": max(per_customer, default=0),
            "blocked": dict(self.blocked),
        }
