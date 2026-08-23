"""Core persistence model.

Three design rules hold across this file, and each exists for a reason that
shows up later in evaluation or in the panel:

1. Money is stored as an integer number of paise. Never a float. Floating point
   money is how you end up with a settlement that is off by a fraction of a
   rupee across a 50,000-row batch.

2. `case_events` is append-only. Nothing updates or deletes a row. The audit
   question "why was this case contacted twice?" has to be answerable from the
   ledger alone, without trusting mutable state elsewhere.

3. Every action that touches money carries an idempotency key with a unique
   constraint behind it. Razorpay redelivers webhooks; the database, not the
   application logic, is what guarantees we act once.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def pg_enum(enum_cls: type[enum.Enum]) -> Enum:
    """Enum column that persists lowercase *values*, not member names.

    SQLAlchemy defaults to storing member names, which would put "FAILED_PAYMENT"
    in the database while the API returns "failed_payment". Keeping them
    identical means a row, a JSON export, and a log line all read the same.
    """
    return Enum(
        enum_cls,
        native_enum=False,
        values_callable=lambda cls: [member.value for member in cls],
    )


def _uuid() -> str:
    return str(uuid.uuid4())


class LossChannel(enum.StrEnum):
    """Where the revenue-at-risk case came from.

    These are the inputs the allocator arbitrates between. Each corresponds to
    a loss channel a merchant would otherwise have a separate agent chasing.
    """

    FAILED_PAYMENT = "failed_payment"
    ABANDONED_CHECKOUT = "abandoned_checkout"
    FAILED_SUBSCRIPTION = "failed_subscription"


class CaseStatus(enum.StrEnum):
    OPEN = "open"  # detected, not yet diagnosed
    # Deterministic lookup did not resolve it; queued for the LLM tail. The tail
    # cannot run inside the webhook request -- see services/ingest.py.
    PENDING_DIAGNOSIS = "pending_diagnosis"
    DIAGNOSED = "diagnosed"  # cause assigned
    ALLOCATED = "allocated"  # allocator selected an action
    ACTED = "acted"  # action executed, awaiting outcome
    RECOVERED = "recovered"  # money came back
    CLOSED_UNRECOVERED = "closed_unrecovered"
    SUPPRESSED = "suppressed"  # deliberately not acted on, with a reason
    EXCEPTION = "exception"  # could not be resolved; goes on the exception list


class CauseMethod(enum.StrEnum):
    """How the root cause was determined. Tracked so the split between
    deterministic and LLM classification is measurable rather than asserted."""

    DETERMINISTIC = "deterministic"
    LLM = "llm"
    UNMAPPED = "unmapped"


class ActionStatus(enum.StrEnum):
    PROPOSED = "proposed"
    BLOCKED = "blocked"  # policy engine refused it
    PENDING_APPROVAL = "pending_approval"
    EXECUTED = "executed"
    FAILED = "failed"


class Case(Base):
    """A single unit of revenue at risk."""

    __tablename__ = "cases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)

    # Identity of the underlying object in Razorpay (the payment id, usually).
    external_ref: Mapped[str] = mapped_column(String(64), index=True)
    # The order this case belongs to. Recovery is attributed by order, not by
    # customer: a later unrelated payment from the same person is not a recovery.
    order_ref: Mapped[str | None] = mapped_column(String(64), index=True)
    channel: Mapped[LossChannel] = mapped_column(pg_enum(LossChannel))

    merchant_id: Mapped[str] = mapped_column(String(64), index=True)
    customer_id: Mapped[str] = mapped_column(String(64), index=True)

    # Money as integer paise. 1 INR = 100 paise, matching Razorpay's own API.
    amount_paise: Mapped[int] = mapped_column(BigInteger)
    currency: Mapped[str] = mapped_column(String(3), default="INR")

    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[CaseStatus] = mapped_column(
        pg_enum(CaseStatus), default=CaseStatus.OPEN, index=True
    )

    # Razorpay's own failure taxonomy, stored raw and unmodified. Using their
    # enum rather than an invented one is what makes the classifier defensible.
    error_code: Mapped[str | None] = mapped_column(String(64))
    # customer | business | bank | gateway | issuer | NPCI
    error_source: Mapped[str | None] = mapped_column(String(32))
    error_step: Mapped[str | None] = mapped_column(String(64))  # e.g. payment_authentication
    error_reason: Mapped[str | None] = mapped_column(String(128))  # e.g. invalid_otp
    error_description: Mapped[str | None] = mapped_column(Text)

    # Canonical cause derived from the above.
    cause: Mapped[str | None] = mapped_column(String(64), index=True)
    cause_confidence: Mapped[float | None] = mapped_column()
    cause_method: Mapped[CauseMethod | None] = mapped_column(pg_enum(CauseMethod))

    payment_method: Mapped[str | None] = mapped_column(String(32))  # upi|card|netbanking|wallet
    issuer: Mapped[str | None] = mapped_column(String(64))
    attempt_no: Mapped[int] = mapped_column(Integer, default=1)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    events: Mapped[list[CaseEvent]] = relationship(back_populates="case", order_by="CaseEvent.seq")
    actions: Mapped[list[Action]] = relationship(back_populates="case")

    __table_args__ = (
        # One case per underlying object per channel. Re-ingesting the same
        # failed payment must not create a second case.
        UniqueConstraint("external_ref", "channel", name="uq_case_ref_channel"),
        Index("ix_case_merchant_status", "merchant_id", "status"),
    )


class CaseEvent(Base):
    """Append-only audit ledger.

    Nothing in the codebase may UPDATE or DELETE a row here. Every decision the
    system makes about a case leaves exactly one row, in order, with whatever
    inputs justified it.
    """

    __tablename__ = "case_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), index=True)

    # Monotonic per case, so the ledger has a stable order independent of clock skew.
    seq: Mapped[int] = mapped_column(Integer)

    event_type: Mapped[str] = mapped_column(String(64))
    # webhook | classifier | allocator | policy | executor | human
    actor: Mapped[str] = mapped_column(String(32))

    # Whatever justified the decision: model inputs, probabilities, the rule
    # that fired, the policy version, prompt hash, cost.
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    case: Mapped[Case] = relationship(back_populates="events")

    __table_args__ = (UniqueConstraint("case_id", "seq", name="uq_event_case_seq"),)


class WebhookDelivery(Base):
    """Transport-level idempotency.

    Razorpay sends an `x-razorpay-event-id` header and may redeliver the same
    event. The unique constraint on `event_id` is the actual guarantee that a
    redelivery is a no-op; the handler relies on the insert failing, not on a
    prior SELECT, because a SELECT-then-INSERT races under concurrent delivery.
    """

    __tablename__ = "webhook_deliveries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    event_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    event_type: Mapped[str] = mapped_column(String(64))
    signature_valid: Mapped[bool] = mapped_column(Boolean, default=False)
    raw_body: Mapped[str] = mapped_column(Text)

    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    result: Mapped[str | None] = mapped_column(String(64))


class Action(Base):
    """A recovery action proposed against a case.

    `idempotency_key` is (case_id, action_type, attempt_no) and carries a unique
    constraint. This is the single mechanism preventing a duplicate nudge or a
    duplicate payment link when a webhook is redelivered or a worker retries.
    """

    __tablename__ = "actions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), index=True)

    action_type: Mapped[str] = mapped_column(String(48))
    attempt_no: Mapped[int] = mapped_column(Integer, default=1)
    idempotency_key: Mapped[str] = mapped_column(String(160), unique=True, index=True)

    status: Mapped[ActionStatus] = mapped_column(
        pg_enum(ActionStatus), default=ActionStatus.PROPOSED, index=True
    )

    # Economics recorded at decision time, so the ranking can be audited after
    # the fact rather than recomputed from a model that has since changed.
    predicted_recovery_prob: Mapped[float | None] = mapped_column()
    expected_value_paise: Mapped[int | None] = mapped_column(BigInteger)
    cost_paise: Mapped[int | None] = mapped_column(BigInteger)

    blocked_reason: Mapped[str | None] = mapped_column(String(128))
    external_ref: Mapped[str | None] = mapped_column(String(64))  # e.g. payment link id

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    case: Mapped[Case] = relationship(back_populates="actions")

    @staticmethod
    def build_idempotency_key(case_id: str, action_type: str, attempt_no: int) -> str:
        return f"{case_id}:{action_type}:{attempt_no}"
