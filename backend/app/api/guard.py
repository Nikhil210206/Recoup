"""Who is allowed to make this service do something.

Two things are protected, and they are protected for different reasons.

**The automation endpoints.** `/tasks/*` exists to be called by a scheduler, not
by a person. On a public URL an unauthenticated scheduler endpoint is an open
invitation to drive someone else's recovery loop.

**The privileged parameters.** `live=true` makes a real Razorpay call.
`force=true` sends a contact ahead of the delay its cause implies and writes
`case.schedule_overridden` to the ledger **as a human decision**. That second one
is the serious one: this project's central claim is that the ledger reconstructs
who decided what, and an anonymous request recorded as a human override puts a
falsehood into the artifact the claim rests on. No amount of "nothing was
actually sent" repairs that.

What is deliberately *not* protected: every read, the approval queue's
approve/reject at `live=false`, and the console's demonstration. A judge opening
the deployed link must be able to use it. The console never sends `live=true`.

**Failure mode is closed in production and open in development.** With no token
configured, a production process refuses the call rather than serving it
unauthenticated -- a missing secret is a deployment mistake, and the wrong way to
handle it is to keep working.
"""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status

from app.config import get_settings

#: Sent by the scheduler and by `make real-loop`. Named for the project so it
#: cannot collide with a platform header on the way through a proxy.
HEADER = "X-Recoup-Token"


def _check(supplied: str | None) -> None:
    settings = get_settings()
    configured = settings.tasks_token

    if not configured:
        if settings.app_env == "production":
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "RECOUP_TASKS_TOKEN is not set. This endpoint is disabled rather "
                "than served without authentication.",
            )
        return

    # compare_digest, not ==. String equality on a secret leaks its prefix
    # through timing, and this one is guessable-length.
    if not supplied or not secrets.compare_digest(supplied, configured):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            f"missing or invalid {HEADER}",
            headers={"WWW-Authenticate": HEADER},
        )


def require_token(x_recoup_token: str | None = Header(default=None)) -> None:
    """FastAPI dependency for the automation endpoints."""
    _check(x_recoup_token)


def assert_privileged(supplied: str | None, *, what: str) -> None:
    """Guard a privileged parameter from inside a handler.

    Called where the parameter arrives in the body, which a dependency reading
    query strings would never see. `what` names the parameter so a rejected call
    says which one it tripped over.
    """
    settings = get_settings()
    if not settings.tasks_token and settings.app_env != "production":
        return
    try:
        _check(supplied)
    except HTTPException as exc:
        if exc.status_code == status.HTTP_401_UNAUTHORIZED:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                f"{what} requires {HEADER}",
                headers={"WWW-Authenticate": HEADER},
            ) from exc
        raise
