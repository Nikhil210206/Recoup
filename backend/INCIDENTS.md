# Incidents

A running log of things that broke while building Recoup, and what I did about
them. Written as they happened, not reconstructed afterwards.

Format: what I expected, what actually happened, why, and the fix.

---

## 2026-08-22 — Day 0

### Postgres came up but rejected its own credentials

**Expected:** `make db` starts Postgres on 5433, integration tests connect.

**Actually:** container failed with `Bind for 0.0.0.0:5433 failed: port is
already allocated`, and the tests then failed with `password authentication
failed for user "recoup"` — which was the confusing part, because that is an
auth error, not a connection error.

**Why:** I'd picked 5433 to dodge a local Postgres already on 5432. But another
project of mine (`examgpt_db`) was already bound to 5433. So the container never
started, and SQLAlchemy connected to *that* database instead and was correctly
rejected. The misleading auth error was the real lesson: a port collision
presents as a credentials problem, because something is listening — just not the
thing you think.

**Fix:** moved the host port to 5434 and made it configurable via
`RECOUP_DB_PORT` rather than hardcoding another number that will collide again
on someone else's machine. `docker compose port db 5432` now confirms the
binding rather than trusting the Makefile's echo, which was itself printing a
stale hardcoded 5433.

---
