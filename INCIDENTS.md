# Incidents

A running log of things that broke while building Recoup, and what I did about
them. Written as they happened, not reconstructed afterwards.

Format: what I expected, what actually happened, why, and the fix.

---

## 2026-08-22 — Day 0

Setup day. Nothing broken yet beyond the usual environment noise. Entries start
once there is a system to break.

---

### The first real webhook returned a failure class I hadn't planned for

**Expected:** paying the test link with card `4111 1111 1111 1111` and a bad OTP
would produce `error_reason: incorrect_otp` — a customer-side authentication
failure, which is the case type I'd designed the classifier around first.

**Actually:** the payment never reached the OTP step. Razorpay returned:

```
error_source : business
error_step   : payment_initiation
error_reason : international_transaction_not_allowed
international: true
```

**Why:** `4111 1111 1111 1111` is a US-issued Visa test card, and the test
account accepts domestic cards only. So it failed at initiation, before
authentication.

**Why this matters more than the fix.** This is a failure the customer cannot do
anything about, caused by the *merchant's own configuration*. Every default
recovery action is wrong here:

- retrying the same card will fail identically, forever
- sending a payment link invites the customer to fail again
- nudging the customer blames them for the merchant's setting

The correct output is a *merchant-facing* alert — "you are losing card payments
because international cards are disabled" — and **zero customer contact**.

I would not have found this class by reasoning about it at a desk. It came from
the first real payload. It's now the anchor example for why `error_source` has to
drive the recovery decision: `customer`, `business`, `bank`, `gateway`, `issuer`
and `NPCI` failures need genuinely different responses, and collapsing them into
"payment failed, go chase the customer" is how a recovery agent burns goodwill on
losses it cannot recover.

**Fix:** no code change. Recorded as the first entry in the cause taxonomy, with
`SUPPRESS_CONTACT + ALERT_MERCHANT` as the correct action. To generate
customer-side failures for testing, use a domestic test card instead.

---

### The test suite deleted the real data it had just proved was working

**Expected:** running `make test-all` after capturing the first real Razorpay
webhook would leave that case in the database.

**Actually:** the end-of-day check reported `ERROR: relation "cases" does not
exist`. The dev database had no tables at all. The real captured payload — the
one that surfaced the `international_transaction_not_allowed` finding — was gone.

**Why:** `tests/conftest.py` called `Base.metadata.drop_all()` in a session
fixture, and the tests read `DATABASE_URL` from the same `.env` the app uses. So
"set up a clean schema for integration tests" meant "drop every table in the
development database." The tests passed, which is what made it invisible: nothing
failed, the data just stopped existing.

**Fix:** integration tests now run against a separate `recoup_test` database,
created by the fixture itself. `conftest.py` overrides `DATABASE_URL` before
`app.db` is imported (the engine is built at import time), and there is a hard
assert that refuses to run `drop_all` unless the connected database is literally
named `recoup_test`.

**The part worth keeping.** My first attempt at this fix silently did nothing.
I ran `cd backend && cat > tests/conftest.py <<EOF`, the `cd` failed because I
was already in `backend`, and `&&` short-circuited the write — but the following
line printed "conftest rewritten" anyway, and the tests still passed, so it
looked fixed. I only caught it by instrumenting the test to print which database
it was actually connected to.

Both failures have the same shape: **a green test suite that was not testing what
I believed it was testing.** That is the argument for the evaluation harness
printing its own configuration — which database, which seed, which policy version
— rather than me trusting that it ran the way I intended.

---
