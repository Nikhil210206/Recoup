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

## 2026-08-23 — Day 1

### The most interesting failure cause was impossible to generate

**Expected:** the generator would produce all 14 canonical causes, including
`merchant_config` — the one discovered from the real day-0 payload, where every
customer-facing recovery action is wrong.

**Actually:** validation showed 13 causes present and `merchant_config` at 0%.
It appeared in the recoverability table and in the Razorpay reason map, but I
had never added it to any of the per-method cause mixes. Nothing errored. The
distributions summed to 1.0 and every test passed, because the cause simply had
no probability mass.

**Why it matters more than the fix:** `merchant_config` is the trap case. It has
`p_retry = 0` and `p_nudge = 0` by construction, so any agent that contacts those
customers provably spends money for a structurally impossible return. It is the
single clearest case where a naive "payment failed → chase the customer" agent
loses and a cause-aware allocator wins. Shipping an evaluation without it would
have quietly removed the strongest evidence for the whole thesis.

**Fix:** added at 4% of the card mix, rebalanced the rest to sum to 1.0, and
added a test asserting the cause is generated *and* that both customer-facing
recovery probabilities are exactly zero. The test fails loudly if the trap case
ever disappears again.

**Pattern, second time in two days:** everything green while a thing I believed
was being exercised was not being exercised at all. Day 0 it was the database,
today it was a cause with no probability mass. Both were caught by checking
distributions against intent rather than by any test failing. That is now a
habit: assert on the *shape of the output*, not just that the code ran.

---

### Four bugs in the generator, all of which shipped green

Asked for a second, adversarial pass over day 1 rather than a re-run of the
checks that had already passed. Found four defects. Every one of them was live
while 21 tests passed, lint was clean, and every distribution summed to 1.0.

**1. Customer history accumulated in processing order, not chronological order.**
The generation loop ran `day -> merchant -> transaction`, so a customer who paid
at two merchants on the same day could carry "prior" history containing events
that happen *later* in wall-clock time. Visible as `prior_attempts` going
backwards: 4 at 12:44, then 3 at 14:45. 41% of customers are active at more than
one merchant, so this was not an edge case — it was look-ahead contamination
across a large fraction of the exact features the model trains on, in a project
whose credibility rests on a clean time-based split.

Fixed by splitting generation into two phases: lay out every attempt, sort
chronologically, then walk in time order accumulating history. 52 negative time
gaps went to 0; non-monotonic customers went from ~41% to 0.

**2. The weekend uplift was applied to Friday and Saturday.** I encoded weekday as
`(weekday() + 1) % 7` and then boosted days 5 and 6, which under that rotation are
Friday and Saturday. Now uses Python's native `Monday=0..Sunday=6` and boosts 5
and 6 directly. Blended weekend uplift measures 1.133x against an expected
1.13–1.16x for retail-weighted volume.

**3. `-1` used as a "no prior failure" sentinel** in a numeric duration column,
on 48% of rows. A model reading that column treats "never failed before" as
"failed an hour in the future". Replaced with `NaN` plus an explicit
`has_prior_failure` flag.

**4. `prior_attempts` included the current attempt while `prior_failures`
excluded the current failure.** Internally inconsistent, so the derived success
rate was computed against a denominator that included the very failure being
predicted. All `prior_*` fields now describe strictly what was knowable before
the attempt.

Also tightened: split boundaries now advance to a change of timestamp, so no
single instant can appear on both sides of a chronological division.

**The pattern, now three days running.** Day 0: a green suite running against the
wrong database. Day 1 morning: a cause with no probability mass. Day 1 review:
four defects behind 21 passing tests. Nothing here was caught by a test failing.
Every one was caught by comparing the *shape of the output* against what I
believed the code did.

The generalisation I am acting on: for a system whose output is a number nobody
can independently check, passing tests are necessary and nowhere near
sufficient. Seven regression tests now pin these specific properties, and the
evaluation harness will print its own configuration on every run rather than
letting me assume it ran the way I intended.

---

## 2026-08-24 — Days 2-3

### Contact-everything was being scored as far less spammy than it is

**Expected:** the maximal-intervention baseline would show a high unnecessary-
contact rate — it messages everyone, and ~22% of those customers were going to
pay anyway.

**Actually:** it reported **zero** unnecessary contacts across 2,383 cases.

**Why:** `simulate_action` resolved a *pending* self-recovery inside the failed-
action branch. If an early retry failed while the customer was going to pay on
their own three days later, the function returned `recovered=True`, the episode
runner saw a recovery and terminated, and every remaining contact in the policy's
ladder never fired. 377 of 2,383 episodes ended this way.

The effect was one-directional and flattering to the wrong arm: contact-
everything looked cheaper (3,424 contacts instead of 4,031), incurred less cost,
and appeared to spam nobody — while its recovery numbers were untouched, because
those cases recovered either way. A comparison that undercounts one arm's costs
while leaving its revenue intact is worse than no comparison.

**Fix:** `simulate_action` now reports only what that action did. The episode
runner owns the timeline and applies any self-recovery after the policy has
finished acting. Unnecessary contacts went 0 → 607 (15% of contacts), and the
cost side finally reflects reality.

**What it changed about the conclusions.** With costs counted properly, running
the arms across all three worlds showed the ranking is *not* stable: in the
pessimistic world, cause-aware recovers more incremental revenue than contact-
everything (24.7% vs 23.5%) using 388 contacts instead of 4,997. That flip is a
real finding and I would have missed it entirely — the bug was hiding the cost
axis on which the flip happens.

It also moved the headline metric. Raw incremental revenue rewards contacting
everyone. Incremental rupees *per contact* does not, and it separates the arms by
almost 9x (Rs 13,260 vs Rs 1,532). That is now the primary number.

---

### Stated limitation: multi-touch is probably modelled optimistically

Not a bug, but worth recording before it becomes an embarrassment in a panel.

Repeated contact attempts draw independently, with only contact fatigue (λ)
coupling them. Real customers who ignore two messages are systematically
different from customers who have not been messaged yet — unresponsiveness
persists in ways a per-attempt decay does not fully capture. The consequence is
that long escalation ladders look better here than they would in production.

Concretely: contact-everything reaches 39.8% incremental in the base world,
against Razorpay's published "up to 20%" for their own multichannel recovery
product. Some of that gap is that the published figure is a single product on
real traffic and this is a four-step ladder in a simulator, but I do not think
all of it is.

This is why the fatigue parameter is swept rather than trusted (0.40 / 0.55 /
0.75), and why the efficiency metric matters more than the gross one: every
mechanism I am unsure about inflates the arms that contact more, so the arm that
wins on rupees-per-contact is the conclusion I would defend.

---

### Day 2-3 audit: a policy was reading the answer key

Second adversarial pass, this time on the simulator. Four defects, and the first
would have invalidated an entire arm.

**1. `CauseAware` read `latent_cause` and `latent_best_delay_h` directly.**

I had written, in the module docstring of the generator, that nothing outside the
outcome simulator may read a `latent_` field. Then I wrote a policy that reads
two of them. It is the first thing a reviewer would grep for, and finding it
would justify discarding every number in the evaluation — not just that arm's.

The irony is that it was not even necessary. The cause is recoverable from
Razorpay's own error fields at 100% accuracy (that round-trip is already a test),
and `best_delay_h` is a published per-cause constant, not a per-case secret. The
policy now classifies from `error_reason` / `error_source` / `error_step` like a
live system would, and refuses to act at all on an unmapped reason rather than
guessing.

Fixed structurally as well as literally: a test now walks the AST of
`policies.py` and fails if any class not explicitly marked
`reads_ground_truth = True` contains a `latent_*` string literal. Verified by
injecting a cheating policy and confirming the guard fires.

**2. Contacts were charged for messages sent to customers who had already paid.**

6.4% of all charged contacts were messages to someone whose payment had already
landed. No real recovery system sends those — it reads payment status first and
stops the workflow. Charging for them penalised contact-heavy arms for messages
they would never actually send.

Note the direction: this biased the comparison the *opposite* way from the
day-2 bug. One was inflating the cost side, the other suppressing it. Both were
invisible in the totals.

Also separated two things I had conflated. An *unnecessary contact* is reaching
someone who was going to pay anyway, at a moment when that was unknowable — a
genuine policy cost. Messaging someone who has already paid is a status-checking
bug, not a policy decision. Only the first belongs in the metric.

**3. The "oracle ceiling" was not a ceiling.** I added an upper-bound arm to make
"% of recoverable revenue captured" meaningful, and a four-step baseline promptly
scored 126% of it — because the oracle was single-shot. A ceiling that the arms
exceed is worse than no ceiling: it silently converts a headroom statement into
nonsense. It now runs under the same four-action budget, and a test asserts no
arm can exceed it.

**4. Win/loss counts were misleading.** The paired comparison reported
contact-everything as "worse than T+3 on 54% of cases", which reads as a
damning result. Decomposing it: 89% of those losses were under Rs 30 — it paid
for a message on a case T+3 recovered for free. Genuine losses were 141 cases,
not 1,282. Differences below the cost of a couple of contacts are now reported as
immaterial rather than as defeats: **materially better on 35%, worse on 6%,
within Rs 30 on 59%.**

**What I added because the audit showed it was missing.** Common random numbers
make arm comparisons paired, which is a much stronger statistical position than I
was actually using — I was reporting differences with no evidence they were not
noise. There is now a paired bootstrap with 95% intervals on every headline
comparison.

**Pattern, fourth consecutive review.** Every defect so far has been found by
interrogating the *shape* of an output against intent, never by a test failing.
The four here were sitting behind 46 passing tests. What is actually working is
asking, of each number: what would this look like if it were wrong, and is that
distinguishable from what I am seeing?

---
