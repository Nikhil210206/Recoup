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

## 2026-08-25 — Day 4

### The cheap model was not just less accurate, it was confidently wrong

Not a bug in the code — a finding that changed a design decision, which is the
reason I ran the measurement rather than picking a model by preference.

No Anthropic API key (no budget for one), so the tail runs on a local model via
Ollama. The obvious choice was the smallest thing that fits: llama3.2:3b, 2GB,
fast. Before defaulting to it I held out eight error codes that *are* in the
deterministic table, hid them from the lookup, and forced them down the LLM path,
which gives real ground truth for exactly the operation the tail performs.

| model | correct | confidently wrong | ms/call |
|---|---|---|---|
| llama3.2:3b | 4/7 | **3** | 3,870 |
| qwen2.5:7b | 8/8 | 0 | 7,364 |

The accuracy gap is not the interesting part. **All three of the 3B model's
errors came at 0.90 confidence** — above the 0.70 acting threshold — so the
confidence gate cannot catch them. A model that abstains when unsure is safe at
any accuracy; a model that is wrong at 0.90 is not.

The worst of the three: it classified `payment_risk_check_failed` as
`hard_decline`. Those two causes have opposite recovery semantics. `risk_blocked`
means never retry, this is a fraud control. `hard_decline` means try a different
instrument. The 3B model would have had the system politely work around a fraud
block, at high confidence, with a clean audit trail saying it was sure.

Default is now qwen2.5:7b. The extra 2.7GB and ~3.5s of latency buy the
difference between a tail that degrades safely and one that fails dangerously.
That is the entire justification, and it is a number rather than a preference.

**And then the 7B model failed the same way, which changed the architecture.**

The integration test I wrote to pin the fix promptly failed. qwen2.5:7b
classified an error code reading `issuer_fraud_suspicion_2027` — description:
"The issuer declined the transaction as suspected fraud" — as `hard_decline` at
0.95 confidence. Reproducibly, at temperature 0. It got four other phrasings of
the identical case right; it appears to anchor on the token "issuer" in the code
name over the description.

So the 8/8 result was optimistic in exactly the way I had already flagged in the
harness docstring, and here was the evidence.

The response is not a better prompt. A probabilistic component is fine for
mapping an unfamiliar code onto a taxonomy; it is not fine as the only thing
between a fraud block and an automated retry. There is now a **deterministic
guardrail applied after the model and not overridable by it**: if the input
carries a risk or fraud signal, the model may not return a cause that permits
retrying or contacting. The case goes to a human.

Escalating a genuine hard decline costs one review. Retrying a fraud block costs
more than that. Verified: the failing phrasing now escalates, correctly-classified
fraud cases still pass through as `risk_blocked`, and ordinary declines with no
risk signal are untouched — the guardrail is not a blanket refusal to classify.

**The test was also wrong.** It asserted a specific model output, which is flaky
by construction and makes a probabilistic component look more reliable than it
is. It now asserts the safety property — *no fraud case is ever marked
retryable* — which is what actually matters and is guaranteed by code rather than
by the model behaving.

**Why this is the right shape of answer even though it is a local model.**
Claude remains the documented production path and the aligned one — Razorpay's
Agent Studio is built on the Claude Agent SDK, and the Anthropic provider is
implemented and used automatically whenever a key is present. But the tail is
closed-set classification over fifteen labels on roughly 1% of traffic. Whether
that needs a frontier model is a question to answer with a measurement. If the
small models had failed the safety check, the honest conclusion would have been
that the tail needs one and the cost is justified.

### Everything else that was checked

- Repeat unknown codes hit a cache: the tail costs money per *code*, not per
  payment. A new Razorpay code appears across thousands of payments.
- Backend outage degrades to the exception list rather than raising. Verified by
  monkeypatching the provider to fail. Payment ingestion keeps working.
- A model returning a label outside the enum is treated as an abstention, not a
  cause. The schema is the containment, not the prompt — so even a fully
  successful prompt injection through `error_description` can at worst produce a
  wrong cause, which the confidence gate then routes to a human.

---

### Day 4 audit: a 31-second model call inside a webhook handler

Five defects. The first was the worst thing found in the project so far, and it
would only have shown up in production.

**1. The classifier ran synchronously inside the webhook request.**

`ingest.diagnose` called the full three-tier classifier, LLM tail included, while
Razorpay waited for a response. Measured cold-path latency: **31 seconds.**

The failure this produces is not a slow endpoint. It is: delivery times out ->
Razorpay retries -> our own idempotency correctly rejects the duplicate as
already-seen -> **the case is never classified at all**, while Razorpay records
repeated delivery failures against the webhook. Two mechanisms that are each
individually correct combine into silent data loss. Nothing errors. The case just
sits at `open` forever.

Fixed by splitting diagnosis. The deterministic tier runs inline at ~0.01 ms and
resolves every published Razorpay code. Anything it cannot resolve is parked in
`PENDING_DIAGNOSIS` and drained by `POST /tasks/classify-pending`, which runs
outside the request. Webhook latency went from 31,000 ms to 10-61 ms, verified
end to end against the running server.

Deliberately a status column and a tick endpoint rather than Celery or a broker.
The work is a bounded scan over rows with a status, the state belongs in Postgres
next to the audit trail, and a tick that can be run by hand with curl is one you
can reason about at 2am.

**2. The risk guardrail only covered the LLM path.** A published code carrying a
fraud description — `card_declined` with "blocked due to suspected fraudulent
activity" — resolved deterministically to `hard_decline` at confidence 1.0, with
the guardrail never consulted. The tier I trusted most was the one with no
safety net. Now applied to every tier.

**3. The enum offered to the model was hand-listed.** Adding a cause to the
taxonomy would silently stop offering it to the classifier — the model could
never return it, and nothing would fail. Now derived from the taxonomy, with a
test asserting they cannot drift.

**4. The tail cache keyed on the error code but not the description.** The first
description ever seen for a new code fixed the answer for every later payment
carrying it — and the description is precisely what the model reads. Now part of
the key, at a small cost in hit rate.

**5. The cache was unbounded.** A slow memory leak in a long-running process.
Now a bounded LRU.

**The near-miss worth recording.** My first attempt to add `PENDING_DIAGNOSIS`
silently did nothing: `ruff format` had reformatted the comments in that enum
earlier, so my anchor text no longer matched and the edit was a no-op. It
surfaced as a 500 at runtime rather than as a failed edit. Same shape as the
day-0 conftest that was never written — an edit that reports success while
changing nothing. I now verify the *state of the file* after any surgical edit,
not the exit code of the thing that made it.

---

## 2026-08-25 — Gap review before Day 5

Asked whether days 0-4 were "perfect". Ran a gap analysis against the plan
rather than re-running the checks that already passed, and found something worse
than a bug: **the core thesis was not expressible in the data.**

### The dataset had one loss channel; the argument needs three

Recoup's pitch is that Agent Studio's agents each optimise their own queue and
nothing arbitrates between them under a shared contact budget — Subscription
Recovery and Abandoned Cart both chasing the same customer in one week.

`LossChannel` declared three channels. One adapter existed. The synthetic
dataset had **no channel column at all**. So the contact-collision problem — the
entire product gap — could not be measured, only asserted. The allocator due on
day 7 would have had nothing to arbitrate between, and I would have discovered
that while building it.

Nothing was failing. 84 tests passed. The gap was in what the tests were *about*.

**Closed:** `subscription.pending` / `subscription.halted` adapters (Razorpay's
documented events), an abandoned-cart adapter routed by payload shape rather than
a guessed event name (Razorpay documents the payload but not a stable event
string), and a channel dimension in the generator.

The channels now genuinely pull against each other, which is the point:

| channel | retryable | recovery route |
|---|---|---|
| `failed_payment` | yes | retry, or contact |
| `failed_subscription` | yes, **without contacting anyone** — a mandate exists | retry is nearly free |
| `abandoned_checkout` | **no** — nothing failed, there is no payment to re-attempt | contact is the only route |

A subscription failure can be recovered for zero contacts. An abandoned cart
cannot be recovered any other way. Both compete for the same finite tolerance of
one customer. That is a real trade-off, and it is now measurable.

### Two bugs fell out of it immediately

**`taxonomy.classify` crashed on a missing error field.** Abandoned checkouts
carry no error code. A webhook represents that absence as `None`, a DataFrame
represents it as `NaN`, and the second one reached `.strip()` and raised. The
same classifier serves both paths, so it now coerces defensively.

**The cause-aware policy left an entire channel unrecovered while looking
frugal.** `customer_abandoned` carries `RetryPolicy.IMMEDIATE`, which is correct
for a cancelled payment and wrong for an abandoned checkout where no payment
exists. The policy retried, the retry could not work, and it recorded zero
contacts and zero recovery for 792 cases and Rs 48.8 lakh at risk. In the totals
that reads as an efficient policy rather than a broken one.

The lesson is narrower than "test more": **a cause is not sufficient to choose an
action.** The channel changes what the same cause means. The policy is now
channel-aware and the case is pinned by a test.

### An honest number moved in the wrong direction, and stays

Cause-aware's contact efficiency was 8.5x contact-everything on single-channel
data. With abandoned checkouts included it is 2.6x.

That is not a regression. Part of the old margin came from a world where
retrying was always an option, which is not the world merchants are in. The test
threshold moved from 3x to 2x with the reason written into the test, rather than
the number being quietly preserved.

### Still open, deliberately

- Multi-touch remains modelled optimistically (flagged day 3, unresolved).
- The Anthropic provider is implemented but has never executed — no key.
- `ARCHITECTURE.md` and `EVALUATION.md` are not written yet (due day 10-11).

---

## 2026-08-26 — Days 5-6

### The project's headline claim was worth 1%, and I found out by testing it

Not a bug. A wrong hypothesis, caught by measuring the thing I had assumed.

The plan's first move was: **rank cases by expected value rather than by
transaction amount.** Rs 10,000 at 90% beats Rs 50,000 at 5%. It is intuitive,
it is the reason the uplift model exists, and it was going to be the opening
beat of the pitch.

Measured on the test slice with a fixed 360-contact budget:

| lever | variant | realised | lift |
|---|---|---|---|
| action | fixed retry @ 24h | Rs 6.1L | — |
| action | fixed link @ 24h | Rs 30.2L | +394% |
| action | retry at cause-best time | Rs 20.3L | +231% |
| action | **cause-aware action + timing** | **Rs 38.2L** | **+524%** |
| ranking | amount ranked | Rs 38.2L | — |
| ranking | **EV ranked, ORACLE uplift** | Rs 38.6L | **+1.0%** |

**Ranking is worth one percent, and that is with perfect knowledge.** The oracle
row matters: this is not "the model needs to improve". The ceiling is that low.

The reason is arithmetic, not modelling. `corr(EV, amount) = 0.937`. Transaction
amounts span roughly 5,000x; uplift spans about 5x. Multiplying a hugely-varying
quantity by a mildly-varying one barely reorders it. I checked whether pooling
four merchants with 50x different ticket sizes caused it — per-merchant the
correlation is still 0.875 to 0.927. I checked whether suppression was the hidden
value — only 28 of 2,400 cases have near-zero true uplift.

**What is actually worth 500% is choosing the right verb.** The actions are not
substitutes. An expired card responds to a method-switch prompt about twenty
times better than to a retry, and no amount of clever case selection recovers
from having chosen the wrong action. This was visible on day 3 in the
per-cause action table; I had attributed the resulting efficiency gap to the
wrong mechanism.

Thesis repointed, with the user's agreement: **the right action at the right
time, or none at all, under a shared contact budget.** EV stays in the system for
the decision of *whether to act*, and the 1% figure is reported rather than
buried. A negative result about your own headline, found and published before a
panel finds it, is worth more than the claim would have been.

### Calibration did not help either, and that is also reported

Isotonic regression on the treated arm moved ECE from 0.028 to **0.032** — worse.
A logistic model optimises log loss and is close to calibrated by construction;
isotonic fitted on a few hundred rows adds variance without removing bias.

Rather than apply it anyway and write "we calibrated the model", the fit now
scores raw, isotonic and sigmoid on a slice used for nothing else and takes the
best by Brier. The chosen method is recorded per arm. On this data the baseline
arm picks sigmoid and the treated arm picks raw.

The miscalibration experiment then produced its own surprise. I expected
over-confidence to wreck the allocation. It does not — realised value moves by
about 2% — for exactly the same reason ranking does not matter. What it wrecks is
the **forecast**: an over-confident model claims Rs 49 lakh and delivers Rs 26
lakh, a 90% overstatement, at identical ROC-AUC. That is not cosmetic. A merchant
staffs and plans against the forecast, and a recovery product that habitually
promises double what it delivers stops being trusted regardless of what it
recovers.

So calibration earns its place for forecasting and for the act/do-not-act
decision, not for ordering a queue. Which is a narrower claim than the one this
project started with, and a true one.

### A Bayes ceiling, because a raw AUC is uninterpretable

The treated model reaches ROC-AUC 0.629, which reads as weak. Reconstructing the
true generating probability — including the timing kernel — gives a Bayes-optimal
AUC of **0.665**. The outcome is a coin flip weighted by p; nothing can predict a
coin flip. So the model captures about **78% of the signal that exists**.

The first version of this ceiling dropped the timing term and produced a value of
0.600 — which the model "beat" at 116%. Exceeding a Bayes ceiling is impossible,
and that impossibility was the tell that the ceiling was wrong rather than the
model exceptional. Worth recording: the check that caught it was noticing a
number that could not exist, not a test failing.

---

### Day 5-6 audit: the evaluation was not reproducible while claiming to be

Four defects. The first invalidated every number reported that day.

**1. `hash()` is randomised per process, so each training run used different data.**

`build_frames` derived each split's exploration seed as `seed + hash(split) % 1000`.
Python randomises string hashing per process unless PYTHONHASHSEED is pinned, so
the "seed 42" run was seeded differently every time. Four consecutive runs of the
identical command produced held-out ROC-AUC of **0.588, 0.604, 0.606 and 0.629**,
and the saved artifact recorded whichever happened to run last.

Nothing failed. The harness printed its configuration, as I had made it do after
day 4, and the configuration it printed was accurate — the seed really was 42.
The seed simply was not the only source of randomness, and the one that mattered
was invisible.

This is the worst class of bug in this project because reproducibility is the
claim that everything else rests on. `make eval` producing the same table twice
is the reason a panel should believe any of it.

Fixed with fixed per-split offsets. Three consecutive runs now produce
byte-identical output.

**2. Every sub-experiment hardcoded `("base", 42)`.** Running
`--world pessimistic` trained on pessimistic data, then reported a Bayes ceiling
and a lever study computed on *base* data. The world sweep — the thing that is
supposed to test whether conclusions survive a different world — was silently
comparing a pessimistic model against base-world conclusions.

Fixed by threading world and seed through. The sweep now works, and it
strengthens the day's finding rather than weakening it:

| world | action selection | EV ranking (oracle) |
|---|---|---|
| pessimistic | +528.9% | +1.1% |
| base | +524.1% | +1.0% |
| optimistic | +519.4% | +0.9% |

The conclusion that repointed the project holds in all three.

**3. The Bayes ceiling joined two frames positionally without checking.** The
case data and the training frame are produced by separate code paths and zipped
by position. If their order diverged, every case would be scored against a
different case's latent probability, and the result would be meaningless in a way
that looks entirely normal. Now asserted on `case_id` before use.

Fixing 1 and 3 together moved the headline: the model captures **88%** of the
achievable signal, not 78%. The earlier figure was comparing a model trained on
one random draw against a ceiling computed from another.

**4. `arms.load` raised `SystemExit` on a missing file.** Fine for a CLI, hostile
in a library — `SystemExit` escapes normal exception handling and tears down the
caller. It surfaced as a test that appeared to fail for an unrelated reason. Now
`FileNotFoundError`.

**The pattern, sixth review running.** Every one of these was found by asking what
a number would look like if it were wrong, and checking whether that was
distinguishable from what I was seeing. Run it three times and compare. Ask
whether the ceiling could be exceeded. Ask whether `--world pessimistic` actually
changed anything downstream.

None of the six adversarial passes has come back empty. I have stopped expecting
one to.

---

## 2026-08-27 — Day 7

### The allocator lost to a 20-line heuristic, three times, before it won

Day 7 built the allocator -- the layer the whole project is named for. It lost to
the cause-aware baseline on the first three attempts, and each loss was a real
defect rather than a tuning problem.

**Loss 1: the model chose the action, and chose it badly.** The allocator asked
the fitted uplift model which action to take. It picked `payment_link_whatsapp`
2,032 times out of 2,173 -- and on a per-cause check, it selected the true-best
action on **3 of 11 causes**.

The model had learned the correct *global* answer (contacts recover more than
retries on average) and could not express the per-cause interaction that actually
decides the action. At ROC 0.62, that interaction is a second-order effect inside
a first-order-noisy Bernoulli signal.

This is day 4's lesson in a different costume: **do not use a model to rediscover
what a published taxonomy already states.** Razorpay's error semantics say an
expired card needs a different instrument and a bank outage needs a retry and no
message. Handing action choice back to the taxonomy took it to 9 of 11.

**Loss 2: the EV floor was suppressing free retries.** The gate ordering applied
a Rs 50 expected-value threshold before checking whether the action even cost
anything. Retries touch the gateway, not the customer, and consume none of the
scarce resource -- but hundreds of them were being suppressed for failing to clear
a threshold that exists to ration *contacts*. It cost more value than every other
gate combined, and it looked like prudence.

**Loss 3: the comparison itself was rigged, against us.** Every baseline had
unlimited contacts while the allocator respected a budget. Contact-everything
spent 4,074 contacts and "won". That is not a result, it is a policy ignoring the
constraint it is supposed to respect. Under an equal budget the allocator wins by
about 21%.

A fourth bug surfaced while fixing the third: the bake-off harness only executed a
baseline's *first* step, so contact-everything recorded **zero contacts** -- its
first rung is a retry. A maximally aggressive policy was being scored as
abstemious.

### The ablation says most of the allocator does nothing

Having made it win, I measured which parts were responsible. At a 600-contact
budget:

| variant | incremental |
|---|---|
| full (cause-rate estimator, cap 2) | Rs 37.07L |
| **no uplift estimate at all** | **Rs 37.11L** |
| no contact cap | Rs 36.87L |
| per-case ML uplift model | Rs 28.09L |

The uplift estimator contributes nothing. The contact cap contributes nothing.
The per-case ML model is 24% *worse* than having no estimate.

A cause-rate group-by -- roughly twenty numbers, computable in SQL -- tracks true
uplift better (+0.362) than the fitted gradient-boosted model (+0.275). The
structure in this problem is per-cause and the per-case model adds variance
without adding signal.

So the ML model is out of the decision path. It is retained for forecasting,
where calibration measurably matters and a per-case number is what a merchant
plans against. The contact cap stays because it protects a customer from being
messaged repeatedly by agents that each believe they are the only one; costing
nothing is the argument for keeping it, not against it.

### A claim from days 5-6 was overstated, and is now corrected

I wrote "ranking is worth ~1%". That compared EV-ranking against *amount*-ranking
-- both value-aware. It never compared either against working a queue in
**arrival order**, which is what a per-case agent actually does.

Measured properly, it is two claims:

- ordering by value vs arrival order: **+21%**
- ordering by EV vs by amount: **+1%**

Collapsing those into "ranking does not matter" was wrong, and it happens to have
been wrong in the direction that made my own allocator look pointless. Both
numbers are now in a test.

### What the allocator is actually made of

In descending order of measured contribution:

1. **Action selection from the taxonomy** -- roughly +520%, deterministic, no model.
2. **Value-ordered spending of a finite contact budget** -- roughly +21% over
   arrival order.
3. **Suppression of structurally futile causes** -- deterministic.
4. Uplift estimation, contact cap, EV floor -- measurably ~0, kept for stated
   reasons.

Three days running now, the honest answer has been that a deterministic rule beat
the learned component. That is becoming the actual finding of this project rather
than an accident: in payments recovery the structure lives in a published
taxonomy, not in the outcome data.

---

### Day 7 audit: my own baseline was a strawman, in my favour

The allocator reported +21% over the best baseline. Asked to verify it, I built
the two controls I had not built, and the number did not survive.

**Bug 1: the bake-off executed one action per case.** Contact-everything's ladder
is retry, WhatsApp, SMS, SMS. The harness walked to the first *contact* step and
stopped, so its retry was discarded and a maximally aggressive multi-touch policy
was scored on a single WhatsApp message. It recorded 530,593 -- a policy designed
around escalation, measured as though it never escalated.

**Bug 2, and the serious one: every baseline pooled all three loss channels into
one queue.** Recoup's entire product argument is that Agent Studio runs separate
agents per channel and nothing coordinates them. My control had already solved
that problem for free. I was comparing a coordinated allocator against a
coordinated baseline and calling the difference coordination.

Both fixed: baselines now run their full ladder, and there is a `siloed` module
that models one agent per channel, each with its own queue and its own slice of
the budget, none aware of the others.

### The honest numbers, after

| component | measured |
|---|---|
| action selection from the taxonomy | **+520%** |
| value-ordering the budget vs arrival order | **+18.5%** |
| allocator vs siloed per-channel agents | **+1.0%** |
| allocator vs an idealised pooled + ranked agent | **-1.3%** |

Swept across budgets, the allocator's edge over siloed agents runs from +3.3% at
a tight budget to **-2.1%** at a loose one. It is noise.

The reason is in the data and I should have checked it before building: only
**9.5% of customers** have cases in more than one channel, so the cross-agent
collision this layer exists to prevent can barely occur. At a 600-contact budget
it affects 13 of 458 contacted customers, and the contact cap prevents exactly
**one** customer from being over-messaged.

### What I did about it

Reframed rather than re-tuned. The allocator is now documented as a **governance
layer, not a revenue layer**, and the ~0 is reported in the CLI output and pinned
in a test that fails if it ever silently becomes a win.

The revenue comes from action selection and value ordering, both of which a
simple per-case policy can do. What the governance layer buys, for about 1%:

- a hard ceiling on how much of one customer's patience a merchant may spend,
  enforced across every channel at once -- which no per-channel agent can promise,
  because none of them can see the others
- suppression where no customer action can help
- quiet hours
- a decision-level audit trail: why each case was acted on or left alone, which
  rule fired, what was predicted

For a payments company that is the difference between an automation that survives
a compliance review and one that does not. It is worth saying as governance
rather than dressing it up as revenue.

There is one caveat I am leaving open rather than acting on. My generator samples
customers uniformly, so only 17% have more than one case in ninety days. Real
merchant traffic is heavy-tailed and cross-channel overlap would be higher, which
would make the collision guard bind harder. That is arguably a realism bug rather
than a thesis convenience -- but changing the data after seeing the result is
exactly the move that should not be made quietly, so it is recorded here and not
made.

### The pattern, seventh review

The bug that mattered was not in the allocator. It was in the thing I was
measuring the allocator *against*, and it was wrong in the direction that
flattered me. Every previous review found defects in the system under test; this
one found that the test itself was rigged, unintentionally, by me.

Worth generalising: when a result is good, the first thing to audit is the
control, not the treatment.

---

## 2026-08-24 — Day 8

### The two halves of the project had never met

Day 8 was meant to add the policy engine, stopping rules, idempotency and the
approval queue. Mapping the plan against the code first turned up something
larger: **the allocator had never touched the database.**

It planned over parquet files. The ingestion path was live and tested; the
decision path was measured and reproducible; there was no code between them. The
`actions` table had existed since day 0, complete with a documented idempotency
key and a unique constraint, and not one row had ever been written to it. The
`build_idempotency_key` helper had never been called.

That is the gap worth naming: a project can have a well-tested front half and a
well-measured back half and still not be a system.

### The bug that would have sent three payment links

With the live path wired, allocation ran five times against the same five cases.
Executed actions were correctly idempotent. But a case sitting in the approval
queue produced a **new action on every pass**, and after three passes the same
Rs 99,000 case appeared in the reviewer's queue three times.

Idempotency is keyed on `(case_id, action_type, attempt_no)` and each pass
increments the attempt, so every proposal got a fresh key and the duplicate check
never fired. The constraint was working exactly as designed and protecting
nothing, because **what was repeating was the decision, not the attempt.**

A reviewer approving all three would have sent three payment links to one
customer for one failed payment.

Fixed with a stopping rule: a case with an action already `PENDING_APPROVAL` or
`PROPOSED` is not re-planned. Five passes now produce one queued action, and the
regression is pinned.

### Two smaller ones, from reading the audit trail

**An executed case stayed `DIAGNOSED`.** The approval endpoint executed the
action but never advanced the case, so an executed case was indistinguishable
from an unworked one in every status query -- including the dashboard's.

**The ledger was being spammed.** Every allocation pass wrote a `case.stopped`
row saying "still awaiting approval". A case in the queue for a day with a
five-minute tick would accumulate hundreds of identical rows and bury the events
that mattered. An append-only ledger is only useful if what gets appended is a
*change*, so a stop is now recorded only when it is not already the latest event.

Both were found by reading a single case's history end to end rather than by any
test failing. That query -- "show me everything that happened to this case and
why" -- has now found bugs on three separate days, and is the one I would expect
in a review.

### Deliberate choices worth recording

**`live=False` everywhere by default.** Actions are recorded without calling
Razorpay unless explicitly asked. An allocation endpoint that creates real
payment links every time it is hit is one nobody can safely run twice, and a test
suite that does it is one that eventually messages a stranger.

**Notifications are disabled even when live.** Recoup decides *whether* to
contact someone; it does not get to have Razorpay SMS a real phone number as a
side effect.

**The insert precedes the external call.** If the process dies mid-send, the row
already exists and a retry refuses to send again. The worst case is an action
recorded as sent that was not -- reconcilable by a human. Reverse the order and
the worst case is a duplicate payment link, which cannot be un-sent.

**Retries never queue for approval.** They touch the gateway rather than the
customer, so putting them behind a human stalls the cheapest recovery path for no
benefit.

---

### The rule was named for an alert it never sent

Running the end-to-end demo, the merchant-misconfiguration case printed:

```
pay_DEMO_MERCHANT   merchant_config   ->  NOTHING   [merchant_alert_only]
```

A rule called `merchant_alert_only` that emits no alert. The case was classified
correctly, suppressed correctly, and the one recoverable thing about it was
silently discarded.

This is the failure class discovered on day 0 from the first real Razorpay
payload -- a payment blocked because the merchant had international cards
disabled. No customer action can clear it, so suppressing customer contact is
right. But telling the *merchant* is the entire value: it is a setting they can
change in thirty seconds, and every payment it blocks until they do is lost.

The allocator now emits a real `merchant_alert` action. It costs nothing and
consumes no contact budget, because it goes to the merchant rather than to a
customer.

**The first fix made it worse, and the output said so.** Emitting the alert from
`preferred_action` was not enough: `MERCHANT_ALERT` is not in the
customer-facing candidate list, so the feasibility filter stripped it and the
case fell through to the *fraud*-suppression branch -- a misconfiguration
labelled as a risk block. The change applied cleanly, lint passed, and the
behaviour was wrong in a new way. Caught by re-reading the demo output rather
than by trusting that the edit had done what it said.

One existing test failed after the fix, correctly: it asserted that
merchant-config cases are never acted on, which was true before and is wrong now.
Updated to assert the thing that actually matters -- a fraud block gets nothing,
a misconfiguration gets an alert and zero customer contact.

---

### Day 8 audit: four bugs, two of which broke a promise to a person

**1. Quiet hours were enforced in the wrong timezone.**

The policy is written in IST -- "no customer contact between 21:00 and 09:00" --
and the allocator compared it against the server's UTC clock. **Ten of twenty-four
hours were classified wrongly.** A payment-recovery message could go out at
01:00 IST because UTC read 19:30, and be suppressed at 13:00 IST because UTC read
07:30.

Quiet hours are not an optimisation. They are a promise that a merchant will not
wake someone up to ask for money. Enforced in the wrong timezone the rule
protects nobody and blocks the wrong sends, while every log line and every test
says it is working.

Fixed by comparing absolute instants converted to IST rather than bare hour
integers. `is_quiet_at(datetime)` and `next_allowed_time(datetime)` replaced the
hour-arithmetic, and the deferral now lands on a real timestamp outside the
window rather than shifting a delay by a modular difference.

**2. A rejection was silently discarded.**

A reviewer rejects an action. It becomes terminal -- and the case stays
`DIAGNOSED`, so the next tick finds no pending action, proposes a fresh one at
`attempt_no + 1`, and puts the same case back in the queue.

Measured: three reject cycles produced three rejected actions for one case. The
queue was asking the same question until it got the answer it wanted, and with a
five-minute tick a reviewer would face the same rs 99,000 case 288 times a day.

Rejection is now terminal for the *case*, not just the action row.

**3. The `live=True` branch had never executed.** The only code that creates
anything at Razorpay was untested, and day 9 depends entirely on it. Now covered
against a stubbed client -- including that a gateway outage is recorded rather
than raised, and that notifications are provably disabled. The genuine
end-to-end call belongs in the recorded demo, not in a suite that runs on every
commit; a test that creates real payment links eventually messages a stranger.

Writing those tests surfaced a fourth problem: a **missing API key raised**
rather than returning, so one misconfigured deployment would take down a batch of
two thousand cases instead of failing one action. Now recorded like any other
failure, with the ledger saying which it was.

**4. The live path fed the model placeholders.**

`to_frame` handed the feature builder zeros for every customer-history field and
`"unknown"` for every category. Nothing broke, because the estimator in use is a
group-by that ignores them -- so this was invisible and would have stayed
invisible until some future model that *does* use history was measured on real
data and run on fakes. That gap does not announce itself; it shows up as a model
that evaluated well and mysteriously does not work.

History is now computed from the case store. It is genuinely thinner than the
simulator's -- a live deployment sees failures, not the successful payments
between them -- so `customer_observed_success_rate` is NaN rather than zero.
Unknown is true; "always fails" is not.

The same fix corrected `hour_ist`, which was being populated with a UTC hour
under a name that says otherwise.

### What the two halves have in common

The timezone bug and the placeholder-features bug are the same mistake wearing
different clothes: **a value that was labelled as one thing and was another.** An
hour called `hour_ist` holding UTC. A field called `in_quiet_hours(hour_ist)`
receiving a UTC hour. A frame column called `customer_prior_failures` holding
zero because nobody had computed it.

None of them fail. They all produce a number, of the right type, in the right
range, that means something other than what its name claims. That is the failure
mode this project keeps finding, and the only reliable defence has been to check
what a value actually contains rather than what it is called.

---

## 2026-08-24 — Day 9

### The real loop closed, and the server was running stale code ten minutes before it did

Rs 2,499 recovered end to end on Razorpay test mode: real order, real declined
payment, real webhook over a public tunnel, a decision the system made
unassisted, a real payment link it created, a real payment, and the recovery
attributed back to the case it started from.

```
1  case.opened      webhook     real event TThLAjhifL6XRB
2  case.diagnosed   classifier  hard_decline, confidence 1.0, deterministic
3  action.claimed   allocator
4  action.executed  executor    payment link plink_TThLhBN3L9LPK3
5  case.recovered   webhook     <- matched by notes.recoup_case_id
```

Evidence kept in `artifacts/real_loop_evidence.json`.

**The blocker found before starting.** Attribution matched only on `order_ref`. I
probed the API rather than assuming, and `payment_link.create` returns
`order_id: None` -- a payment link creates its *own* order, lazily, at payment
time. So a link-driven recovery arrives on an order the case has never seen, and
the recovery this system exists to cause would have been logged as an unrelated
payment. Attribution is now three strategies, most specific first, and the ledger
records which one matched.

**The near-miss.** Ten minutes before the payment I checked when the API process
had started: **12:57**. I had made the attribution fix at **16:32**. The running
server was executing stale code and would have matched on `order_id` only. The
payment would have succeeded, the webhook would have arrived, the signature would
have verified, and the recovery would simply never have been attributed. No
error, no failing test -- a number that never appears.

Caught by comparing process start time against file mtime, which is not a thing
any test does. Restarted, verified the loaded module actually contained the new
function, then paid.

**Two bugs in my own harness, found by running it.** The script polled the ten
most recent payments and took the first failed one -- which was a payment from
*the previous day*, so it then waited forever for a case that had long since been
truncated, looking entirely correct while doing so. Now filtered to payments
created after the run starts. And the instructions told the user to force a
failure with a short OTP; the checkout validates OTP length in the browser and
rejects it before anything reaches Razorpay, so no payment is attempted and no
webhook fires. A rejected form is not a failed payment.

**The pattern, again.** Stale code that runs. A payment from yesterday that looks
like today's. A form rejection that looks like a failure. None of them error.
Every one produces a plausible outcome that means something other than what it
appears to. That is now nine days of the same failure mode, and the only defence
that has worked is checking what a thing *is* rather than what it looks like.

---

### The second real run claimed a recovery it had nothing to do with

Re-ran the real loop to prove the *script* worked, since the first run had been
interrupted twice by my own fixes. The second run reported `RECOVERED` in 59
seconds. It was wrong, and the way it was wrong is the worst thing this project
has produced.

**What actually happened:**

```
1  case.opened                webhook
2  case.diagnosed             classifier   hard_decline
3  action.claimed             allocator
4  action.failed              executor     <- the payment link was never created
5  case.recovered             webhook      <- matched by order_id
```

The payment link creation failed. The case was marked `ACTED` regardless. The
customer then paid through the *original* checkout link, `resolve_recovery` saw
an `ACTED` case and credited the recovery to Recoup.

**The system claimed to have recovered money it had done nothing about.**

Every guard in this codebase exists to prevent precisely that. The counterfactual
attribution, the self-recovery distinction, the unnecessary-contact metric, the
whole argument that a naive recovery dashboard overstates impact by 2x -- all of
it undone by one unconditional line in the live path: `_execute` set
`case.status = ACTED` whether or not the action had happened.

Fixed: `_execute` returns whether the action was performed, and the case only
advances if it was. A failure now writes `case.action_did_not_execute` and leaves
the case unworked, so a later payment on it is correctly a self-recovery. Three
tests pin it, including the end-to-end version.

The database record has been corrected -- run 2 is now `closed_unrecovered`,
which is the truth.

**Why the link failed, and why run 1 had worked.** Razorpay caps `reference_id`
at 40 characters. I had added `f"recoup_{case.id}"` -- seven characters plus a
36-character UUID, so 43. Every live link creation failed with a
`BadRequestError`.

Run 1 succeeded only because the server was still executing older code that never
sent the field. So the fix I made to *improve* attribution silently broke link
creation, and the run that proved the loop worked was proving it against code
that no longer existed on disk.

**What I take from this.** I have twice now been caught by a running process
whose code did not match the repository. The first time it hid a fix; this time
it hid a break. Restarting the API is now part of the loop rather than something
I remember to do, and the acceptance check reads the loaded module rather than
the file.

The deeper point is about the result itself. A green `RECOVERED` in 59 seconds
looked like the best evidence in the project. It was a false positive produced by
two bugs cancelling into a plausible outcome, and the only reason it was caught is
that I read the ledger instead of the status field. **A summary can be wrong in
ways the trail it summarises cannot.**

---

### Third run: genuine

Ran the loop a third time on fixed code. It completed start to finish with no
intervention, and this time it survives cross-checking:

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| action | executed | **failed** | executed |
| razorpay link | created | **none** | `plink_TTiTQCnaqNT3HK` |
| matched by | notes | **order_id** | notes |
| server code | **stale** | current | current |
| verdict | accidental | **false positive** | genuine |

Razorpay's own records agree: the link reports status `paid`, carries the
`reference_id` Recoup generated, and names the capturing payment.

Three attempts to demonstrate one thing, and the two failures were worth more
than the success. The first passed for the wrong reason; the second passed while
being false. Only the third means anything, and it only means anything because
the first two were checked properly rather than accepted.

`make real-loop` is now a claim a judge can run.

---

## 2026-08-25 — Day 10

### The obvious robustness sweep would have been reassuring and meaningless

Built the two sweeps. The first design for the "degraded oracle" sweep was to
corrupt the *uplift estimate* and watch the allocator degrade -- which is what
the plan called for, and it is the wrong experiment.

The measurements already say ranking is worth about 1%. Degrading the thing that
drives ranking can therefore only move the result by about 1%, and the sweep
would have produced a flat line that looked like robustness and demonstrated
nothing.

The system's actual claim rests on **action selection from the cause**. So the
question worth pricing is: what if the cause is wrong? Razorpay's codes are
reliable, but an integration can mis-map them and the day-4 measurement caught a
model assigning a fraud block to a retryable decline at 0.95 confidence.

**First version of that sweep was also too weak.** Measured against fixed retry,
the system still won by +478% at 50% classification error, which is true and
uninformative: a wrong cause usually still produces a *contact*, and contacting
beats retrying for most causes. The comparison was measuring "does it retry
blindly", not "does it know the cause".

Added a second, harder baseline -- always send a payment link, cause-blind but
not stupid. Against that, knowing the cause is worth **+36%** at zero error,
decaying to **+19%** at 50%.

That decomposition is more honest than the headline it replaces. The +512% splits
into two claims: **not blindly retrying** is most of it, and **knowing the cause**
adds roughly a third on top. Both are real; only the second is the interesting
one, and it is the one that degrades.

### A test that failed for being wrong about the code

`test_every_section_is_present` asserted the harness produced sections containing
the words "lever" and "ceiling". The sections are titled "Which decision recovers
the money" and "How much was left to take". The test was wrong; the harness was
right.

Rewrote it to assert on phrases the titles actually use, and — more usefully —
with a comment on each explaining which claim in README or ARCHITECTURE would
become unsupported if that section disappeared. A test that pins a name is
brittle; one that pins a dependency is worth keeping.

---

### Day 10 audit: a paired test that was not pairing anything

**1. The paired bootstrap compared unrelated cases.**

One arm -- the idealised pooled-and-ranked baseline -- is evaluated on a
value-sorted copy of the frame, so its rows are in a different order from every
other arm's. `paired_bootstrap` subtracted row *i* of one from row *i* of the
other. It ran, returned Rs 983 with a 95% interval, and marked the result
significant.

What it corrupted is subtler than it first looked. The point estimate was
*correct*: `mean(a - b) = mean(a) - mean(b)` whatever the pairing. What broke was
the variance, and therefore every interval and every significance verdict:

| | mean | std | 95% CI | width |
|---|---|---|---|---|
| unaligned | 983 | 9,692 | [586, 1,365] | 780 |
| aligned | 983 | 6,657 | [729, 1,268] | 539 |

The interval was **1.45x too wide**. Common random numbers exist precisely to
shrink that variance; pairing the wrong rows discarded the entire benefit while
still calling it a paired test. It erred toward *under*-claiming, which is the
better direction to be wrong in and still wrong.

Fixed structurally rather than locally: `ArmResult` now carries the case ids
belonging to each entry, `paired_bootstrap` aligns on them, and it raises rather
than guesses if two arms were evaluated on different case sets.

**2. The README and the evaluation described different experiments.**

The README quoted a lever study run at a 15% contact budget; `make eval` used
25%. Both were internally correct and they disagreed -- +524% against +512%, and
different rupee figures throughout. Neither document was wrong on its own terms,
which is what made it survive a claim-verification pass that only checked whether
each number was *reproducible*, not whether the two documents were describing the
same thing.

`BUDGET_FRACTION` is now defined once and imported. Two tests enforce it: one
fails if it is ever defined in more than one place, another fails on any
hardcoded `len(x) * 0.<n>` outside that definition. Four such literals were still
scattered across `train.py` and `test_model.py`.

**3. A test that asserted the wrong thing about its own subject.**

`test_every_section_is_present` looked for sections containing the words "lever"
and "ceiling"; the sections are titled *"Which decision recovers the money"* and
*"How much was left to take"*. The test failed, the harness was right. Rewritten
to assert on the phrases actually used, with a note on each explaining which
README or ARCHITECTURE claim goes unsupported if that section disappears -- a test
that pins a name is brittle, one that pins a dependency is worth having.

### The shape, tenth day running

The pairing bug produced a number of the right type, in a plausible range, marked
significant, that was not measuring what its name said. So did the budget drift:
two documents, both reproducible, both accurate, describing different experiments.

Neither would ever fail. Both were found by asking what the number would look like
if it were wrong, and whether that was distinguishable from what was on screen.

---

## 2026-08-26 — Day 11

Building the console meant putting the live path on a screen, which turned out
to be a different kind of test. Reading the ledger back as prose surfaced two
defects that every existing test agreed were fine.

---

### The system computed a send time, recorded it, and then ignored it

**Expected:** the evidence timeline would read as a clean sequence — failure,
diagnosis, decision, send, recovery.

**Actually:** step three said the deciding rule was `quiet_hours`, and step four
said the payment link was created 0.6 seconds later. The recovery in
`docs/evidence/` happened at **00:36 IST**, which is inside the quiet window the
rule exists to enforce.

**Why:** `send_at` was a local variable in `allocator/policy.py`. It adjusted
`Decision.delay_h` and never left the function. `Decision` has no send time, and
`_execute` in `live_allocator.py` called `send_payment_link` immediately.

The shape of it is worse than a missed check. `_execute` **did** honour
`delay_h` — for `RETRY`, which it passed to `schedule_retry`. And the allocator
only ever applies the quiet-hours shift to actions that contact a customer;
retries return earlier, unrationed. So the delay was honoured on exactly the
actions that touch a gateway, and dropped on exactly the ones that touch a
person. A rule enforced with perfect precision on everything it was not for.

Nobody was messaged, but only because link notifications are disabled
everywhere in this codebase. That is a second guard doing the first guard's job,
and it is not something to be reassured by.

**The larger half.** The same discarded value carries the **per-cause delay** —
48 hours for insufficient funds so the retry lands after payday, 26 for a spent
limit, 6 for a hard decline. Timing is half of the result this project leads
with: *cause-aware action **and timing**, +512%*. The evaluation models it. The
deployed path sent everything immediately. The system being measured and the
system that runs were not the same system, and the headline number is the one
that named the difference.

**Fix:** one function, `contact_due_at`, used by both the executor and the tick,
computing the send time from **detection** rather than from now — "48 hours
after the payment failed" is what the evaluation scores, and a case picked up
three days late is already due rather than owed another 48 hours. Contacts that
are not due are recorded with `scheduled_for`, a `case.action_deferred` ledger
entry, and a third execution outcome (`DEFERRED`) that is neither executed nor
failed. `/tasks/execute-due` sends them when the time comes, re-checking quiet
hours at that moment rather than trusting the earlier arithmetic.

**What it cost.** A hard decline now waits six hours, so the 48-second real-loop
recording is no longer reproducible live. Rather than weaken the rule, there is
an explicit operator override that sends ahead of the cause-implied delay and
writes `case.schedule_overridden` to the ledger as a human decision. It does
**not** bypass quiet hours: the delay is an optimisation worth a little expected
value, and being asleep is not.

**Why no test caught it.** Every test ingested a webhook stamped `now` and then
asserted on what happened. A contact for a case detected a millisecond ago is
correctly not due — so the tests were, unknowingly, all asserting the deferral
was absent. Six of them failed on the fix, and each one was right about its own
subject and wrong about this. They now pin the clock and age the case
deliberately, because the live path genuinely behaves differently at 23:00 IST
than at 14:00, and a test that reads the wall clock passes all afternoon.

---

### Four smaller ones, all found by rendering

- **`Decimal` reached the browser as a string.** `SUM` over a `BIGINT` comes back
  from psycopg as `Decimal`, which FastAPI serialises as `"2385375.3"`. Every
  arithmetic operation on it in JavaScript would have been string concatenation.
- **The approval queue counted the wrong thing.** A case whose action is awaiting
  a human keeps status `DIAGNOSED` — the queue is a property of the *action*. The
  overview reported an empty queue while three sat in it.
- **A `GROUP BY` on a JSONB expression.** SQLAlchemy emits a fresh bind parameter
  per occurrence, so Postgres saw two different expressions and refused.
- **The bar chart had never worked.** `.fill` is a `<span>` inside a plain `div`,
  so it stayed inline and dropped `width` and `height` silently. Every bar
  rendered at zero and read as a deliberately minimal style. The tell was
  `getComputedStyle` returning `"100%"` where a laid-out element returns pixels.

Also: quiet hours **defer** a contact, they do not cancel one. The first draft of
the console counted them under "blocked", which would have overstated the panel
in our own favour. Refusals are now reported as three separate outcomes — never
planned, planned then declined, postponed — because collapsing them is the easy
read and the wrong one.

---

**The pattern, on the eleventh day.** Both real defects were values that were
correct at the moment they were computed and wrong by the time they were used.
The ledger said `quiet_hours`. The decision said `delay_h`. Both were right.
Nothing compared either to what the system actually did, and no test can catch a
number that is only wrong in a place it was never carried to.

What found them was rendering the ledger as sentences and reading it. "Chose
method switch prompt at 00:36" and "sent at 00:36" are both unremarkable alone.
Next to each other they are a bug.
