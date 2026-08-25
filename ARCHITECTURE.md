# Architecture

How Recoup is put together, and — more usefully — why several obvious choices
were rejected after being measured.

The short version: **the structure in payments recovery lives in a published
taxonomy, not in the outcome data.** Three separate times a deterministic rule
beat the learned component it was competing with, and each time the rule won on
evidence rather than preference. That is the through-line of every decision below.

---

## The path a failure takes

```
Razorpay webhook
      │
      ▼
signature check ──── invalid ──▶ 400, nothing recorded
      │
      ▼
delivery claim ───── duplicate ──▶ 200, no-op
      │                            (unique constraint, not a SELECT)
      ▼
loss-channel adapter          failed payment · abandoned checkout · subscription
      │
      ▼
deterministic classifier      26 published Razorpay codes, 0 model calls, ~0.004 ms
      │
      ├── unrecognised code ──▶ PENDING_DIAGNOSIS ──▶ /tasks/classify-pending
      │                                                       │
      │                                              LLM tail (schema-constrained)
      │                                                       │
      │                                              risk guardrail (not overridable)
      ▼                                                       ▼
canonical cause  ◀────────────────────────────────────── or exception list
      │
      ▼
TAXONOMY selects the action           +512%   deterministic, no model
      │
      ▼
value-ordered under a shared budget   +17%    vs working a queue in arrival order
      │
      ▼
policy gates                          caps · quiet hours · stopping rules ·
      │                               approval queue · suppression
      ▼
idempotent execution                  unique (case_id, action_type, attempt_no)
      │
      ▼
Razorpay test mode  ·  or the outcome simulator
      │
      ▼
append-only audit ledger              every decision, and why
```

---

## Where a model is used, and where one was refused

This is the section worth reading. Each row was decided by a measurement, and the
measurement is named.

| Decision | Mechanism | Why not a model |
|---|---|---|
| Classify a **published** error code | lookup table | A closed, documented enum. A model is slower, costs money, and adds non-determinism to a money path for **zero** accuracy gain — the lookup is already exact. Round-trips at 100% on 12,004 generated cases. |
| Classify an **unpublished** code | LLM, schema-constrained | Razorpay ships new codes; an integration that breaks on one is a bad integration. ~1% of traffic, cached per code so the cost is per *code*, not per payment. |
| Anything carrying a **fraud signal** | deterministic guardrail, applied after the model and not overridable | The tail model classified a fraud block as a retryable decline at **0.95 confidence**, reproducibly. `risk_blocked` and `hard_decline` have opposite semantics. Escalating a hard decline costs one review; retrying a fraud block costs more. |
| Choose the **recovery action** | taxonomy | The fitted uplift model picked the true-best action on **3 of 11** causes — it always chose a payment link, which is right *on average* and useless per-case. The taxonomy gets **9 of 11**. |
| Estimate **magnitude** | cause-rate group-by | Tracks true uplift better (**+0.362**) than the per-case gradient-boosted model (**+0.275**). In the allocator the ML model is **24% worse than no estimate at all**. |
| Caps, stopping rules, budgets | pure code | A cap a model can be talked out of is not a cap. |
| **Forecasting** | the calibrated model | The one place it earns its place. An over-confident model claims ₹49L and delivers ₹26L at identical ROC-AUC, and a merchant staffs against the forecast. |

The machine learning is therefore **not in the decision path**. That was not the
plan; it is what the measurements produced, and reversing it would mean ignoring
them.

---

## Decisions that cost something

### Money is integer paise, never a float

Matches Razorpay's own API and avoids fractional drift accumulating across a
50,000-row batch.

### The ledger is append-only

`case_events` is never updated or deleted. *"Why was this case contacted twice?"*
has to be answerable from the ledger alone, without trusting mutable state
elsewhere.

This is not decoration. A run of the live loop once reported `RECOVERED` while
the payment link had silently failed to create — the status field said one thing
and the trail said `action.failed`. **A summary can be wrong in ways the trail it
summarises cannot.**

### Idempotency is a database constraint, not application logic

Every money-touching action carries a unique `(case_id, action_type, attempt_no)`.
Razorpay redelivers webhooks; a check-then-act in Python races under concurrent
retries, and the race is invisible until it sends a customer two payment links.

### The LLM never runs inside a webhook request

A cold local-model call measured **31 seconds**. Razorpay retries a slow webhook,
our idempotency correctly rejects the retry, and the case is *never classified* —
silent data loss produced by two individually-correct mechanisms. Unknown codes
are parked and drained by a tick endpoint. Webhook latency is ~10 ms.

### A tick endpoint, not a queue broker

The work is a bounded scan over rows with a status. The state belongs in Postgres
next to the audit trail, and a tick you can run by hand with `curl` is one you can
reason about at 2am. A broker would add a moving part without removing one.

### Failure is recorded, never raised

A gateway outage, a missing API key, a model returning nonsense — all record and
continue. One misconfigured deployment must not take down a batch of two thousand
cases.

---

## The counterfactual boundary

The simulator knows things no production system could: whether a customer would
have paid anyway, and how recoverable each case truly was. Those fields are
prefixed `latent_` and **no policy may read them**.

It is enforced structurally rather than by care: a test walks the AST of the
policy module and fails if any class not explicitly marked `reads_ground_truth`
contains a `latent_*` literal. It was added after a policy was found reading its
own answer key — which would have justified discarding every number in the
evaluation, not just that arm's.

The one exemption is the oracle, which exists to define the ceiling and is never
ranked against the arms.

---

## What the evaluation can and cannot support

**It can** test whether the allocator makes better decisions than the baselines,
given a probability oracle of a stated quality, and whether those conclusions
survive a different world and a worse classifier.

**It cannot** demonstrate that the recovery model is accurate. The generating
process is known, so a model trained and scored on it measures gradient descent,
not payments. That circularity is addressed by attacking the conclusions two
ways — a wrong world and a wrong cause — rather than by pretending the simulator
validates the model.

The **one** measurement here that is not simulated is the real Razorpay loop in
[`docs/evidence/`](docs/evidence/).

---

## Layout

```
backend/app/
  api/            webhook receiver, task endpoints, approval queue
  services/       classifier, LLM providers, ingest adapters, actions, ledger
  simulation/     generator, outcome simulator, baseline policies, arm harness
  model/          features, uplift model, calibration, the lever study
  allocator/      taxonomy action selection, budget, governance, bake-off
  evaluation/     the full harness and both robustness sweeps
  taxonomy.py     canonical causes, mapped from Razorpay's published error codes
```

Run `make eval` to regenerate every number in [`EVALUATION.md`](EVALUATION.md).
Two runs produce byte-identical output.
