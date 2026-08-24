# Recoup

**Razorpay's error taxonomy already knows the right recovery action. Almost nobody uses it.**

Using it is worth roughly **5×** against a fixed retry schedule — measured, on
held-out data, in all three world parameterisations. This repository is that
finding, the system that acts on it, and the governance layer that makes
automating it safe.

*Simulation benchmark throughout. Not production Razorpay data.*

---

## The finding

When a payment fails, Razorpay returns a structured error:

```json
{ "error_reason": "card_expired", "error_source": "customer",
  "error_step": "payment_initiation" }
```

That is not just a reason. It is an instruction. An expired card responds to a
**method-switch prompt** about twenty times better than to a retry. A bank
outage resolves on its own — retry, and do not message the customer about their
bank's downtime. A merchant-configuration failure can never be fixed by anything
the customer does, so every customer-facing action has expected value zero.

Recovery systems mostly ignore this. Razorpay's own subscription retry is a
fixed T+3 schedule — next day, three times, regardless of cause, issuer or
customer. Same schedule for an authentication failure, where the customer is
still at the checkout and the window is *minutes*, as for insufficient funds,
where 24 hours is too **early** to help.

Holding case selection and budget fixed, and varying only the action and its
timing:

| action policy | recovered | lift |
|---|---|---|
| fixed retry @ 24h (T+3 style) | ₹6.1L | — |
| fixed payment link @ 24h | ₹30.2L | +394% |
| retry at the cause's best moment | ₹20.3L | +231% |
| **cause-aware action + timing** | **₹38.2L** | **+524%** |

Holds at +519% to +529% across pessimistic, base and optimistic worlds.

## What is *not* worth much, and is reported anyway

This project set out to prove that ranking cases by expected value beats ranking
by transaction amount. ₹10,000 at 90% beats ₹50,000 at 5%. It is intuitive, and
measured against an **oracle that knows the true uplift** it is worth **+1%**.

`corr(EV, amount) = 0.94`. Amounts span ~5,000×, uplift ~5×, so the product
barely reorders. That is arithmetic, not a modelling failure — no better model
rescues it.

Two further negative results, kept because they are true:

- **A per-case gradient-boosted uplift model is 24% worse than no estimate at
  all** in the allocator. A cause-rate group-by — about twenty numbers,
  computable in SQL — tracks true uplift better (+0.362) than the fitted model
  (+0.275). The structure here is per-cause; a per-case model adds variance
  without adding signal.
- **The allocator's distinctive machinery is worth ~0** on revenue (+1.0% vs
  siloed per-channel agents, −1.3% vs an idealised pooled one). It is documented
  as a governance layer, which is what it is.

## Architecture

```
Razorpay webhook ──▶ deterministic classifier (26 published codes, 0 model calls)
                          │
                          ├── unmapped code ──▶ deferred queue ──▶ LLM tail
                          │                                          │
                          │                            risk guardrail (not overridable)
                          ▼                                          ▼
                    canonical cause  ───────────────────────▶  exception list
                          │
                          ▼
              TAXONOMY selects the action        ◀── +520%, deterministic, no model
                          │
                          ▼
              value-ordered under a budget       ◀── +18.5% over arrival order
                          │
                          ▼
              governance: contact caps, quiet hours,      ◀── ~0% revenue,
              suppression, append-only audit trail            hard guarantees
                          │
                          ▼
              Razorpay test-mode execution / outcome simulator
```

### Where AI is used, and where it is deliberately not

Three times this project measured a learned component against a deterministic
one and the deterministic one won. That is the actual finding, not an accident:
**in payments recovery the structure lives in a published taxonomy, not in the
outcome data.**

| decision | how | why |
|---|---|---|
| Classify a published error code | lookup table | A closed, documented enum. A model would be slower, cost money, and add non-determinism to a money path for zero accuracy gain. |
| Classify an *unpublished* code | LLM, schema-constrained | Razorpay ships new codes; an integration that breaks on one is a bad integration. |
| Anything with a fraud signal | deterministic guardrail | The tail model classified a fraud block as a retryable decline at **0.95 confidence**, reproducibly. Escalating a hard decline costs one review; retrying a fraud block costs more. |
| Choose the recovery action | taxonomy | The model picked the true-best action on **3 of 11** causes; the taxonomy gets **9 of 11**. |
| Estimate magnitude | cause-rate group-by | Beat a GBM on correlation with true uplift. |
| Caps, stopping rules, budgets | pure code | A cap a model can be talked out of is not a cap. |

The LLM tail runs on a local model (Ollama, `qwen2.5:7b`) with Claude as the
documented production path — `ANTHROPIC_API_KEY` switches it automatically. Model
choice was decided by a held-out evaluation, not preference: a 3B model scored
4/7 with **three confidently-wrong answers above the acting threshold**.

## It works on the real thing

One payment recovered end to end on Razorpay test mode — real order, real
declined payment, real webhook, a decision the system made unassisted, a real
payment link it created, real payment, recovery attributed:

```
1  case.opened      webhook     real event TThLAjhifL6XRB
2  case.diagnosed   classifier  hard_decline, confidence 1.0, deterministic
3  action.claimed   allocator
4  action.executed  executor    payment link plink_TThLhBN3L9LPK3
5  case.recovered   webhook     <- matched by notes.recoup_case_id
```

It chose a **method switch, not a retry** — the issuer had declined the card, and
retrying the same instrument gets declined again. Razorpay's own error taxonomy
already said so.

Unedited trail in [`docs/evidence/`](docs/evidence/). Reproduce with
`make real-loop`.

## Evaluation

- **12,004 cases**, 90 days, 3 loss channels, chronological train/calibration/test
  split — never random, so a customer's later behaviour cannot inform a
  prediction about their earlier failure.
- **Common random numbers**: every arm faces the same customers with the same
  latent willingness, so a difference between arms is the policy, not the dice.
- **Counterfactual attribution**: separates money we *caused* from money that was
  arriving anyway. Contacting everyone recovers 44% but *causes* 22% — a naive
  recovery dashboard overstates impact by ~2×.
- **Bayes ceiling**: the outcome is a coin flip weighted by *p*, so no model can
  exceed ROC-AUC 0.637 here. Ours reaches 0.620 — **88% of the achievable
  signal**. A raw 0.62 reads as weak without that.
- **Two robustness sweeps**: a degraded probability oracle, and three world
  parameterisations. Every headline holds in all three.
- **Paired bootstrap** with 95% intervals on every comparison.

Every number is reproducible: `make eval` twice gives byte-identical output.

## Quickstart

Python 3.13+, Docker, Node 20+.

```bash
cp .env.example .env      # Razorpay TEST keys; live keys are refused
make install
make db
make data-worlds          # generate all three worlds
make model                # train, calibrate, run the lever study
make allocate             # the equal-budget bake-off
```

```bash
make test          # unit, no database
make test-all      # + integration (needs make db and make ollama)
make lint
```

### Receiving real webhooks

Recoup runs against Razorpay **test mode** and refuses to start with a
`rzp_live_` key.

```bash
cloudflared tunnel --url http://localhost:8000
```

Point a Razorpay webhook at `https://<host>/webhooks/razorpay`, subscribe to
`payment.failed`, `payment.captured`, `subscription.pending` and
`subscription.halted`, and set the same secret as `.env`.

The LLM tail never runs inside the webhook request — a cold model call measured
31 seconds, and Razorpay would time out, retry, and hit our own idempotency
guard, leaving the case permanently unclassified. Unknown codes are parked and
drained by `POST /tasks/classify-pending`. Webhook latency is ~10ms.

## Engineering notes

**Money is integer paise, never a float.** Matches Razorpay's API and avoids
fractional drift across a batch.

**The ledger is append-only.** *"Why was this case contacted twice?"* must be
answerable from `case_events` alone, without trusting mutable state.

**Idempotency is a database constraint, not application logic.** Every
money-touching action carries a unique `(case_id, action_type, attempt_no)`.
Razorpay redelivers webhooks; a check-then-act in Python races under concurrent
retries.

**"Zero policy violations" is true by construction and proves nothing.** The
number that shows a gate is load-bearing is how often it *refused*.

## What this cannot tell you

- Recovery probabilities are **estimates**, labelled as such in
  [`data/ASSUMPTIONS.md`](data/ASSUMPTIONS.md). Nobody publishes conditional
  recovery rates by failure cause; that is exactly the proprietary knowledge a
  payments company accumulates.
- The simulator **cannot validate the model** — the generating process is known.
  It validates the *allocator*, which is why the oracle is degraded deliberately
  rather than trusted.
- Multi-touch is probably modelled **optimistically**: repeated contacts draw
  independently, coupled only by fatigue.
- Customers are sampled **uniformly**, so only 17% have multiple cases in 90
  days. Real merchant traffic is heavy-tailed, which would make the collision
  guard bind harder. Left unchanged deliberately — altering the data after seeing
  the result is not a move worth making.

[`INCIDENTS.md`](INCIDENTS.md) is the log of everything that broke, including
seven adversarial reviews that each found real defects — among them a policy
reading its own answer key, an evaluation that was not reproducible while
claiming to be, and a baseline that was rigged in my own favour.

## Layout

```
backend/app/
  api/          webhook receiver, deferred-task endpoints
  services/     classifier, LLM providers, ingest adapters, ledger
  simulation/   generator, outcome simulator, baseline policies, arm harness
  model/        features, uplift model, calibration, lever study
  allocator/    taxonomy action selection, budget, governance, bake-off
  taxonomy.py   canonical causes mapped from Razorpay's published error codes
```

## Licence

MIT
