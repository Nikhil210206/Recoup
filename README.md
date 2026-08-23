# Recoup

**The revenue recovery control plane.**

*Every recovery agent asks "can I recover this?" Recoup asks "should I?"*

---

## The problem

When a payment fails, that revenue is not necessarily gone. Merchants can retry
it, send a payment link, nudge the customer, or suggest a different method.
There are already good agents that do this — Razorpay's Agent Studio ships a
Subscription Recovery Agent and an Abandoned Cart Conversion Agent, and Failed
Payment Recovery retargets dropped payers over WhatsApp, email and SMS.

What no published layer does is decide **which** of those agents should act, on
**whom**, **when**, and whether the intervention is worth more than it costs.

Each agent optimises its own queue. So:

- Three agents chasing the same customer in the same week is a spam problem the
  customer experiences and nobody owns.
- Ranking by transaction value sends effort to the biggest amounts rather than
  the most recoverable ones. ₹10,000 at 90% is worth more than ₹50,000 at 5%.
- Contacting someone who would have paid anyway is a pure loss, and it does not
  appear in any recovery metric — it looks like a success.

Recoup is the layer above the fleet: a calibrated, budget-aware allocator that
turns *"recover everything"* into *"recover what's worth recovering, once."*

## Status

Early. Built for the Razorpay AI Buildathon, Track 03 (AI Revenue Recovery).

| | |
|---|---|
| Detection + case store | in progress |
| Append-only audit ledger | in progress |
| Root-cause classifier | not started |
| Recovery model + calibration | not started |
| Allocator | not started |
| Policy engine | not started |
| Evaluation harness | not started |

## Quickstart

Requires Python 3.13+, Docker, and Node 20+.

```bash
cp .env.example .env     # fill in Razorpay test keys
make install             # venv + backend deps
make db                  # Postgres on localhost:5434
make api                 # http://localhost:8000/docs
```

```bash
make test        # unit tests, no database required
make test-all    # adds integration tests (needs make db)
make lint
```

### Receiving real webhooks

Recoup is built against Razorpay **test mode** and refuses to start with a
`rzp_live_` key. To take real `payment.failed` events locally:

```bash
ngrok http 8000
```

Then in the Razorpay Dashboard → Settings → Webhooks, point a webhook at
`https://<your-ngrok-host>/webhooks/razorpay`, subscribe to `payment.failed`
and `payment.captured`, and set the same secret you put in `.env`.

## Design notes

Three rules hold throughout, each for a reason that shows up later in the
evaluation:

**Money is integer paise, never a float.** Matching Razorpay's own API, and
avoiding fractional drift across a large batch.

**The ledger is append-only.** `case_events` is never updated or deleted. The
question *"why was this case contacted twice?"* has to be answerable from the
ledger alone, without trusting mutable state elsewhere.

**Idempotency is enforced by the database, not by application logic.** Every
money-touching action carries a unique key of `(case_id, action_type,
attempt_no)`. Razorpay redelivers webhooks; a check-then-act in Python races
under concurrent delivery, so the unique constraint is what actually guarantees
we act once.

## Layout

```
backend/
  app/
    api/          FastAPI routers (webhooks, health)
    services/     ledger, ingest adapters, signature verification
    models.py     case store, append-only ledger, actions
  tests/
frontend/         dashboard (not started)
```

## Licence

MIT
