# Declared assumptions

Every parameter behind the synthetic dataset, with an honest label for where it
came from. **Generated from `backend/app/simulation/assumptions.py` — do not edit
by hand.** Regenerate with `make assumptions`.

| Basis | Meaning |
|---|---|
| `sourced` | Taken from published figures (Razorpay documentation, NPCI/UPI reporting). |
| `anchored` | Derived from a sourced figure plus a stated inference. |
| `estimate` | My judgement. Not sourced. Sensitivity-tested rather than trusted. |

## Why so many estimates

Nobody publishes conditional recovery probabilities by failure cause, because
that is exactly the proprietary knowledge a payments company accumulates from
production traffic. Pretending otherwise would be the dishonest move.

The response is not to dress estimates up as facts. It is to label them, hold
them fixed and reproducible, and re-run the entire evaluation under different
parameterisations to see which conclusions survive. For the recoverability
parameters in particular, the load-bearing claim is their **relative ordering** —
expired collect requests recover far more readily than hard declines — not their
absolute values. No plausible parameterisation reverses that ordering.

## What this dataset can and cannot support

**It can** test whether the allocator makes better decisions than the baselines,
given a probability oracle of a stated quality.

**It cannot** demonstrate that the recovery model is accurate. The generating
process is known, so a model trained and scored here measures gradient descent,
not payments. That circularity is addressed by degrading the oracle and varying
the world — not by pretending the simulator validates the model.


## Rates and volumes

| Parameter | Value | Basis | Note |
|---|---|---|---|
| `BLENDED_SUCCESS_RATE` | 0.935 | `sourced` | Published Indian merchant blended success rates cluster at 92-96%. |
| `TECHNICAL_DECLINE_RATE` | 0.008 | `sourced` | UPI technical decline rate ~0.8% (NPCI reporting, 2024-25). |
| `CUSTOMER_SIDE_SHARE` | 0.55 | `sourced` | Razorpay: '>50% payment failures are due to customer errors or network issues'. |
| `PEAK_HOUR_SUCCESS_MULTIPLIER` | 0.9 | `anchored` | Derived from reported peak dips to 80-85% against a 93.5% base. |
| `CONTACT_FATIGUE_LAMBDA` | 0.55 | `estimate` | Each prior contact scales response probability by 0.55. Swept in world sweep. |
| `BRAND_ABANDONMENT_AFTER_FAILURE` | 0.6 | `sourced` | ~60% of customers abandon a brand after a failed payment experience. |
| `PUBLISHED_RECOVERY_CEILING` | 0.2 | `sourced` | Razorpay Failed Payment Recovery: 'recover up to 20% of failed payments'. |

Peak hours (IST): 12, 13, 19, 20, 21


## Payment method mix

| Parameter | Value | Basis | Note |
|---|---|---|---|
| `METHOD_MIX[upi]` | 62% | `estimate` | UPI-dominant mix typical of Indian D2C. |
| `METHOD_MIX[card]` | 22% | `estimate` | Skews to higher ticket sizes. |
| `METHOD_MIX[netbanking]` | 9% | `estimate` | Higher ticket, lower volume. |
| `METHOD_MIX[wallet]` | 7% | `estimate` | Small-ticket tail. |

## Cause mix by method

Anchored on two sourced constraints — customer-side causes carry ~55% of the mass, and bank/gateway technical declines are ~1% — with the split *within* those bands estimated. The profile genuinely differs by method: UPI fails at collect and cancellation, cards fail at authentication and decline.

| Cause | upi | card | netbanking | wallet |
|---|---|---|---|---|
| `account_mismatch` | 1% | — | — | — |
| `authentication_failed` | 10% | 27% | 18% | 12% |
| `card_blocked` | — | 2% | — | — |
| `card_disabled_online` | — | 10% | — | — |
| `card_expired` | — | 5% | — | — |
| `collect_expired` | 26% | — | — | — |
| `customer_abandoned` | 22% | 8% | 25% | 24% |
| `hard_decline` | 5% | 18% | 9% | 7% |
| `insufficient_funds` | 16% | 13% | 15% | 38% |
| `invalid_instrument` | 7% | — | — | — |
| `limit_exceeded` | 3% | 3% | — | — |
| `merchant_config` | — | 4% | — | — |
| `risk_blocked` | 1% | 3% | 3% | 3% |
| `transient_bank_downtime` | 9% | 7% | 30% | 16% |

## Latent recoverability (counterfactual ground truth)

These are the parameters the agent never observes. `p_self_recover` is the denominator that makes *unnecessary contact* measurable: contacting someone who would have paid regardless is a cost with no matching benefit, and it is invisible to any metric that only counts recoveries.

| Cause | p_self_recover | p_retry | p_nudge | best delay (h) | Note |
|---|---|---|---|---|---|
| `collect_expired` | 0.18 | 0.55 | 0.48 | 0.5 | Customer often never saw the request; re-sending converts well. |
| `authentication_failed` | 0.34 | 0.62 | 0.40 | 0.05 | Highest intent, fastest decay. Many retry unaided within minutes. |
| `customer_abandoned` | 0.22 | 0.30 | 0.34 | 2 | Genuine intent and a change of mind are indistinguishable here. |
| `insufficient_funds` | 0.16 | 0.28 | 0.31 | 48 | Recovery is gated on the balance changing, not on persuasion. |
| `transient_bank_downtime` | 0.30 | 0.68 | 0.10 | 3 | Resolves on its own once the issuer recovers. Retry, do not contact. |
| `limit_exceeded` | 0.14 | 0.45 | 0.22 | 26 | Daily limits reset; a next-day retry is materially better. |
| `account_mismatch` | 0.12 | 0.35 | 0.44 | 1 | Needs an instruction, not a retry. |
| `invalid_instrument` | 0.08 | 0.04 | 0.30 | 1 | Same VPA fails identically; only a method switch helps. |
| `card_expired` | 0.06 | 0.02 | 0.35 | 1 | Deterministically dead on this card, fine on another. |
| `card_disabled_online` | 0.07 | 0.05 | 0.33 | 1 | Fixable, but only by a customer who is told what to fix. |
| `card_blocked` | 0.05 | 0.03 | 0.18 | 2 | Often follows a fraud report. |
| `hard_decline` | 0.09 | 0.12 | 0.16 | 6 | Issuer refused without an actionable reason. |
| `risk_blocked` | 0.01 | 0.01 | 0.01 | 0 | Treated as unrecoverable by policy, not by probability. |
| `merchant_config` | 0.02 | 0.00 | 0.00 | 0 | Zero by construction. No customer action can clear a merchant setting, so every customer-directed recovery has expected value zero and non-zero cost. |

## Recovery semantics per cause

Derived from Razorpay's published error tables. Three axes drive every downstream action.

| Cause | Who can fix | Retry policy | Contact OK |
|---|---|---|---|
| `transient_bank_downtime` | bank | delayed | **no** |
| `insufficient_funds` | customer | delayed | yes |
| `authentication_failed` | customer | immediate | yes |
| `customer_abandoned` | customer | immediate | yes |
| `collect_expired` | customer | immediate | yes |
| `invalid_instrument` | customer | different_instrument | yes |
| `card_expired` | customer | different_instrument | yes |
| `card_disabled_online` | customer | different_instrument | yes |
| `card_blocked` | bank | different_instrument | yes |
| `hard_decline` | bank | different_instrument | yes |
| `risk_blocked` | nobody | never | **no** |
| `limit_exceeded` | customer | delayed | yes |
| `merchant_config` | merchant | never | **no** |
| `account_mismatch` | customer | immediate | yes |

## Intervention costs

| Parameter | Value | Basis | Note |
|---|---|---|---|
| `ACTION_COST_PAISE[retry]` | ₹0.00 | `anchored` | Gateway retry has no per-attempt customer cost. |
| `ACTION_COST_PAISE[payment_link_sms]` | ₹0.25 | `estimate` | ~INR 0.25 per transactional SMS. |
| `ACTION_COST_PAISE[payment_link_whatsapp]` | ₹0.80 | `estimate` | ~INR 0.80 per WhatsApp utility message. |
| `ACTION_COST_PAISE[payment_link_email]` | ₹0.02 | `estimate` | Negligible per-email cost. |
| `ACTION_COST_PAISE[method_switch_prompt]` | ₹0.25 | `estimate` | Delivered over the same channels. |
| `ACTION_COST_PAISE[merchant_alert]` | ₹0.00 | `anchored` | Internal notification, no external cost. |
| `ACTION_COST_PAISE[human_review]` | ₹40.00 | `estimate` | ~INR 40 of an agent's time per case. |

## Merchant profiles

| Merchant | Category | Txns/day | Avg ticket |
|---|---|---|---|
| Daily Basket | grocery | 420 | ₹680 |
| Thread & Co | fashion | 180 | ₹2,490 |
| LearnLoop | education | 60 | ₹14,990 |
| Voyage Desk | travel | 45 | ₹32,000 |

## World parameterisations

The world sweep re-runs the full evaluation under each of these. Worlds hold the *same failures* and vary only how recoverable they are, so any difference in results is attributable to the world and nothing else.

| World | Recoverability scale | Contact fatigue λ |
|---|---|---|
| `base` | 1.00 | 0.55 |
| `pessimistic` | 0.65 | 0.40 |
| `optimistic` | 1.35 | 0.75 |

## Dataset shape

- Default seed: `42`
- Window: 90 days
- Target cases: 10,000 (minimum 8,000)
- Calibration holdout: 20% (minimum 2,000 cases)
- Test holdout: 20%, taken as the final slice of the timeline — never a random sample

