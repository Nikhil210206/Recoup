"""Calibration metrics and reliability curves.

## Why this file is the centre of the project rather than a detail

The allocator ranks by expected value: `amount x uplift - cost`. That is a
**price**, not a ranking. Two models can order cases identically -- same ROC-AUC,
same PR-AUC -- while one says a case recovers with probability 0.8 and the other
says 0.3. Ranking metrics cannot tell them apart. Expected value can, and gets it
wrong for the miscalibrated one on every case simultaneously.

The failure is worse than it sounds, because it is not random. A model that
systematically overstates recovery probability produces expected values that
clear the intervention cost when they should not, so the allocator spends its
budget on cases that were never worth working. Nothing errors, the ranking looks
sensible, and the money goes to the wrong places.

So the headline model metrics here are **Brier score and expected calibration
error**, not AUC. AUC is reported because it is the number people expect, and
because the whole point is showing that it does not move while the thing that
matters does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


@dataclass
class Metrics:
    n: int
    base_rate: float
    roc_auc: float
    pr_auc: float
    brier: float
    ece: float
    log_loss: float
    #: Mean predicted probability vs observed rate. A model can be perfectly
    #: ranked and still be systematically 15 points too optimistic.
    mean_predicted: float
    mean_observed: float

    @property
    def bias(self) -> float:
        return self.mean_predicted - self.mean_observed

    def row(self, label: str) -> str:
        return (
            f"{label:26}{self.roc_auc:>8.3f}{self.pr_auc:>8.3f}"
            f"{self.brier:>9.4f}{self.ece:>8.4f}{self.bias:>+9.3f}"
        )


def expected_calibration_error(y_true, y_prob, n_bins: int = 15) -> float:
    """Mean absolute gap between predicted probability and observed frequency.

    Equal-count bins rather than equal-width: with a skewed score distribution,
    equal-width bins leave most of the range nearly empty and the resulting
    number is dominated by noise in bins holding a handful of cases.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return 0.0

    order = np.argsort(y_prob)
    bins = np.array_split(order, min(n_bins, max(1, len(order) // 20)))
    total, error = 0, 0.0
    for idx in bins:
        if len(idx) == 0:
            continue
        error += len(idx) * abs(y_prob[idx].mean() - y_true[idx].mean())
        total += len(idx)
    return float(error / total) if total else 0.0


def evaluate(y_true, y_prob) -> Metrics:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    return Metrics(
        n=len(y_true),
        base_rate=float(y_true.mean()),
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        brier=float(brier_score_loss(y_true, y_prob)),
        ece=expected_calibration_error(y_true, y_prob),
        log_loss=float(log_loss(y_true, y_prob)),
        mean_predicted=float(y_prob.mean()),
        mean_observed=float(y_true.mean()),
    )


def reliability_curve(y_true, y_prob, n_bins: int = 10):
    """(mean predicted, observed frequency, count) per equal-count bin.

    Perfect calibration is the diagonal. Points above it mean the model is
    under-confident; below, over-confident -- and over-confidence is what makes
    an allocator overspend.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    order = np.argsort(y_prob)
    bins = np.array_split(order, n_bins)
    return [
        (float(y_prob[i].mean()), float(y_true[i].mean()), int(len(i)))
        for i in bins
        if len(i) > 0
    ]


def render_reliability(y_true, y_prob, n_bins: int = 10, width: int = 44) -> str:
    """Text reliability diagram, so the curve is readable in a terminal and in
    a repository without shipping an image."""
    rows = reliability_curve(y_true, y_prob, n_bins)
    lines = [
        f"  {'predicted':>10}{'observed':>10}{'n':>7}   {'':<{width}}",
        f"  {'-' * 10}{'-' * 10}{'-' * 7}   {'-' * width}",
    ]
    for pred, obs, count in rows:
        p_col = int(round(pred * width))
        o_col = int(round(obs * width))
        bar = [" "] * (width + 1)
        lo, hi = sorted((p_col, o_col))
        for i in range(lo, hi + 1):
            bar[i] = "-"
        bar[p_col] = "P"
        bar[o_col] = "O"
        lines.append(f"  {pred:>10.3f}{obs:>10.3f}{count:>7}   {''.join(bar)}")
    lines.append("  P = predicted, O = observed. They coincide when calibrated.")
    return "\n".join(lines)
