"""Every headline number in README.md, checked against live output.

A README is a judged artifact and the most likely place for a number to go
stale: the code moves, the prose does not, and nothing fails. This file already
caught one -- the case count was quoted as 11,917 after adding the third loss
channel changed it to 12,004.

Tolerances are deliberately loose. These pin the *claim*, not the exact float:
a number drifting by a percent is fine, a number that has become a different
number is not.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.allocator.bake_off import ranked, run_bake_off
from app.allocator.budget import BudgetPolicy
from app.allocator.estimator import CauseRate
from app.allocator.policy import Allocator
from app.allocator.siloed import run_siloed
from app.model.levers import compare
from app.model.train import build_frames
from app.simulation.arms import load
from app.simulation.generator import WORLDS
from app.simulation.policies import CauseAware

README = Path(__file__).resolve().parents[2] / "README.md"
BUDGET = 600


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text()


@pytest.fixture(scope="module")
def cases():
    return load("base", 42)


@pytest.fixture(scope="module")
def test_slice(cases):
    return cases[cases.split == "test"].reset_index(drop=True)


@pytest.fixture(scope="module")
def controls(test_slice):
    est = CauseRate().fit(build_frames("base", 42)["train"])

    def allocator(cap=2):
        return Allocator(
            estimator=est, budget_policy=BudgetPolicy(max_contacts_per_customer=cap)
        )

    alloc = run_bake_off(test_slice, allocator(), {}, BUDGET)[0].result
    arrival = next(
        r.result
        for r in run_bake_off(test_slice, allocator(99), {"a": CauseAware()}, BUDGET)
        if r.result.arm == "a"
    )
    pooled = next(
        r.result
        for r in run_bake_off(ranked(test_slice), allocator(99), {"p": CauseAware()}, BUDGET)
        if r.result.arm == "p"
    )
    siloed, _ = run_siloed(test_slice, CauseAware, BUDGET)
    return alloc, arrival, pooled, siloed


def _quoted_int(readme: str, pattern: str) -> int:
    match = re.search(pattern, readme)
    assert match, f"README no longer contains a number matching {pattern!r}"
    return int(match.group(1).replace(",", ""))


class TestDatasetClaims:
    def test_case_count_matches_the_readme(self, readme, cases):
        assert _quoted_int(readme, r"\*\*([\d,]+) cases\*\*, 90 days") == len(cases)

    def test_three_loss_channels(self, readme, cases):
        assert "3 loss channels" in readme
        assert cases.channel.nunique() == 3

    def test_published_error_codes_count(self, readme):
        from app import taxonomy

        assert f"{len(taxonomy.REASON_TO_CAUSE)} published codes" in readme


class TestHeadlineLever:
    def test_action_selection_lift(self, test_slice):
        """The project's largest claim: ~+520% for cause-aware action and timing."""
        results = compare(test_slice, budget=int(len(test_slice) * 0.15))
        lift = max(r.lift for r in results if r.lever == "action")
        assert 5.0 < lift < 5.5, f"README says ~+524%, measured {lift:+.0%}"

    def test_action_lift_holds_in_every_world(self):
        """README claims +519% to +529% across worlds."""
        for world in WORLDS:
            cases = load(world, 42)
            slice_ = cases[cases.split == "test"].reset_index(drop=True)
            results = compare(slice_, budget=int(len(slice_) * 0.15))
            lift = max(r.lift for r in results if r.lever == "action")
            assert 5.0 < lift < 5.5, f"{world}: {lift:+.0%}"

    def test_ev_ranking_is_worth_about_one_percent(self, test_slice):
        results = compare(test_slice, budget=int(len(test_slice) * 0.15))
        lift = max(r.lift for r in results if r.lever == "ranking")
        assert lift < 0.05, f"README says +1% with oracle uplift, measured {lift:+.1%}"


class TestAllocatorClaims:
    def test_value_ordering_lift(self, controls):
        _, arrival, pooled, _ = controls
        lift = (pooled.incremental_paise - arrival.incremental_paise) / arrival.incremental_paise
        assert 0.14 < lift < 0.24, f"README says +18.5%, measured {lift:+.1%}"

    def test_allocator_is_within_noise_of_siloed_agents(self, controls):
        """The honest ~0. README states it plainly; this stops it drifting into
        an unearned win."""
        alloc, _, _, siloed = controls
        delta = (alloc.incremental_paise - siloed.incremental_paise) / siloed.incremental_paise
        assert -0.05 < delta < 0.05, f"README says +1.0%, measured {delta:+.1%}"

    def test_allocator_is_within_noise_of_pooled_ranked(self, controls):
        alloc, _, pooled, _ = controls
        delta = (alloc.incremental_paise - pooled.incremental_paise) / pooled.incremental_paise
        assert -0.05 < delta < 0.05, f"README says -1.3%, measured {delta:+.1%}"


class TestHonestyClaims:
    def test_readme_labels_results_as_simulated(self, readme):
        assert "Simulation benchmark" in readme or "SIMULATION BENCHMARK" in readme

    def test_readme_states_its_limitations(self, readme):
        """The section that says what the work cannot support. Its absence would
        be a bigger problem than any single number being wrong."""
        assert "What this cannot tell you" in readme
        for limitation in ("optimistically", "uniformly", "cannot validate the model"):
            assert limitation in readme, f"missing stated limitation: {limitation}"

    def test_readme_reports_the_negative_results(self, readme):
        """The GBM losing to a group-by, and the allocator's ~0, are load-bearing
        for this project's credibility. They must stay in the README."""
        assert "worse than no estimate" in readme
        assert "worth ~0" in readme or "worth ~0%" in readme
