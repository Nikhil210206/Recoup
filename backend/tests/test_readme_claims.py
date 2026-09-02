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
from app.evaluation import BUDGET_FRACTION
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


class TestRefusalClaims:
    """The README quotes what the gates refused. These recompute it.

    The governance total and the budget-exhaustion line are asserted separately
    and their sum is asserted NOT to appear, because an earlier draft of this
    section reported the combined 507 cases / Rs 481,799 as "how often the gate
    refused". That number is in a believable range, has a correct-sounding
    label, and measures neither quantity.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def refusals(test_slice):
        est = CauseRate().fit(build_frames("base", 42)["train"])
        alloc = Allocator(
            estimator=est, budget_policy=BudgetPolicy(max_contacts_per_customer=2)
        )
        alloc.budget_policy.max_total_contacts = BUDGET
        decisions, _ = alloc.plan(test_slice)
        amount = dict(zip(test_slice.case_id, test_slice.amount_paise, strict=True))
        refused = [d for d in decisions if not d.acted]
        return decisions, refused, amount

    def test_governance_refusal_totals_match_the_readme(self, readme, refusals):
        _, refused, amount = refusals
        gov = [d for d in refused if d.reason != "global_budget_exhausted"]
        rupees = sum(amount[d.case_id] for d in gov) / 100

        assert _quoted_int(readme, r"\*\*([\d,]+)\*\* \| \*\*₹[\d,]+\*\* \|") == len(gov)
        assert _quoted_int(readme, r"\*\*₹([\d,]+)\*\* \|") == round(rupees)

    def test_the_denominator_matches_the_readme(self, readme, refusals):
        """The sentence introducing the table quotes the split size and how many
        cases were acted on. Both move if the budget or the world changes, and
        neither was covered until this test."""
        decisions, _, _ = refusals
        assert _quoted_int(readme, r"([\d,]+)-case test split") == len(decisions)
        assert _quoted_int(readme, r"([\d,]+) were acted on") == sum(
            1 for d in decisions if d.acted
        )

    def test_the_suppressed_money_is_all_risk_blocked(self, readme, refusals):
        """The README says all 27 are `risk_blocked`. An earlier draft said they
        were a mix including merchant misconfiguration -- which is false: a
        merchant_config case is ACTED on with a merchant alert, not refused."""
        _, refused, amount = refusals
        blocked = [d for d in refused if d.rule == "risk_suppression"]

        assert {d.cause for d in blocked} == {"risk_blocked"}
        assert len(blocked) == 27
        assert round(sum(amount[d.case_id] for d in blocked) / 100) == 176_142
        assert "₹176,142" in readme

    def test_merchant_config_is_acted_on_not_refused(self, readme, refusals):
        decisions, _, _ = refusals
        alerted = [d for d in decisions if d.rule == "merchant_alert_only"]
        assert alerted, "no merchant alerts planned; the README claims 9"
        assert {d.cause for d in alerted} == {"merchant_config"}
        assert all(d.acted for d in alerted)
        # \s+ not a literal newline: this repo has lost edits before to text
        # that had been rewrapped since it was written.
        assert _quoted_int(readme, r"([\d]+)\s+such\s+cases\s+are\s+acted\s+on") == len(
            alerted
        )

    def test_exhaustion_is_reported_separately_and_never_summed(self, readme, refusals):
        _, refused, amount = refusals
        gov = [d for d in refused if d.reason != "global_budget_exhausted"]
        out = [d for d in refused if d.reason == "global_budget_exhausted"]
        gov_rs = round(sum(amount[d.case_id] for d in gov) / 100)
        out_rs = round(sum(amount[d.case_id] for d in out) / 100)

        assert f"{len(out):,} cases (₹{out_rs:,})" in readme

        # The combined figure must not appear. This is the assertion that
        # would have caught the original draft.
        assert f"₹{gov_rs + out_rs:,}" not in readme, (
            "README quotes governance + exhaustion as one number; they measure "
            "different things and must stay on separate lines"
        )


class TestHeadlineLever:
    def test_action_selection_lift(self, test_slice):
        """The project's largest claim: cause-aware action and timing."""
        results = compare(test_slice, budget=int(len(test_slice) * BUDGET_FRACTION))
        lift = max(r.lift for r in results if r.lever == "action")
        assert 5.0 < lift < 5.5, f"README says ~+512%, measured {lift:+.0%}"

    def test_action_lift_holds_in_every_world(self):
        """README claims +506% to +517% across worlds."""
        for world in WORLDS:
            cases = load(world, 42)
            slice_ = cases[cases.split == "test"].reset_index(drop=True)
            results = compare(slice_, budget=int(len(slice_) * BUDGET_FRACTION))
            lift = max(r.lift for r in results if r.lever == "action")
            assert 5.0 < lift < 5.5, f"{world}: {lift:+.0%}"

    def test_ev_ranking_is_worth_about_one_percent(self, test_slice):
        results = compare(test_slice, budget=int(len(test_slice) * BUDGET_FRACTION))
        lift = max(r.lift for r in results if r.lever == "ranking")
        assert lift < 0.05, f"README says +1% with oracle uplift, measured {lift:+.1%}"


class TestAllocatorClaims:
    def test_value_ordering_lift(self, controls):
        _, arrival, pooled, _ = controls
        lift = (pooled.incremental_paise - arrival.incremental_paise) / arrival.incremental_paise
        assert 0.14 < lift < 0.24, f"README quotes this figure; measured {lift:+.1%}"

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
        """The label may be phrased freely, but it must be near the top and it
        must be impossible to read the headline numbers without meeting it.

        This asserts the section, not just the phrase: an earlier version only
        checked for the words "Simulation benchmark", which a rewrite silently
        dropped while the honesty itself was being strengthened.
        """
        assert "simulation benchmark" in readme.lower()

        heading = "## What is real and what is not"
        assert heading in readme, "the real/simulated split must be a section"

        # Before the first result. "The finding" is where the numbers start.
        assert readme.index(heading) < readme.index("## The finding")

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


class TestEvaluationHarness:
    """The evaluation is only worth anything if a stranger can reproduce it.

    `make eval` producing the same table twice is the reason a panel should
    believe the rest. It has been broken once already -- `hash()` is randomised
    per process, so four consecutive runs of the same command produced four
    different held-out scores while the harness printed an accurate-looking
    configuration.
    """

    def test_two_runs_produce_identical_output(self):
        import io
        from contextlib import redirect_stdout

        from app.evaluation.harness import run

        outputs = []
        for _ in range(2):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                report, meta = run("base", 42)
            # `generated` is a date, and is the only field allowed to move.
            meta.pop("generated", None)
            outputs.append((buffer.getvalue(), meta))

        assert outputs[0][0] == outputs[1][0], "console output differed between runs"
        assert outputs[0][1] == outputs[1][1], "metadata differed between runs"

    def test_every_section_is_present(self):
        import io
        from contextlib import redirect_stdout

        from app.evaluation.harness import run

        with redirect_stdout(io.StringIO()):
            report, _ = run("base", 42)

        titles = [t.lower() for t, _ in report.sections]
        assert len(titles) == 8, titles

        # Each of these is a claim the write-up depends on. If a section is
        # dropped, the corresponding claim in README or ARCHITECTURE becomes
        # unsupported, and nothing else would notice.
        required = [
            "gates refused",          # governance, quantified rather than asserted
            "recovers the money",     # which decision is the lever
            "identical contact budget",  # every arm, fairly compared
            "component is worth",     # the ablation, including the parts worth ~0
            "significance",           # paired bootstrap vs Razorpay's T+3
            "left to take",           # the oracle ceiling
            "different world",        # sweep A
            "wrong cause",            # sweep B
        ]
        for expected in required:
            assert any(expected in t for t in titles), f"missing section: {expected}"


class TestArchitectureDoc:
    @staticmethod
    def _doc() -> str:
        from pathlib import Path

        return (Path(__file__).resolve().parents[2] / "ARCHITECTURE.md").read_text()

    def test_it_documents_where_a_model_was_refused(self):
        """A scored criterion: 'the right tool in the right place, and where you
        chose not to use one'. If this section disappears, the strongest part of
        the write-up has gone with it."""
        doc = self._doc()
        assert "Where a model is used, and where one was refused" in doc
        for claim in ("3 of 11", "0.95 confidence", "24% worse than no estimate"):
            assert claim in doc, f"missing measured justification: {claim}"

    def test_it_states_what_the_evaluation_cannot_support(self):
        doc = self._doc()
        assert "cannot" in doc
        assert "circularity" in doc or "generating process is known" in doc


class TestNoDriftBetweenDocsAndHarness:
    """README, EVALUATION.md and the harness must describe the same experiment.

    They did not. The README quoted a lever study run at a 15% contact budget
    while `make eval` used 25%, so the two documents reported different numbers
    for the same experiment and neither was wrong on its own terms. The budget is
    now defined once and imported.
    """

    def test_only_one_budget_fraction_is_defined(self):
        import subprocess
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        hits = [
            line
            for line in subprocess.run(
                ["grep", "-rn", "^BUDGET_FRACTION = ", str(root / "app")],
                capture_output=True, text=True,
            ).stdout.strip().splitlines()
            if line
        ]
        assert len(hits) == 1, f"BUDGET_FRACTION defined in {len(hits)} places: {hits}"

    def test_no_hardcoded_budget_fractions_remain(self):
        """`int(len(x) * 0.15)` in one file and `* 0.25` in another is exactly how
        the drift happened."""
        import subprocess
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        hits = [
            line
            for line in subprocess.run(
                ["grep", "-rnE", r"len\([a-z_]+\) \* 0\.[0-9]+",
                 str(root / "app"), str(root / "tests")],
                capture_output=True, text=True,
            ).stdout.strip().splitlines()
            # This file describes the pattern it forbids, so it must exempt itself.
            if line and "test_readme_claims" not in line
        ]
        assert not hits, "hardcoded budget fraction still present:\n" + "\n".join(hits)

    def test_readme_lever_number_matches_a_fresh_run(self):
        from app.evaluation import BUDGET_FRACTION
        from app.model.levers import compare

        cases = load("base", 42)
        test = cases[cases.split == "test"].reset_index(drop=True)
        results = compare(test, int(len(test) * BUDGET_FRACTION))
        measured = max(r.lift for r in results if r.lever == "action")

        readme = README.read_text()
        # The README quotes this to the nearest whole percent.
        assert f"+{measured * 100:.0f}%" in readme, (
            f"README does not quote the measured lever lift of {measured:+.0%}"
        )


class TestDocumentsAgreeWithEachOther:
    """Both documents quote the same measurements. They must quote them alike.

    The existing checks assert a measured value falls inside a range and that
    the README mentions the right number *somewhere*. Neither notices a second,
    different number for the same claim: the README's diagram said +520% while
    its own table said +512%, and the value-ordering figure read +18.5% in the
    README and +17% in ARCHITECTURE. Every test passed. The ranges were wide
    enough to contain both, and "is this number present" cannot detect that a
    contradicting one is present too.
    """

    DOCS = ("README.md", "ARCHITECTURE.md")

    def _text(self, name: str) -> str:
        return (README.parent / name).read_text()

    def _measured_action_lift(self) -> float:
        from app.evaluation import BUDGET_FRACTION
        from app.model.levers import compare

        cases = load("base", 42)
        test = cases[cases.split == "test"].reset_index(drop=True)
        results = compare(test, int(len(test) * BUDGET_FRACTION))
        return max(r.lift for r in results if r.lever == "action")

    def test_every_large_percentage_matches_something_measured(self):
        """No document may quote a hundreds-of-percent figure nothing produces.

        These documents legitimately contain several: the other lever variants
        (+229%, +386%) and the range across worlds (+506% to +517%). So the
        check is not "is it the headline" but "is it *any* lift a fresh run
        produces". A leftover +520% from a previous budget setting matches
        nothing and fails, which is exactly how it survived before -- every
        range was wide enough to contain it and no test compared it to a run.
        """
        import re

        from app.evaluation import BUDGET_FRACTION

        measured: set[float] = set()
        for world in WORLDS:
            cases = load(world, 42)
            slice_ = cases[cases.split == "test"].reset_index(drop=True)
            for result in compare(slice_, int(len(slice_) * BUDGET_FRACTION)):
                if result.lift > 0.5:
                    measured.add(result.lift * 100)

        for name in self.DOCS:
            for match in re.findall(r"\+(\d{3})%", self._text(name)):
                quoted = float(match)
                assert any(abs(quoted - m) < 1.0 for m in measured), (
                    f"{name} quotes +{match}%, which no fresh run reproduces; "
                    f"measured lifts are {sorted(round(m) for m in measured)}"
                )

    def test_both_documents_quote_the_same_value_ordering_lift(self, controls):
        """The value-ordering figure, stated in two places.

        Note which arms this compares. Ordering alone is `pooled+ranked` against
        `arrival` -- the same policy, differing only in queue order. The
        allocator against arrival is +17%, because it also pays for contact
        caps, and the two were quoted interchangeably under one label.
        """
        import re

        _, arrival, pooled, _ = controls
        measured = (
            pooled.incremental_paise - arrival.incremental_paise
        ) / arrival.incremental_paise

        quoted = set()
        for name in self.DOCS:
            for match in re.finditer(
                r"\+(\d+(?:\.\d+)?)%[^\n]*arrival order", self._text(name)
            ):
                quoted.add(float(match.group(1)))

        assert quoted, "neither document states the value-ordering lift"
        assert len(quoted) == 1, (
            "the documents quote different figures for the same comparison: "
            f"{sorted(quoted)}"
        )
        assert abs(quoted.pop() - measured * 100) < 0.6, (
            f"documents disagree with a fresh run, which measures {measured:+.1%}"
        )
