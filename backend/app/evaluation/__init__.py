"""Evaluation harness and robustness sweeps."""

#: Fraction of cases that may receive a customer contact.
#:
#: Defined once, here, because it was defined twice: the harness used 25% and
#: the README-claim tests used 15%, so the two produced different numbers for
#: the same experiment and the README drifted from the evaluation it quoted.
#: Any comparison that varies the budget must vary it explicitly.
BUDGET_FRACTION = 0.25
