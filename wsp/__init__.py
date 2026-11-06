"""Weight-space programming harness.

Pure-Python core (metrics, ops, diff, session, align) runs anywhere; the
subjects that touch real weights (kraken, LoRA) import their engines lazily
and run on asteraix. See docs/DESIGN.md.
"""
from wsp.session import Session, BudgetExhausted  # noqa: F401
from wsp.subject import Subject, Prediction  # noqa: F401

__all__ = ["Session", "BudgetExhausted", "Subject", "Prediction"]
