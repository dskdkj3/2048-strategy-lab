"""From-scratch afterstate n-tuple TD(0) learner."""

from strategy2048.learning.td import (
    DEFAULT_TUPLES,
    TD1PAgent,
    TDLearner,
    TupleValueFunction,
)

__all__ = ["DEFAULT_TUPLES", "TD1PAgent", "TDLearner", "TupleValueFunction"]
