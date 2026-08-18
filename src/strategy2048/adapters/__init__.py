"""Explicit external baseline adapters."""

from strategy2048.adapters.tdl import (
    FIXED_TDL_COMMIT,
    TDLAdapter,
    TDLAdapterError,
    TDLWorkload,
)

__all__ = ["FIXED_TDL_COMMIT", "TDLAdapter", "TDLAdapterError", "TDLWorkload"]
