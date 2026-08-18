"""Scientific, reproducible 2048 reference implementation."""

__version__ = "0.1.0"

from strategy2048.rules.core import Action, Board, ChanceEvent, MoveResult

__all__ = ["Action", "Board", "ChanceEvent", "MoveResult", "__version__"]
