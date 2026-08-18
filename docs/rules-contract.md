# Official rules contract

`strategy2048.rules.core` is the slow, independent oracle. A board is a
row-major tuple of 16 non-negative exponents (`0` means empty); a tile with
exponent `n` has value `2**n`.

`move_without_spawn` is pure. It compresses each row/column toward the action,
merges adjacent equal exponents once from the moving edge, and returns the
official score delta. A valid move then spawns one `ChanceEvent` in a uniformly
selected empty-cell rank. The spawn exponent is 1 with probability 0.9 and 2
with probability 0.1. Invalid actions do not spawn and do not consume the RNG.

`won` records whether the configured win tile has ever appeared. It is not
termination. `terminated` means no legal action remains; a runner step limit is
reported separately as `truncated`.
