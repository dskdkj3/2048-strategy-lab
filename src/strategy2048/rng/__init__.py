"""Stable, domain-separated randomness for scientific experiments."""

from strategy2048.rng.stream import (
    RNG_SCHEMA_VERSION,
    ScientificRNG,
    derive_seed,
    normalize_root_seed,
    rng_for,
)

__all__ = ["RNG_SCHEMA_VERSION", "ScientificRNG", "derive_seed", "normalize_root_seed", "rng_for"]
