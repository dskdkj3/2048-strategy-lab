"""NumPy PCG64DXSM streams with explicit derivation and snapshots."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

RNG_SCHEMA_VERSION = "rng-v1"
_UINT64_SPACE = 1 << 64


def normalize_root_seed(seed: int | str) -> str:
    if isinstance(seed, bool):
        raise TypeError("boolean is not a valid root seed")
    if isinstance(seed, int):
        return str(seed)
    if isinstance(seed, str) and seed:
        return seed
    raise TypeError("root seed must be a non-empty string or integer")


def derive_seed(
    root_seed: int | str,
    purpose: str,
    environment_id: str = "",
    episode_id: int = 0,
    schema_version: str = RNG_SCHEMA_VERSION,
) -> int:
    """Derive a stable 128-bit PCG seed with domain separation."""

    root = normalize_root_seed(root_seed)
    if episode_id < 0:
        raise ValueError("episode_id must be non-negative")
    payload = "\0".join((schema_version, root, purpose, environment_id, str(episode_id))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big", signed=False)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True, slots=True)
class RNGSnapshot:
    schema_version: str
    bit_generator: str
    seed: int
    counter: int
    state: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "bit_generator": self.bit_generator,
            "seed": self.seed,
            "counter": self.counter,
            "state": _jsonable(self.state),
        }

    @classmethod
    def from_json(cls, value: object) -> RNGSnapshot:
        if not isinstance(value, dict):
            raise ValueError("rng snapshot must be an object")
        required = ("schema_version", "bit_generator", "seed", "counter", "state")
        if any(key not in value for key in required):
            raise ValueError("rng snapshot is missing required fields")
        return cls(
            schema_version=str(value["schema_version"]),
            bit_generator=str(value["bit_generator"]),
            seed=int(value["seed"]),
            counter=int(value["counter"]),
            state=dict(value["state"]),
        )


class ScientificRNG:
    """Explicit raw-integer stream used by all official chance sampling."""

    def __init__(self, seed: int, *, purpose: str = "unspecified") -> None:
        self.seed = int(seed)
        self.purpose = purpose
        self._bit_generator = np.random.PCG64DXSM(self.seed)
        self._counter = 0

    @property
    def counter(self) -> int:
        return self._counter

    def raw_uint64(self) -> int:
        value = int(self._bit_generator.random_raw())
        self._counter += 1
        return value

    def randbelow(self, upper: int) -> int:
        if upper <= 0:
            raise ValueError("upper must be positive")
        if upper > _UINT64_SPACE:
            raise ValueError("upper must not exceed the uint64 sample space")
        limit = (_UINT64_SPACE // upper) * upper
        while True:
            value = self.raw_uint64()
            if value < limit:
                return value % upper

    def snapshot(self) -> RNGSnapshot:
        return RNGSnapshot(
            schema_version=RNG_SCHEMA_VERSION,
            bit_generator="PCG64DXSM",
            seed=self.seed,
            counter=self.counter,
            state=_jsonable(self._bit_generator.state),
        )

    def restore(self, snapshot: RNGSnapshot | dict[str, Any]) -> None:
        if isinstance(snapshot, dict):
            snapshot = RNGSnapshot.from_json(snapshot)
        if snapshot.schema_version != RNG_SCHEMA_VERSION:
            raise ValueError(f"unsupported RNG schema: {snapshot.schema_version}")
        if snapshot.bit_generator != "PCG64DXSM":
            raise ValueError(f"unsupported bit generator: {snapshot.bit_generator}")
        if snapshot.seed != self.seed:
            raise ValueError("RNG seed mismatch")
        self._bit_generator.state = cast(Any, snapshot.state)
        self._counter = snapshot.counter

    def clone(self) -> ScientificRNG:
        clone = ScientificRNG(self.seed, purpose=self.purpose)
        clone.restore(self.snapshot())
        return clone

    def to_json(self) -> str:
        return json.dumps(self.snapshot().to_json(), sort_keys=True, separators=(",", ":"))


def rng_for(
    root_seed: int | str,
    purpose: str,
    environment_id: str = "",
    episode_id: int = 0,
) -> ScientificRNG:
    return ScientificRNG(
        derive_seed(root_seed, purpose, environment_id, episode_id),
        purpose=purpose,
    )
