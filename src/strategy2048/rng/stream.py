"""NumPy PCG64DXSM streams with explicit derivation and snapshots."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

RNG_SCHEMA_VERSION = "rng-v2"
RNG_DERIVATION_SCHEMA_VERSION = "rng-v1"
_SUPPORTED_RNG_SCHEMA_VERSIONS = {"rng-v1", RNG_SCHEMA_VERSION}
_RNG_LINEAGE_FIELDS = {
    "root_seed",
    "purpose",
    "environment_id",
    "episode_id",
    "derivation_schema",
}
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
    schema_version: str = RNG_DERIVATION_SCHEMA_VERSION,
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
    lineage: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "bit_generator": self.bit_generator,
            "seed": self.seed,
            "counter": self.counter,
            "state": _jsonable(self.state),
            "lineage": _jsonable(self.lineage),
        }

    @classmethod
    def from_json(cls, value: object) -> RNGSnapshot:
        if not isinstance(value, dict):
            raise ValueError("rng snapshot must be an object")
        required = {"schema_version", "bit_generator", "seed", "counter", "state"}
        allowed = required | {"lineage"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError("rng snapshot contains unknown fields: " + ", ".join(sorted(unknown)))
        if not required <= set(value):
            raise ValueError("rng snapshot is missing required fields")
        if not isinstance(value["schema_version"], str):
            raise ValueError("rng snapshot schema_version must be a string")
        if not isinstance(value["bit_generator"], str):
            raise ValueError("rng snapshot bit_generator must be a string")
        if type(value["seed"]) is not int or type(value["counter"]) is not int:
            raise ValueError("rng snapshot seed and counter must be integers")
        if value["seed"] < 0 or value["counter"] < 0:
            raise ValueError("rng snapshot seed and counter must be non-negative")
        if not isinstance(value["state"], dict):
            raise ValueError("rng snapshot state must be an object")
        schema_version = value["schema_version"]
        if schema_version not in _SUPPORTED_RNG_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported RNG schema: {schema_version}")
        lineage = value.get("lineage")
        if schema_version == RNG_SCHEMA_VERSION and not isinstance(lineage, dict):
            raise ValueError("rng snapshot lineage must be an object")
        if lineage is not None and not isinstance(lineage, dict):
            raise ValueError("rng snapshot lineage must be an object")
        if schema_version == RNG_SCHEMA_VERSION:
            if not isinstance(lineage, dict):
                raise ValueError("rng snapshot lineage must be an object")
            lineage_v2 = lineage
            if set(lineage_v2) != _RNG_LINEAGE_FIELDS:
                raise ValueError("rng-v2 lineage must contain the complete field set")
            if lineage_v2["root_seed"] is not None and not isinstance(lineage_v2["root_seed"], str):
                raise ValueError("rng lineage root_seed must be a string or null")
            if not isinstance(lineage_v2["purpose"], str) or not lineage_v2["purpose"]:
                raise ValueError("rng lineage purpose must be a non-empty string")
            if lineage_v2["environment_id"] is not None and not isinstance(
                lineage_v2["environment_id"], str
            ):
                raise ValueError("rng lineage environment_id must be a string or null")
            episode_id = lineage_v2["episode_id"]
            if episode_id is not None and (type(episode_id) is not int or episode_id < 0):
                raise ValueError("rng lineage episode_id must be a non-negative integer or null")
            if lineage_v2["derivation_schema"] is not None and not isinstance(
                lineage_v2["derivation_schema"], str
            ):
                raise ValueError("rng lineage derivation_schema must be a string or null")
        return cls(
            schema_version=schema_version,
            bit_generator=value["bit_generator"],
            seed=value["seed"],
            counter=value["counter"],
            state=dict(value["state"]),
            lineage=dict(
                lineage
                or {
                    "root_seed": None,
                    "purpose": "legacy-snapshot",
                    "environment_id": None,
                    "episode_id": None,
                    "derivation_schema": RNG_DERIVATION_SCHEMA_VERSION,
                }
            ),
        )


class ScientificRNG:
    """Explicit raw-integer stream used by all official chance sampling."""

    def __init__(
        self,
        seed: int,
        *,
        purpose: str = "unspecified",
        lineage: Mapping[str, Any] | None = None,
    ) -> None:
        self.seed = int(seed)
        self.purpose = purpose
        self.lineage = dict(
            lineage
            or {
                "root_seed": None,
                "purpose": purpose,
                "environment_id": None,
                "episode_id": None,
                "derivation_schema": None,
            }
        )
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
            lineage=dict(self.lineage),
        )

    def restore(self, snapshot: RNGSnapshot | dict[str, Any]) -> None:
        if isinstance(snapshot, dict):
            snapshot = RNGSnapshot.from_json(snapshot)
        elif isinstance(snapshot, RNGSnapshot):
            snapshot = RNGSnapshot.from_json(snapshot.to_json())
        else:
            raise ValueError("RNG snapshot must be an RNGSnapshot or object")
        if snapshot.schema_version not in _SUPPORTED_RNG_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported RNG schema: {snapshot.schema_version}")
        if snapshot.bit_generator != "PCG64DXSM":
            raise ValueError(f"unsupported bit generator: {snapshot.bit_generator}")
        if snapshot.seed != self.seed:
            raise ValueError("RNG seed mismatch")
        if snapshot.counter < 0:
            raise ValueError("RNG counter must be non-negative")
        bit_generator = np.random.PCG64DXSM(self.seed)
        try:
            bit_generator.state = cast(Any, snapshot.state)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid RNG bit-generator state") from exc
        self._bit_generator = bit_generator
        self._counter = snapshot.counter
        self.lineage = dict(snapshot.lineage)
        self.purpose = str(self.lineage.get("purpose", self.purpose))

    def clone(self) -> ScientificRNG:
        clone = ScientificRNG(self.seed, purpose=self.purpose, lineage=self.lineage)
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
        lineage={
            "root_seed": normalize_root_seed(root_seed),
            "purpose": purpose,
            "environment_id": environment_id,
            "episode_id": episode_id,
            "derivation_schema": RNG_DERIVATION_SCHEMA_VERSION,
        },
    )
