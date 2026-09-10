"""Regex-driven parameter placement and model sharding constraints.

See docs/configuration.md#optimizers-and-sharding and JAX sharding APIs:
https://docs.jax.dev/en/latest/jax.sharding.html
"""

import re
from typing import Any

import jax
from flax import traverse_util
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from omegaconf import OmegaConf

from utils.typing_utils import PyTree

AxisName = str | tuple[str, ...] | None


def _axis(value: str | list[str] | tuple[str, ...] | None) -> AxisName:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return str(value)


def _partition_spec(
    values: list[str | list[str] | tuple[str, ...] | None],
) -> P:
    return P(*(_axis(value) for value in values))


def _validate(path: str, value: jax.Array, spec: P, mesh: Mesh) -> None:
    if len(spec) > value.ndim:
        raise ValueError(f"sharding rank {len(spec)} exceeds {path} rank {value.ndim}")
    for dimension, axes in enumerate(spec):
        axes = () if axes is None else (axes if isinstance(axes, tuple) else (axes,))
        size = 1
        for axis in axes:
            if axis not in mesh.axis_names:
                raise ValueError(f"unknown mesh axis {axis!r} in sharding for {path}")
            size *= mesh.shape[axis]
        if value.shape[dimension] % size:
            raise ValueError(
                f"{path} dimension {dimension} ({value.shape[dimension]}) "
                f"is not divisible by sharding size {size}"
            )


def parameter_shardings(params: PyTree, mesh: Mesh, config: Any) -> PyTree:
    """Build parameter shardings from explicit, ordered regex rules."""
    # fully replicate by default
    config = (
        OmegaConf.to_container(config, resolve=True)
        if OmegaConf.is_config(config)
        else config
    )
    default = _partition_spec(config.get("default", []))
    rules = [
        (re.compile(rule["pattern"]), _partition_spec(rule["spec"]), rule["pattern"])
        for rule in config.get("rules", [])
    ]
    hits = {pattern: 0 for _, _, pattern in rules}
    output = {}
    for path, value in traverse_util.flatten_dict(params).items():
        name = "/".join(path)
        spec = default
        for pattern, candidate, source in rules:
            if pattern.search(name):
                spec = candidate
                hits[source] += 1
                break
        _validate(name, value, spec, mesh)
        output[path] = NamedSharding(mesh, spec)

    unmatched = [pattern for pattern, count in hits.items() if count == 0]
    if unmatched:
        raise ValueError(f"parameter sharding rules matched nothing: {unmatched}")
    return traverse_util.unflatten_dict(output)


def place_parameters(params: PyTree, mesh: Mesh, config: Any) -> PyTree:
    """Place every parameter using the configured sharding rules."""
    return jax.device_put(params, parameter_shardings(params, mesh, config))


def with_sharding_constraint(
    value: jax.Array,
    spec: list[str | list[str] | tuple[str, ...] | None],
) -> jax.Array:
    """Constrain ``value`` when a named mesh is active, otherwise leave it local."""
    if jax.sharding.get_abstract_mesh().empty:
        return value
    return jax.lax.with_sharding_constraint(value, _partition_spec(spec))
