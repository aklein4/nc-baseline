"""Optimizer construction and model-owned parameter annotations.

See docs/architecture.md#optimization and the Flax module-path API:
https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.Module.module_paths
"""

from typing import Any

import jax
import optax
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf

from utils.typing_utils import PyTree

FROZEN = "frozen"


def gradient_norm(grads: PyTree, labels: PyTree | None = None) -> jax.Array:
    """Return the norm of all gradients that belong to trainable groups."""
    if labels is None:
        return optax.tree.norm(grads)
    trainable = tuple(
        gradient
        for gradient, label in zip(jax.tree.leaves(grads), jax.tree.leaves(labels))
        if label != FROZEN
    )
    return optax.tree.norm(trainable)


def no_muon_mask(model: Any, params: PyTree) -> PyTree:
    """Resolve submodule-local ``no_muon_patterns`` over a parameter tree."""
    if not hasattr(model, "get_init_inputs") or not hasattr(model, "module_paths"):
        return jax.tree.map(lambda _: False, params)
    args, kwargs = model.get_init_inputs()
    modules = model.module_paths(jax.random.key(0), *args, **kwargs)
    rules = [
        (tuple(filter(None, path.split("/"))), module.no_muon_patterns)
        for path, module in modules.items()
        if getattr(module, "no_muon_patterns", ())
    ]

    def annotate(path: tuple[Any, ...], _value: Any) -> bool:
        parts = tuple(
            str(
                getattr(
                    entry,
                    "key",
                    getattr(entry, "name", getattr(entry, "idx", entry)),
                )
            )
            for entry in path
        )
        return any(
            parts[: len(prefix)] == prefix
            and any(pattern in "/".join(parts[len(prefix) :]) for pattern in patterns)
            for prefix, patterns in rules
        )

    return jax.tree_util.tree_map_with_path(annotate, params)


def muon_dimension_numbers(
    path: Any, value: jax.Array, excluded: bool
) -> optax.contrib.MuonDimensionNumbers | None:
    """Classify one scanned JAX leaf as a logical matrix or Adam fallback.

    Linen scan stacks per-layer parameters on axis zero, so kernels become rank
    three while biases become rank two. The returned axes recover the logical
    unscanned classification expected by Optax Muon. Bias and scale names
    distinguish vector leaves from matrices; algorithm-specific exclusions
    come from the owning module through ``excluded``.
    """
    name = str(
        getattr(
            path[-1],
            "key",
            getattr(path[-1], "name", getattr(path[-1], "idx", path[-1])),
        )
    )
    is_matrix = value.ndim >= 2 and name not in ("bias", "scale")
    if excluded or not is_matrix or min(value.shape[-2:]) <= 1:
        return None
    return optax.contrib.MuonDimensionNumbers(
        reduction_axis=value.ndim - 2,
        output_axis=value.ndim - 1,
    )


def schedule(config: DictConfig) -> optax.Schedule | float:
    """Build the constant, warmup/cosine, or WSD schedule in ``config``."""
    learning_rate = float(config.lr)
    warmup = int(config.get("warmup_steps", 0))
    stable = int(config.get("stable_steps", 0))
    decay = int(config.get("decay_steps", 0))
    minimum = float(config.get("min_lr_ratio", 1.0))
    if not decay:
        return (
            optax.linear_schedule(0, learning_rate, max(warmup, 1))
            if warmup
            else learning_rate
        )
    if stable:
        return optax.join_schedules(
            schedules=[
                optax.linear_schedule(0, learning_rate, max(warmup, 1)),
                optax.constant_schedule(learning_rate),
                optax.cosine_decay_schedule(
                    learning_rate,
                    decay_steps=decay,
                    alpha=minimum,
                ),
            ],
            boundaries=[warmup, warmup + stable],
        )
    return optax.warmup_cosine_decay_schedule(
        init_value=0,
        peak_value=learning_rate,
        warmup_steps=warmup,
        decay_steps=decay,
        end_value=learning_rate * minimum,
    )


def optimizer(
    config: DictConfig,
    learning_rate: optax.Schedule | float | None = None,
    no_muon_mask: PyTree | None = None,
) -> optax.GradientTransformation:
    """Construct an optimizer and route annotated Muon leaves to AdamW."""
    ignored = {
        "_target_",
        "lr",
        "warmup_steps",
        "stable_steps",
        "decay_steps",
        "min_lr_ratio",
    }
    kwargs = {
        key: value
        for key, value in OmegaConf.to_container(config, resolve=True).items()
        if key not in ignored
    }
    if learning_rate is not None:
        kwargs["learning_rate"] = learning_rate
    target = get_method(config["_target_"])

    if no_muon_mask is not None and target is optax.contrib.muon:
        # Linen scan stacks each layer on a leading axis. Dense kernels become
        # rank three, while stacked biases are rank two but remain logical
        # vectors. Explicit dimension numbers preserve the unscanned semantics.
        exclusion_by_path = {
            tuple(path): value
            for path, value in jax.tree_util.tree_flatten_with_path(no_muon_mask)[0]
        }

        def dimensions(path: Any, value: Any) -> Any:
            return muon_dimension_numbers(path, value, exclusion_by_path[tuple(path)])

        kwargs.setdefault(
            "adam_weight_decay",
            config.get("adam_weight_decay", config.get("weight_decay", 0.0)),
        )

        def dimension_tree(params: PyTree) -> PyTree:
            return jax.tree_util.tree_map_with_path(dimensions, params)

        kwargs["muon_weight_dimension_numbers"] = dimension_tree
    return target(**kwargs)


def make_optimizer(
    trainer_config: DictConfig,
    labels: Any = None,
    no_muon: PyTree | None = None,
    *,
    model: Any = None,
    params: PyTree | None = None,
) -> tuple[optax.GradientTransformation, dict[str, optax.Schedule | float]]:
    """Build the configured optimizer, optionally over a labeled parameter tree."""
    configs = (
        (group for name, group in trainer_config.groups.items() if name != FROZEN)
        if labels is not None
        else (trainer_config.optimizer,)
    )
    uses_muon = any(
        get_method(config["_target_"]) is optax.contrib.muon for config in configs
    )
    if no_muon is None and uses_muon and model is not None and params is not None:
        no_muon = no_muon_mask(model, params)

    max_norm = trainer_config.get("max_grad_norm")
    # Sanitize each gradient independently before clipping or optimizer state
    # updates. Infinities are intentionally left unchanged.
    prefix = [optax.zero_nans()]
    if labels is not None:
        # Frozen leaves must not affect the global norm used to scale trainable
        # updates.
        frozen = jax.tree.map(lambda label: label == FROZEN, labels)
        prefix.append(optax.masked(optax.set_to_zero(), frozen))
    if max_norm is not None and max_norm > 0:
        prefix.append(optax.clip_by_global_norm(float(max_norm)))
    if labels is None:
        schedules = {"main": schedule(trainer_config.optimizer)}
        tx = optimizer(trainer_config.optimizer, schedules["main"], no_muon)
    else:
        schedules = {
            name: schedule(group)
            for name, group in trainer_config.groups.items()
            if name != FROZEN and "lr" in group
        }
        transforms = {FROZEN: optax.set_to_zero()}
        transforms.update(
            {
                name: optimizer(group, schedules.get(name), no_muon)
                for name, group in trainer_config.groups.items()
                if name != FROZEN
            }
        )
        tx = optax.multi_transform(transforms, labels)
    return optax.chain(*prefix, tx), schedules
