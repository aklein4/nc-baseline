"""Memory-bounded, reusable language-model losses.

See docs/architecture.md#memory-bounded-computation and JAX rematerialization:
https://docs.jax.dev/en/latest/gradient-checkpointing.html
"""

from typing import Any

import jax
import jax.numpy as jnp

from models.base import CustomModel
from utils.typing_utils import PyTree


def _loss_chunks(
    states: jax.Array,
    labels: jax.Array,
    weights: jax.Array,
    chunks: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    size = states.shape[0] * states.shape[1]
    if chunks <= 0:
        raise ValueError(f"{chunks=} must be positive")
    single_weight = weights.ndim == labels.ndim
    if single_weight:
        weights = weights[..., None]
    elif weights.ndim != labels.ndim + 1:
        raise ValueError("weights must match labels or add one trailing reduction axis")
    reductions = weights.shape[-1]
    padding = (-size) % chunks
    states = jnp.pad(states.reshape(size, states.shape[-1]), ((0, padding), (0, 0)))
    labels = jnp.pad(labels.reshape(size), ((0, padding),))
    weights = jnp.pad(
        weights.reshape(size, reductions).astype(jnp.float32),
        ((0, padding), (0, 0)),
    )
    chunk_size = (size + padding) // chunks
    # this transpose may improve the sharding behavior against non-transposed reshaping
    states = states.reshape(chunk_size, chunks, states.shape[-1]).swapaxes(0, 1)
    labels = labels.reshape(chunk_size, chunks).swapaxes(0, 1)
    weights = weights.reshape(chunk_size, chunks, reductions).swapaxes(0, 1)
    labels = jnp.where(jnp.any(weights > 0, axis=-1), labels, 0)
    return states, labels, weights


def chunked_lm_loss(
    model: CustomModel,
    params: PyTree,
    states: jax.Array,
    labels: jax.Array,
    weights: jax.Array,
    chunks: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Cross entropy with one short-lived vocabulary projection per chunk."""
    single_weight = weights.ndim == labels.ndim
    states, labels, weights = _loss_chunks(states, labels, weights, chunks)
    reductions = weights.shape[-1]

    def body(carry: Any, values: Any) -> Any:
        loss, correct = carry
        chunk_states, chunk_labels, chunk_weights = values
        logits = model.apply(
            {"params": params}, chunk_states, method=model.head
        ).astype(jnp.float32)
        target = jnp.take_along_axis(logits, chunk_labels[:, None], axis=-1)[:, 0]
        token_loss = jax.nn.logsumexp(logits, axis=-1) - target
        loss += jnp.sum(token_loss[:, None] * chunk_weights, axis=0)
        correct += jax.lax.stop_gradient(
            jnp.sum(
                (jnp.argmax(logits, axis=-1) == chunk_labels)[:, None] * chunk_weights,
                axis=0,
            )
        )
        return (loss, correct), None

    # Rematerializing the scan body makes reverse mode recompute each chunk's
    # logits instead of retaining vocabulary-sized residuals across iterations.
    # https://docs.jax.dev/en/latest/gradient-checkpointing.html
    body = jax.checkpoint(body, prevent_cse=False)
    initial = (jnp.zeros(reductions, jnp.float32), jnp.zeros(reductions, jnp.float32))
    totals, _ = jax.lax.scan(body, initial, (states, labels, weights))
    if single_weight:
        loss, correct = totals
        return loss[0], correct[0]
    return totals


def frozen_head_lm_loss(
    model: CustomModel,
    params: PyTree,
    states: jax.Array,
    input_ids: jax.Array,
    masks: jax.Array,
    objective_weights: jax.Array,
    chunks: int = 1,
    *,
    normalizer: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Weighted objective with an immediate frozen-head backward.

    Returns the differentiable scalar objective and nondifferentiable per-mask
    loss metrics. Only hidden states receive gradients. Each chunk computes its
    loss and hidden-state cotangent together, retaining no vocabulary residuals
    and requiring no head recomputation in the outer backward.
    A supplied normalizer divides token sums for every mask; otherwise masks
    are normalized per example before averaging over the batch.
    """
    masks = masks[:, 1:].astype(jnp.float32)
    if normalizer is None:
        normalizer = (
            jnp.maximum(masks.sum(axis=1, keepdims=True), 1) * input_ids.shape[0]
        )
    weights = masks / jnp.where(normalizer > 0, normalizer, 1)
    shape = states.shape

    def evaluate(hidden: jax.Array, head_params: PyTree) -> Any:
        values = _loss_chunks(hidden, input_ids[:, 1:], weights, chunks)

        def body(totals: jax.Array, values: Any) -> Any:
            hidden, labels, weights = values

            def objective(hidden: jax.Array) -> Any:
                logits = model.apply(
                    {"params": head_params}, hidden, method=model.head
                ).astype(jnp.float32)
                target = jnp.take_along_axis(logits, labels[:, None], axis=-1)[:, 0]
                token_loss = jax.nn.logsumexp(logits, axis=-1) - target
                losses = jnp.sum(token_loss[:, None] * weights, axis=0)
                return jnp.sum(losses * objective_weights), losses

            (_, losses), gradient = jax.value_and_grad(objective, has_aux=True)(hidden)
            return totals + losses, gradient

        losses, gradients = jax.lax.scan(
            body, jnp.zeros(weights.shape[-1], jnp.float32), values
        )
        gradient = gradients.swapaxes(0, 1).reshape(-1, shape[-1])
        gradient = gradient[: shape[0] * shape[1]].reshape(shape)
        return (
            jnp.sum(losses * objective_weights),
            jax.lax.stop_gradient(losses),
        ), gradient

    @jax.custom_vjp
    def loss(hidden: jax.Array, head_params: PyTree) -> Any:
        return evaluate(hidden, head_params)[0]

    def backward(gradient: jax.Array, cotangent: Any) -> Any:
        return (gradient * cotangent[0]).astype(gradient.dtype), None

    loss.defvjp(evaluate, backward)
    return loss(states, {"lm_head": params["lm_head"]})


def lm_loss(
    model: CustomModel,
    params: PyTree,
    states: jax.Array,
    input_ids: jax.Array,
    pad_token_id: int,
    chunks: int = 1,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute globally token-normalized next-token loss and metrics."""
    labels = input_ids[:, 1:]
    mask = labels != pad_token_id
    weights = mask / jnp.maximum(mask.sum(), 1)
    loss, accuracy = chunked_lm_loss(model, params, states, labels, weights, chunks)
    return loss, {
        "lm_loss": loss,
        "lm_acc": accuracy,
        "atom_count": mask.sum(),
    }


def per_example_lm_losses(
    model: CustomModel,
    params: PyTree,
    states: jax.Array,
    input_ids: jax.Array,
    masks: jax.Array,
    chunks: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Compute one next-token loss per mask, weighting each example equally.

    ``masks`` may have shape ``[batch, sequence]`` or add a final objective
    dimension. Each mask is normalized within each example before the batch
    average, which is useful when examples contain different token counts.
    """
    labels = input_ids[:, 1:]
    single_mask = masks.ndim == input_ids.ndim
    masks = masks[:, 1:].astype(jnp.float32)
    if single_mask:
        masks = masks[..., None]
    elif masks.ndim != input_ids.ndim + 1:
        raise ValueError("masks must match input_ids or add one objective dimension")
    batch_size = input_ids.shape[0]
    weights = masks / (jnp.maximum(masks.sum(axis=1, keepdims=True), 1) * batch_size)
    losses, accuracy = chunked_lm_loss(model, params, states, labels, weights, chunks)
    if single_mask:
        return losses[0], accuracy[0]
    return losses, accuracy
