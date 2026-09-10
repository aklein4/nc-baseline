"""Portable and accelerator-specific exact attention backends.

See docs/components.md#attention-utilities. Algorithm and upstream sources:
- https://arxiv.org/abs/2205.14135
- https://docs.jax.dev/en/latest/_autosummary/jax.nn.dot_product_attention.html
- https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py
- https://github.com/ROCm/jax-aiter
"""

import math
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P


def _attention_mask_blocks(
    mask: jax.Array,
    batch: int,
    heads: int,
    length: int,
    padded: int,
    block_size: int,
) -> jax.Array:
    mask = jnp.asarray(mask)
    if mask.dtype != jnp.bool_:
        raise TypeError(
            "attention mask must be boolean, with True entries allowed to attend"
        )
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[:, None]
    elif mask.ndim != 4:
        raise ValueError(
            "attention mask must have shape [q, k], [batch, q, k], or [batch, heads, q, k]"
        )
    try:
        mask = jnp.broadcast_to(mask, (batch, heads, length, length))
    except ValueError as error:
        raise ValueError(
            f"attention mask shape {mask.shape} cannot broadcast to {(batch, heads, length, length)}"
        ) from error
    mask = jnp.pad(mask, ((0, 0), (0, 0), (0, padded - length), (0, padded - length)))
    blocks = padded // block_size
    mask = mask.reshape(batch, heads, blocks, block_size, blocks, block_size)
    return mask.transpose(2, 4, 0, 1, 3, 5)


def _portable_flash_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    block_size: int,
    causal: bool,
    mask: jax.Array | None,
) -> jax.Array:
    """Backend-neutral blockwise FlashAttention with an online softmax.

    This is Algorithm 1 from the official FlashAttention paper, expressed as
    JAX scans so it never materializes the full attention matrix.
    https://arxiv.org/abs/2205.14135
    """
    batch, length, heads, head_dim = query.shape
    key_heads = key.shape[2]
    if heads % key_heads:
        raise ValueError("query heads must be divisible by key/value heads")
    if heads != key_heads:
        repeats = heads // key_heads
        key = jnp.repeat(key, repeats, axis=2)
        value = jnp.repeat(value, repeats, axis=2)

    padded = math.ceil(length / block_size) * block_size
    pad = ((0, 0), (0, padded - length), (0, 0), (0, 0))
    query = jnp.pad(query, pad)
    key = jnp.pad(key, pad)
    value = jnp.pad(value, pad)
    blocks = padded // block_size

    def split(x: jax.Array) -> jax.Array:
        x = x.transpose(0, 2, 1, 3).reshape(batch, heads, blocks, block_size, head_dim)
        return x.transpose(2, 0, 1, 3, 4)

    query, key, value = split(query), split(key), split(value)
    positions = jnp.arange(padded).reshape(blocks, block_size)
    mask_blocks = (
        _attention_mask_blocks(mask, batch, heads, length, padded, block_size)
        if mask is not None
        else None
    )
    scale = head_dim**-0.5

    def query_body(_: Any, values: Any) -> Any:
        if mask_blocks is None:
            q, q_positions = values
            custom_masks = None
        else:
            q, q_positions, custom_masks = values
        maximum = jnp.full((batch, heads, block_size), -jnp.inf, jnp.float32)
        denominator = jnp.zeros_like(maximum)
        numerator = jnp.zeros((*maximum.shape, head_dim), jnp.float32)

        def key_body(carry: Any, values: Any) -> Any:
            maximum, denominator, numerator = carry
            if custom_masks is None:
                k, v, k_positions = values
                custom_mask = None
            else:
                k, v, k_positions, custom_mask = values
            scores = (
                jnp.einsum(
                    "bnqd,bnkd->bnqk", q.astype(jnp.float32), k.astype(jnp.float32)
                )
                * scale
            )
            block_mask = k_positions[None, :] < length
            if causal:
                block_mask &= q_positions[:, None] >= k_positions[None, :]
            block_mask = block_mask[None, None]
            if custom_mask is not None:
                block_mask &= custom_mask
            scores = jnp.where(block_mask, scores, -jnp.inf)
            block_maximum = jnp.max(scores, axis=-1)
            new_maximum = jnp.maximum(maximum, block_maximum)
            safe_maximum = jnp.where(jnp.isfinite(new_maximum), new_maximum, 0)
            correction = jnp.where(
                jnp.isfinite(maximum), jnp.exp(maximum - safe_maximum), 0
            )
            probability = jnp.where(
                block_mask, jnp.exp(scores - safe_maximum[..., None]), 0
            )
            denominator = correction * denominator + jnp.sum(probability, axis=-1)
            numerator = correction[..., None] * numerator + jnp.einsum(
                "bnqk,bnkd->bnqd", probability, v.astype(jnp.float32)
            )
            return (new_maximum, denominator, numerator), None

        key_values = (
            (key, value, positions)
            if custom_masks is None
            else (key, value, positions, custom_masks)
        )
        (_, denominator, numerator), _ = jax.lax.scan(
            key_body, (maximum, denominator, numerator), key_values
        )
        output = numerator / jnp.maximum(denominator[..., None], 1e-20)
        return None, output.astype(q.dtype)

    query_values = (
        (query, positions) if mask_blocks is None else (query, positions, mask_blocks)
    )
    _, output = jax.lax.scan(query_body, None, query_values)
    output = output.transpose(1, 0, 3, 2, 4).reshape(batch, padded, heads, head_dim)
    return output[:, :length]


def flash_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    backend: str = "auto",
    block_size: int = 128,
    causal: bool = True,
    mask: jax.Array | None = None,
) -> jax.Array:
    """Select fused attention, using portable attention for a custom boolean mask."""
    platform = jax.default_backend()
    device_kind = jax.devices()[0].device_kind.lower()
    if mask is not None:
        backend = "portable"
    elif backend == "auto":
        if (
            platform == "tpu"
            and query.shape[1] % block_size == 0
            and key.shape[1] % block_size == 0
        ):
            backend = "tpu"
        elif (
            platform == "gpu"
            and "nvidia" in device_kind
            and query.dtype
            in (
                jnp.bfloat16,
                jnp.float16,
            )
        ):
            backend = "cudnn"
        elif platform == "gpu":
            try:
                from jax_aiter.mha import flash_attn_func

                backend = "aiter"
            except ImportError:
                backend = "portable"
        else:
            backend = "portable"

    if backend == "cudnn":
        # JAX routes this to cuDNN FlashAttention.
        # https://docs.jax.dev/en/latest/_autosummary/jax.nn.dot_product_attention.html
        return jax.nn.dot_product_attention(
            query, key, value, is_causal=causal, implementation="cudnn"
        )
    if backend == "tpu":
        # Official JAX Pallas TPU FlashAttention implementation.
        # https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py
        from jax.experimental.pallas.ops.tpu import flash_attention as tpu_flash

        def attention(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
            block_sizes = None
            if q.shape[1] % 512 == 0 and k.shape[1] % 512 == 0:
                block_sizes = tpu_flash.BlockSizes(
                    block_q=512,
                    block_k_major=512,
                    block_k=512,
                    block_b=min(2, q.shape[0]),
                    block_q_major_dkv=512,
                    block_k_major_dkv=512,
                    block_q_dkv=512,
                    block_k_dkv=512,
                    block_q_dq=512,
                    block_k_dq=256,
                    block_k_major_dq=512,
                )
            if q.shape[2] != k.shape[2]:
                repeats = q.shape[2] // k.shape[2]
                k = jnp.repeat(k, repeats, axis=2)
                v = jnp.repeat(v, repeats, axis=2)
            return tpu_flash.flash_attention(
                q.transpose(0, 2, 1, 3),
                k.transpose(0, 2, 1, 3),
                v.transpose(0, 2, 1, 3),
                causal=causal,
                sm_scale=q.shape[-1] ** -0.5,
                block_sizes=block_sizes,
            ).transpose(0, 2, 1, 3)

        mesh = jax.sharding.get_abstract_mesh()
        if mesh.empty:
            return attention(query, key, value)

        # Pallas kernels cannot be automatically partitioned. Attention has no
        # cross-example communication, so invoke one kernel per batch shard and
        # leave its sequence, head, and feature axes local.
        batch_spec = P(tuple(mesh.axis_names), None, None, None)
        return jax.shard_map(
            attention,
            mesh=mesh,
            in_specs=(batch_spec, batch_spec, batch_spec),
            out_specs=batch_spec,
            check_vma=False,
        )(query, key, value)
    if backend == "aiter":
        # AMD's official ROCm FlashAttention binding, when its architecture-
        # specific wheel is installed. https://github.com/ROCm/jax-aiter
        from jax_aiter.mha import flash_attn_func

        return flash_attn_func(query, key, value, causal=causal)
    if backend == "portable":
        if query.shape[1] != key.shape[1] and mask is None and not causal:
            return jax.nn.dot_product_attention(
                query, key, value, is_causal=False, implementation="xla"
            )
        return _portable_flash_attention(query, key, value, block_size, causal, mask)
    raise ValueError(f"unknown attention backend: {backend}")
