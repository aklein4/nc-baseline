"""Reusable Linen layers and parameter-initialization helpers.

See docs/components.md#linen-module-utilities and the Linen Module API:
https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html
"""

import math
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
from flax import linen as nn

from utils.array_utils import rms_norm


def linear(
    features: int,
    *,
    compute_dtype: Any,
    use_bias: bool = False,
    kernel_std: float | None = None,
    name: str | None = None,
) -> nn.Dense:
    """Construct a float32-parameter Dense with Piano's Gaussian scaling.

    ``variance_scaling(..., distribution="normal")`` is an untruncated normal
    with standard deviation ``1 / sqrt(fan_in)``, exactly the distribution used
    by ``outer-loop``'s ``gaussian_init``; only the PRNG stream differs.
    """
    if kernel_std is not None:
        init_fn = nn.initializers.normal(kernel_std)
    else:
        init_fn = nn.initializers.variance_scaling(
            scale=1.0,
            mode="fan_in",
            distribution="normal",
        )
    return nn.Dense(
        features,
        use_bias=use_bias,
        dtype=compute_dtype,
        param_dtype=jnp.float32,
        kernel_init=init_fn,
        name=name,
        # precision accumulates in fp32 by default on relevant platforms
    )


class RMSNorm(nn.Module):
    size: int
    eps: float

    def setup(self) -> None:
        self.scale = self.param(
            "scale", nn.initializers.ones, (self.size,), jnp.float32
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return (rms_norm(x.astype(jnp.float32), self.eps) * self.scale).astype(x.dtype)


class ResidualConvMixer(nn.Module):
    no_muon_patterns: ClassVar[tuple[str, ...]] = ("conv",)
    size: int
    kernel_size: int
    compute_dtype: Any

    def setup(self) -> None:
        self.conv = nn.Conv(
            self.size,
            (self.kernel_size,),
            padding="SAME",
            feature_group_count=self.size,
            use_bias=False,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
            kernel_init=nn.initializers.normal(
                0.1 / math.sqrt(self.size * self.kernel_size)
            ),
            name="conv",
        )

    def __call__(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        return x + self.conv(x * mask[..., None]) * math.sqrt(self.size)


class SoftPass(nn.Module):
    pool_size: int
    compute_dtype: Any

    def setup(self) -> None:
        self.w_proj = linear(
            self.pool_size,
            compute_dtype=self.compute_dtype,
            use_bias=True,
            name="w_proj",
        )
        self.v_proj = linear(
            self.pool_size,
            compute_dtype=self.compute_dtype,
            use_bias=True,
            name="v_proj",
        )
        self.r_proj = linear(
            self.pool_size,
            compute_dtype=self.compute_dtype,
            name="r_proj",
        )
        self.g_proj = linear(
            self.pool_size,
            compute_dtype=self.compute_dtype,
            use_bias=True,
            name="g_proj",
        )

    def __call__(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        weights = jnp.where(mask[..., None], self.w_proj(x), -100.0)
        weights = jax.nn.softmax(weights, axis=-2)
        pooled = jnp.sum(weights * self.v_proj(x), axis=-2, keepdims=True)
        pooled = self.r_proj(pooled)
        pooled = rms_norm(pooled.astype(jnp.float32), 1e-7).astype(x.dtype)
        gate = 2 * jax.nn.sigmoid(self.g_proj(x))
        return gate * pooled
