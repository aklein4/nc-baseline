"""Small, model-independent JAX array and gradient helpers.

See docs/components.md#tree-and-array-utilities.
JAX custom derivatives:
https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
Newton--Schulz coefficients: https://kellerjordan.github.io/posts/muon/
"""

import math
from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp


@jax.custom_jvp
def scale_gradient(x: jax.Array, scale: jax.Array | float) -> jax.Array:
    """Return ``x`` unchanged while scaling its gradient."""
    return x


@scale_gradient.defjvp
def _scale_gradient_jvp(
    primals: tuple[jax.Array, jax.Array | float],
    tangents: tuple[jax.Array, jax.Array | float],
) -> tuple[jax.Array, jax.Array]:
    x, scale = primals
    x_tangent, _ = tangents
    return x, x_tangent * scale


def attach_gradient(real: jax.Array, ghost: jax.Array) -> jax.Array:
    """Return ``real`` while sending its gradient to both inputs."""

    @jax.custom_vjp
    def identity(real_value: jax.Array, ghost_value: jax.Array) -> jax.Array:
        return real_value

    def forward(
        real_value: jax.Array, ghost_value: jax.Array
    ) -> tuple[jax.Array, None]:
        return real_value, None

    def backward(_: None, gradient: jax.Array) -> tuple[jax.Array, jax.Array]:
        return gradient, gradient

    identity.defvjp(forward, backward)
    return identity(real, ghost)


def transform_gradient(
    x: jax.Array, fn: Callable[[jax.Array, jax.Array], jax.Array]
) -> jax.Array:
    """Return ``x`` while transforming its cotangent with ``fn(x, gradient)``."""

    @jax.custom_vjp
    def identity(value: jax.Array) -> jax.Array:
        return value

    def forward(value: jax.Array) -> tuple[jax.Array, jax.Array]:
        return value, value

    def backward(value: jax.Array, gradient: jax.Array) -> tuple[jax.Array]:
        return (fn(value, gradient),)

    identity.defvjp(forward, backward)
    return identity(x)


def print_gradient(x: jax.Array, name: str = "gradient") -> jax.Array:
    """Return ``x`` and print its gradient when reverse mode reaches it."""
    return transform_gradient(x, lambda _, gradient: _print_gradient(gradient, name))


def _print_gradient(gradient: jax.Array, name: str) -> jax.Array:
    jax.debug.print("{}: {}", name, gradient)
    return gradient


def unsqueeze_to_batch(x: jax.Array, target: jax.Array) -> jax.Array:
    """Add leading singleton axes until ``x`` has ``target.ndim`` axes."""
    return jnp.reshape(x, (1,) * (target.ndim - x.ndim) + x.shape)


def unsqueeze_to_channel(x: jax.Array, target: jax.Array) -> jax.Array:
    """Add trailing singleton axes until ``x`` has ``target.ndim`` axes."""
    return jnp.reshape(x, x.shape + (1,) * (target.ndim - x.ndim))


def expand_to_batch(x: jax.Array, target: jax.Array) -> jax.Array:
    """Broadcast ``x`` across the leading axes of ``target``."""
    leading = target.ndim - x.ndim
    x = unsqueeze_to_batch(x, target)
    return jnp.broadcast_to(x, target.shape[:leading] + x.shape[leading:])


def expand_to_channel(x: jax.Array, target: jax.Array) -> jax.Array:
    """Broadcast ``x`` across the trailing axes of ``target``."""
    trailing = target.ndim - x.ndim
    if not trailing:
        return x
    x = unsqueeze_to_channel(x, target)
    return jnp.broadcast_to(x, x.shape[:-trailing] + target.shape[-trailing:])


def shift(
    x: jax.Array,
    n: int,
    axis: int,
    direction: Literal["left", "right"],
    narrow: bool = True,
) -> jax.Array:
    """Shift an array with zero padding, optionally preserving its size."""
    padding = [(0, 0)] * x.ndim
    if direction == "right":
        padding[axis] = (n, 0)
        x = jax.lax.slice_in_dim(x, 0, x.shape[axis] - n, axis=axis) if narrow else x
    elif direction == "left":
        padding[axis] = (0, n)
        x = jax.lax.slice_in_dim(x, n, x.shape[axis], axis=axis) if narrow else x
    else:
        raise ValueError(f"invalid direction: {direction}")
    return jnp.pad(x, padding)


def inv_softplus(x: jax.Array) -> jax.Array:
    """Evaluate the inverse of softplus for positive ``x``."""
    return jnp.log(jnp.expm1(x))


def inv_unit_softplus(x: jax.Array) -> jax.Array:
    """Evaluate the inverse of unit softplus for positive ``x``."""
    return jnp.log(jnp.expm1(x * math.log(2)))


def unit_softplus(x: jax.Array) -> jax.Array:
    """Evaluate softplus scaled so that zero maps to one."""
    return jax.nn.softplus(x) / math.log(2)


def linear_warmup(step: jax.Array | int, warmup_steps: int) -> jax.Array:
    """Return a linear zero-to-one warmup multiplier."""
    return jnp.clip(step / warmup_steps, 0, 1)


def cosine_warmup(step: jax.Array | int, warmup_steps: int) -> jax.Array:
    """Return a half-cosine zero-to-one warmup multiplier."""
    return 0.5 * (1 - jnp.cos(math.pi * linear_warmup(step, warmup_steps)))


def rms_norm(x: jax.Array, eps: float) -> jax.Array:
    """Normalize the final axis of ``x`` by its root mean square."""
    return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)


def sequence_rms(x: jax.Array, valid_count: jax.Array | int, eps: float) -> jax.Array:
    """RMS-normalize the penultimate axis using an explicit valid-item count."""
    return x * jax.lax.rsqrt(
        jnp.sum(jnp.square(x), axis=-2, keepdims=True) / valid_count + eps**2
    )


def nudge(x: jax.Array, direction: jax.Array) -> jax.Array:
    """Move ``x`` a bounded amount in ``direction`` without changing its shape."""
    return x + jnp.tanh(jnp.abs(x)) * jnp.tanh(direction)


def slerp(
    start: jax.Array,
    end: jax.Array,
    amount: jax.Array | float,
    eps: float = 1e-7,
) -> jax.Array:
    """Spherically interpolate along the final dimension."""
    dtype = start.dtype
    start, end = start.astype(jnp.float32), end.astype(jnp.float32)
    start_norm = jnp.linalg.norm(start, axis=-1, keepdims=True)
    end_norm = jnp.linalg.norm(end, axis=-1, keepdims=True)
    start_unit = start / jnp.maximum(start_norm, eps)
    end_unit = end / jnp.maximum(end_norm, eps)
    amount = jnp.asarray(amount, jnp.float32)
    amount = unsqueeze_to_batch(amount[..., None], start)
    theta = jnp.arccos(
        jnp.clip(
            jnp.sum(start_unit * end_unit, axis=-1, keepdims=True), -1 + eps, 1 - eps
        )
    )
    sine = jnp.sin(theta)
    spherical = (
        start_unit * jnp.sin((1 - amount) * theta) + end_unit * jnp.sin(amount * theta)
    ) / jnp.maximum(sine, eps)
    linear = (1 - amount) * start_unit + amount * end_unit
    direction = jnp.where(jnp.abs(sine) > eps, spherical, linear)
    direction /= jnp.maximum(jnp.linalg.norm(direction, axis=-1, keepdims=True), eps)
    norm = (1 - amount) * start_norm + amount * end_norm
    return (direction * norm).astype(dtype)


NS_POLAR_COEFFICIENTS = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
)
NS_STANDARD_COEFFICIENTS = ((3.4445, -4.7750, 2.0315),) * 8


def newton_schulz(
    matrix: jax.Array,
    steps: int = 5,
    eps: float = 1e-7,
    polar: bool = False,
    safety: float | None = None,
) -> jax.Array:
    """Apply Muon-style Newton--Schulz whitening with one compiled scan."""
    if matrix.ndim < 2:
        raise ValueError("newton_schulz requires an array with at least two dimensions")
    coefficients = NS_POLAR_COEFFICIENTS if polar else NS_STANDARD_COEFFICIENTS
    if not 0 <= steps <= len(coefficients):
        raise ValueError(f"steps must be between 0 and {len(coefficients)}")
    if safety is None:
        safety = 0.5 if polar else 1.0

    transposed = matrix.shape[-2] > matrix.shape[-1]
    x = jnp.swapaxes(matrix, -2, -1) if transposed else matrix
    x_float = x.astype(jnp.float32)
    x = (
        x_float / (jnp.linalg.norm(x_float, axis=(-2, -1), keepdims=True) + eps)
    ).astype(x.dtype)
    x = x * safety

    def body(value: jax.Array, coefficients: jax.Array) -> tuple[jax.Array, None]:
        a_coefficient, b_coefficient, c_coefficient = coefficients
        gram = value @ jnp.swapaxes(value, -2, -1)
        value = (
            a_coefficient * value
            + (b_coefficient * gram + c_coefficient * gram @ gram) @ value
        )
        return value, None

    coefficients = jnp.asarray(coefficients[:steps], dtype=x.dtype)
    x, _ = jax.lax.scan(body, x, coefficients)
    return jnp.swapaxes(x, -2, -1) if transposed else x
