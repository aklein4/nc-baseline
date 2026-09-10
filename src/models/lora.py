"""LoRA projections and ephemeral per-example Adam state.

See docs/architecture.md#model-state.
The low-rank ingredient follows LoRA (https://arxiv.org/abs/2106.09685); the
episodic fast-state update is project-specific.
"""

import logging
import math
from dataclasses import dataclass
from functools import partial
from typing import Any, ClassVar, cast

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct

from models.llama import (
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    map_hf_params,
    rotary_embeddings,
    rotate_half,
)
from utils.attention_utils import flash_attention
from utils.checkpoints import ParameterCheckpoint
from utils.modules import RMSNorm
from utils.sharding_utils import with_sharding_constraint
from utils.typing_utils import PyTree

logger = logging.getLogger(__name__)


@partial(jax.jit, static_argnames=("rank",))
def _svd_split(kernels: jax.Array, rank: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Split kernels on device, with one FP32 SVD workspace at a time.

    https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.svd.html
    """

    def split(kernel):
        u, singular, vh = jnp.linalg.svd(kernel.T, full_matrices=False)
        root = jnp.sqrt(singular[:rank])
        down = (root[:, None] * vh[:rank]).T
        up = (u[:, :rank] * root[None]).T
        residual = kernel - jnp.matmul(down, up, precision=jax.lax.Precision.HIGHEST)
        return residual, down, up

    return jax.lax.map(split, kernels.astype(jnp.float32))


@dataclass(frozen=True)
class LoRAConfig(LlamaConfig):
    fast_weight_rank: int
    base_lr: float
    momentum_beta: float
    second_moment_beta: float | None
    grad_rms_eps: float

    @property
    def second_beta(self) -> float:
        return (
            math.sqrt(self.momentum_beta)
            if self.second_moment_beta is None
            else self.second_moment_beta
        )


@struct.dataclass
class LoRAState:
    value: PyTree
    momentum: PyTree
    second_moment: PyTree
    step: jax.Array


@jax.custom_vjp
def collect_fast_gradient(
    activations: jax.Array, output: jax.Array, buffer: jax.Array
) -> jax.Array:
    """Return ``output`` while routing its linear-weight gradient to ``buffer``."""
    del activations, buffer
    return output


def _collect_fwd(
    activations: jax.Array, output: jax.Array, buffer: jax.Array
) -> tuple[jax.Array, jax.Array]:
    del buffer
    return output, activations


def _collect_bwd(
    activations: jax.Array, output_grad: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    update = jnp.einsum(
        "blo,bli->boi", output_grad.astype(jnp.float32), activations.astype(jnp.float32)
    )
    return jnp.zeros_like(activations), output_grad, update


collect_fast_gradient.defvjp(_collect_fwd, _collect_bwd)


class LoRALinear(nn.Module):
    config: LoRAConfig
    in_features: int
    out_features: int
    kernel_std: float | None = None
    no_muon_patterns: ClassVar[tuple[str, ...]] = (
        "fast_down_log_lr",
        "fast_up_log_lr",
    )

    def setup(self) -> None:
        config = self.config
        rank = config.fast_weight_rank
        std = self.kernel_std or config.hidden_size**-0.5
        self.kernel = self.param(
            "kernel",
            nn.initializers.normal(std),
            (self.in_features, self.out_features),
            jnp.float32,
        )
        self.base_down = nn.Dense(
            rank,
            use_bias=False,
            dtype=config.compute_dtype,
            param_dtype=jnp.float32,
            kernel_init=nn.initializers.normal(self.in_features**-0.5),
            name="base_down",
        )
        self.base_up = nn.Dense(
            self.out_features,
            use_bias=False,
            dtype=config.compute_dtype,
            param_dtype=jnp.float32,
            kernel_init=nn.initializers.normal(rank**-0.5),
            name="base_up",
        )
        self.fast_down_log_lr = self.param(
            "fast_down_log_lr",
            nn.initializers.zeros,
            (rank, self.in_features),
            jnp.float32,
        )
        self.fast_up_log_lr = self.param(
            "fast_up_log_lr",
            nn.initializers.zeros,
            (self.out_features, rank),
            jnp.float32,
        )

    def _fast(
        self,
        x: jax.Array,
        state: jax.Array,
        buffer: jax.Array,
        log_lr: jax.Array,
    ) -> jax.Array:
        scale = self.config.base_lr / math.sqrt(state.shape[-1])
        learning_rate = scale * jnp.exp(log_lr * math.sqrt(state.shape[-1]))
        kernel = jax.lax.stop_gradient(state) * learning_rate[None]
        output = jnp.einsum(
            "bli,boi->blo",
            x.astype(self.config.compute_dtype),
            kernel.astype(self.config.compute_dtype),
        )
        return collect_fast_gradient(x, output, buffer)

    def __call__(self, x: jax.Array, state: PyTree, buffer: PyTree) -> jax.Array:
        base = jnp.einsum(
            "...i,io->...o",
            x.astype(self.config.compute_dtype),
            self.kernel.astype(self.config.compute_dtype),
        )
        hidden = self.base_down(x) + self._fast(
            x, state["down"], buffer["down"], self.fast_down_log_lr
        )
        return (
            base
            + self.base_up(hidden)
            + self._fast(hidden, state["up"], buffer["up"], self.fast_up_log_lr)
        )


def _projection_state(
    layers: int, batch: int, inputs: int, outputs: int, rank: int
) -> dict[str, jax.Array]:
    return {
        "down": jnp.zeros((layers, batch, rank, inputs), jnp.float32),
        "up": jnp.zeros((layers, batch, outputs, rank), jnp.float32),
    }


class LoRAAttention(nn.Module):
    config: LoRAConfig

    def setup(self) -> None:
        config = self.config
        h = config.hidden_size
        self.q_proj = LoRALinear(
            config, h, config.num_attention_heads * config.head_dim, name="q_proj"
        )
        self.k_proj = LoRALinear(
            config, h, config.num_key_value_heads * config.head_dim, name="k_proj"
        )
        self.v_proj = LoRALinear(
            config, h, config.num_key_value_heads * config.head_dim, name="v_proj"
        )
        self.o_proj = LoRALinear(config, h, h, name="o_proj")

    def __call__(
        self,
        x: jax.Array,
        state: PyTree,
        buffer: PyTree,
        attention_mask: jax.Array | None,
    ) -> jax.Array:
        config = self.config
        q = self.q_proj(x, state["q_proj"], buffer["q_proj"])
        k = self.k_proj(x, state["k_proj"], buffer["k_proj"])
        v = self.v_proj(x, state["v_proj"], buffer["v_proj"])
        q = q.reshape(*q.shape[:2], config.num_attention_heads, config.head_dim)
        k = k.reshape(*k.shape[:2], config.num_key_value_heads, config.head_dim)
        v = v.reshape(*v.shape[:2], config.num_key_value_heads, config.head_dim)

        cos, sin = rotary_embeddings(config, x.shape[1], config.compute_dtype)
        q = q * cos[None, :, None] + rotate_half(q) * sin[None, :, None]
        k = k * cos[None, :, None] + rotate_half(k) * sin[None, :, None]
        y = flash_attention(
            q,
            k,
            v,
            backend=config.attention_backend,
            mask=attention_mask,
        )
        y = y.reshape(*y.shape[:2], config.hidden_size)
        return self.o_proj(y, state["o_proj"], buffer["o_proj"])


class LoRAMLP(nn.Module):
    config: LoRAConfig

    def setup(self) -> None:
        config = self.config
        self.gate_proj = LoRALinear(
            config, config.hidden_size, config.intermediate_size, name="gate_proj"
        )
        self.up_proj = LoRALinear(
            config, config.hidden_size, config.intermediate_size, name="up_proj"
        )
        self.down_proj = LoRALinear(
            config,
            config.intermediate_size,
            config.hidden_size,
            kernel_std=config.intermediate_size**-0.5,
            name="down_proj",
        )

    def __call__(self, x: jax.Array, state: PyTree, buffer: PyTree) -> jax.Array:
        gate = self.gate_proj(x, state["gate_proj"], buffer["gate_proj"])
        up = self.up_proj(x, state["up_proj"], buffer["up_proj"])
        return self.down_proj(
            nn.silu(gate) * up, state["down_proj"], buffer["down_proj"]
        )


class LoRADecoderLayer(LlamaDecoderLayer):
    def setup(self) -> None:
        config = cast(LoRAConfig, self.config)
        self.input_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, name="input_layernorm"
        )
        self.self_attn = LoRAAttention(config, name="self_attn")
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, name="post_attention_layernorm"
        )
        self.mlp = LoRAMLP(config, name="mlp")

    def __call__(
        self,
        x: jax.Array,
        state: PyTree,
        buffer: PyTree,
        attention_mask: jax.Array | None,
    ) -> Any:
        residual = x
        x = self.input_layernorm(x)
        x = residual + self.self_attn(
            x, state["self_attn"], buffer["self_attn"], attention_mask
        )
        residual = x
        x = self.post_attention_layernorm(x)
        x = residual + self.mlp(x, state["mlp"], buffer["mlp"])
        return x, ()


class LoRAModel(LlamaForCausalLM):
    config: LoRAConfig
    config_type = LoRAConfig
    layer_type = LoRADecoderLayer

    def layer_in_axes(self) -> Any:
        return (0, 0, nn.broadcast)

    def get_init_inputs(
        self, batch_size: int = 1, sequence_length: int = 2
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        args, kwargs = super().get_init_inputs(batch_size, sequence_length)
        state = self.init_state(batch_size)
        kwargs.update(lora_state=state.value, grad_buffer=state.value)
        return args, kwargs

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        lora_state: PyTree | None = None,
        grad_buffer: PyTree | None = None,
        shift_states: bool = False,
        compute_logits: bool = True,
    ) -> jax.Array:
        if lora_state is None:
            lora_state = self.init_state(input_ids.shape[0]).value
        if grad_buffer is None:
            grad_buffer = jax.tree.map(jnp.zeros_like, lora_state)
        return self.forward(
            input_ids,
            (lora_state, grad_buffer, attention_mask),
            shift_states,
            compute_logits,
        )

    def init_state(self, batch_size: int) -> LoRAState:
        config = self.config
        n, h, f, r = (
            config.num_hidden_layers,
            config.hidden_size,
            config.intermediate_size,
            config.fast_weight_rank,
        )
        attention_outputs = {
            "q_proj": config.num_attention_heads * config.head_dim,
            "k_proj": config.num_key_value_heads * config.head_dim,
            "v_proj": config.num_key_value_heads * config.head_dim,
            "o_proj": h,
        }
        value = {
            "self_attn": {
                name: _projection_state(n, batch_size, h, output, r)
                for name, output in attention_outputs.items()
            },
            "mlp": {
                "gate_proj": _projection_state(n, batch_size, h, f, r),
                "up_proj": _projection_state(n, batch_size, h, f, r),
                "down_proj": _projection_state(n, batch_size, f, h, r),
            },
        }
        spec = [None, ["data", "fsdp"], None, None]
        value = jax.tree.map(lambda x: with_sharding_constraint(x, spec), value)
        zeros = jax.tree.map(jnp.zeros_like, value)
        return LoRAState(value, zeros, zeros, jnp.zeros((), jnp.int32))

    def update_state(self, state: LoRAState, update: PyTree) -> LoRAState:
        config = self.config
        beta1, beta2 = config.momentum_beta, config.second_beta
        step = state.step + 1
        momentum = jax.tree.map(
            lambda old, grad: beta1 * old + (1 - beta1) * grad.astype(jnp.float32),
            state.momentum,
            update,
        )
        second = jax.tree.map(
            lambda old, grad: beta2 * old + (1 - beta2) * grad.astype(jnp.float32) ** 2,
            state.second_moment,
            update,
        )
        correction1 = 1 - beta1 ** step.astype(jnp.float32)
        correction2 = 1 - beta2 ** step.astype(jnp.float32)
        value = jax.tree.map(
            lambda value, first, square: value.astype(jnp.float32)
            - (first / correction1)
            / (jnp.sqrt(square / correction2) + config.grad_rms_eps),
            state.value,
            momentum,
            second,
        )
        return LoRAState(value, momentum, second, step)

    def load_params(self, params: PyTree, checkpoint: ParameterCheckpoint) -> PyTree:
        if "model.embed_tokens.weight" in checkpoint:
            params = map_hf_params(params, self.config, checkpoint.read)
            return self._svd_initialize(params)
        return super().load_params(params, checkpoint)

    def _svd_initialize(self, params: PyTree) -> PyTree:
        """Split provider kernels on the active JAX device and keep them there."""
        rank = self.config.fast_weight_rank
        params = dict(params)
        layers = dict(params["layers"])
        for group_name in ("self_attn", "mlp"):
            group = dict(layers[group_name])
            for projection_name, projection in group.items():
                projection = dict(projection)
                kernels = jnp.asarray(projection["kernel"], dtype=jnp.float32)
                logger.info(
                    "LoRA SVD: %s.%s on %s",
                    group_name,
                    projection_name,
                    kernels.device,
                )
                residual, down, up = jax.block_until_ready(_svd_split(kernels, rank))
                projection["kernel"] = residual
                base_down = dict(projection["base_down"])
                base_up = dict(projection["base_up"])
                base_down["kernel"] = down
                base_up["kernel"] = up
                projection["base_down"] = base_down
                projection["base_up"] = base_up
                group[projection_name] = projection
            layers[group_name] = group
        params["layers"] = layers
        return params
