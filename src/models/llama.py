"""Provider-compatible Llama model and Hugging Face checkpoint mapping.

Repository guides:
- docs/architecture.md#initialization-and-checkpoints
- docs/components.md#model-abstraction

Upstream sources:
- https://arxiv.org/abs/2407.21783
- https://huggingface.co/meta-llama/Llama-3.2-1B
- https://github.com/huggingface/transformers/tree/main/src/transformers/models/llama
"""

import math
from dataclasses import dataclass, fields
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.core import unfreeze
from omegaconf import DictConfig

from models.base import CustomModel, ModelConfig
from utils.attention_utils import flash_attention
from utils.checkpoints import ParameterCheckpoint
from utils.modules import RMSNorm, linear
from utils.typing_utils import PyTree


@dataclass(frozen=True)
class RopeConfig:
    theta: float
    factor: float
    low_freq_factor: float
    high_freq_factor: float
    original_context_len: int


@dataclass(frozen=True)
class LlamaConfig(ModelConfig):
    vocab_size: int
    pad_token_id: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rope: RopeConfig
    rms_norm_eps: float
    compute_dtype: str
    gradient_checkpointing: bool
    attention_backend: str

    @classmethod
    def from_config(cls, config: DictConfig) -> "LlamaConfig":
        values = {
            item.name: config[item.name] for item in fields(cls) if item.name in config
        }
        values["rope"] = RopeConfig(**config.rope)
        return cls(**values)


def map_hf_params(
    params: PyTree,
    config: LlamaConfig,
    read: Any,
) -> PyTree:
    """Map official Hugging Face Llama weights into the scanned Linen tree."""
    params = unfreeze(params)
    params["embed_tokens"]["embedding"] = read("model.embed_tokens.weight", False)
    params["norm"]["scale"] = read("model.norm.weight", False)
    layers = params["layers"]

    def stack(suffix: str, transpose: bool = False) -> np.ndarray:
        return np.stack(
            [
                read(f"model.layers.{i}.{suffix}", transpose)
                for i in range(config.num_hidden_layers)
            ]
        )

    layers["input_layernorm"]["scale"] = stack("input_layernorm.weight")
    layers["post_attention_layernorm"]["scale"] = stack(
        "post_attention_layernorm.weight"
    )
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        layers["self_attn"][projection]["kernel"] = stack(
            f"self_attn.{projection}.weight", transpose=True
        )
    for projection in ("gate_proj", "up_proj", "down_proj"):
        layers["mlp"][projection]["kernel"] = stack(
            f"mlp.{projection}.weight", transpose=True
        )
    try:
        params["lm_head"]["kernel"] = read("lm_head.weight", True)
    except KeyError:
        params["lm_head"]["kernel"] = read("model.embed_tokens.weight", True)
    return params


def llama3_rope_frequencies(config: LlamaConfig) -> jax.Array:
    """Llama 3 RoPE from the official Transformers implementation.

    https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py
    """
    rope = config.rope
    inv = 1.0 / (
        rope.theta
        ** (jnp.arange(0, config.head_dim, 2, dtype=jnp.float32) / config.head_dim)
    )
    wavelength = 2 * math.pi / inv
    low = rope.original_context_len / rope.low_freq_factor
    high = rope.original_context_len / rope.high_freq_factor
    smooth = (rope.original_context_len / wavelength - rope.low_freq_factor) / (
        rope.high_freq_factor - rope.low_freq_factor
    )
    scaled = (1 - smooth) * inv / rope.factor + smooth * inv
    return jnp.where(
        wavelength < high,
        inv,
        jnp.where(wavelength > low, inv / rope.factor, scaled),
    )


def rotary_embeddings(
    config: LlamaConfig, length: int, dtype: jnp.dtype
) -> tuple[jax.Array, jax.Array]:
    positions = jnp.arange(length, dtype=jnp.float32)
    freqs = jnp.einsum("t,d->td", positions, llama3_rope_frequencies(config))
    emb = jnp.concatenate((freqs, freqs), axis=-1)
    return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)


def rotate_half(x: jax.Array) -> jax.Array:
    """Rotate paired RoPE features by 90 degrees."""
    first, second = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-second, first), axis=-1)


class LlamaAttention(nn.Module):
    config: LlamaConfig

    def setup(self) -> None:
        config = self.config
        self.q_proj = linear(
            config.num_attention_heads * config.head_dim,
            compute_dtype=config.compute_dtype,
            name="q_proj",
        )
        self.k_proj = linear(
            config.num_key_value_heads * config.head_dim,
            compute_dtype=config.compute_dtype,
            name="k_proj",
        )
        self.v_proj = linear(
            config.num_key_value_heads * config.head_dim,
            compute_dtype=config.compute_dtype,
            name="v_proj",
        )
        self.o_proj = linear(
            config.hidden_size, compute_dtype=config.compute_dtype, name="o_proj"
        )

    def __call__(
        self, x: jax.Array, attention_mask: jax.Array | None = None
    ) -> jax.Array:
        config = self.config
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
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
        return self.o_proj(y)


class LlamaMLP(nn.Module):
    config: LlamaConfig

    def setup(self) -> None:
        config = self.config
        self.gate_proj = linear(
            config.intermediate_size,
            compute_dtype=config.compute_dtype,
            name="gate_proj",
        )
        self.up_proj = linear(
            config.intermediate_size, compute_dtype=config.compute_dtype, name="up_proj"
        )
        self.down_proj = linear(
            config.hidden_size,
            compute_dtype=config.compute_dtype,
            kernel_std=config.intermediate_size**-0.5,
            name="down_proj",
        )

    def __call__(self, x: jax.Array, **_: Any) -> jax.Array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    config: LlamaConfig
    mlp_type: ClassVar[type[nn.Module]] = LlamaMLP

    def setup(self) -> None:
        config = self.config
        self.input_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, name="input_layernorm"
        )
        self.self_attn = LlamaAttention(config, name="self_attn")
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, name="post_attention_layernorm"
        )
        self.mlp = self.mlp_type(config, **self.mlp_options(), name="mlp")

    def mlp_options(self) -> dict[str, Any]:
        return {}

    def forward(
        self,
        x: jax.Array,
        attention_mask: jax.Array | None,
        **mlp_inputs: Any,
    ) -> jax.Array:
        residual = x
        x = self.input_layernorm(x)
        x = residual + self.self_attn(x, attention_mask)

        residual = x
        x = self.post_attention_layernorm(x)
        return residual + self.mlp(x, **mlp_inputs)

    def __call__(self, x: jax.Array, attention_mask: jax.Array | None) -> Any:
        return self.forward(x, attention_mask), ()


class LlamaForCausalLM(CustomModel):
    config: LlamaConfig
    config_type: ClassVar[type[LlamaConfig]] = LlamaConfig
    layer_type: ClassVar[type[nn.Module]] = LlamaDecoderLayer
    no_muon_patterns: ClassVar[tuple[str, ...]] = ("embed_tokens", "lm_head")

    def setup(self) -> None:
        config = self.config
        self.embed_tokens = nn.Embed(
            config.vocab_size,
            config.hidden_size,
            dtype=config.compute_dtype,
            param_dtype=jnp.float32,
            embedding_init=nn.initializers.normal(1.0),
            name="embed_tokens",
        )
        Layer = self.layer_type
        if config.gradient_checkpointing:
            Layer = nn.remat(Layer, prevent_cse=False)
        Layers = nn.scan(
            Layer,
            variable_axes={"params": 0, "intermediates": 0},
            split_rngs={"params": True},
            in_axes=self.layer_in_axes(),
            out_axes=0,
            length=config.num_hidden_layers,
            unroll=1,
        )
        self.layers = Layers(config, **self.layer_options(), name="layers")
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, name="norm")
        self.lm_head = linear(
            config.vocab_size, compute_dtype=config.compute_dtype, name="lm_head"
        )

    def get_init_inputs(
        self, batch_size: int = 1, sequence_length: int = 2
    ) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
        return (np.ones((batch_size, sequence_length), np.int32),), {}

    def layer_in_axes(self) -> Any:
        return (nn.broadcast,)

    def layer_options(self) -> dict[str, Any]:
        return {}

    def load_params(self, params: PyTree, checkpoint: ParameterCheckpoint) -> PyTree:
        if "model.embed_tokens.weight" in checkpoint:
            return map_hf_params(params, self.config, checkpoint.read)
        return super().load_params(params, checkpoint)

    def embed(self, input_ids: jax.Array) -> jax.Array:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: jax.Array,
        layer_inputs: tuple[Any, ...],
        shift_states: bool = False,
        compute_logits: bool = True,
    ) -> jax.Array:
        x = self.embed(input_ids)
        x, _ = self.layers(x, *layer_inputs)
        x = self.norm(x)
        if shift_states:
            x = x[:, :-1]
        if not compute_logits:
            return x
        return self.head(x)

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        shift_states: bool = False,
        compute_logits: bool = True,
    ) -> jax.Array:
        return self.forward(input_ids, (attention_mask,), shift_states, compute_logits)

    def head(self, states: jax.Array) -> jax.Array:
        return self.lm_head(states)
