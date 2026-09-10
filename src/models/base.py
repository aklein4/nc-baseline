"""Shared typed lifecycle for Flax Linen models.

See docs/components.md#model-abstraction and the Linen module reference:
https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html
"""

from dataclasses import dataclass, fields
from typing import Any, ClassVar, Self

import jax
from flax import linen as nn
from omegaconf import DictConfig

from utils.checkpoints import ParameterCheckpoint
from utils.typing_utils import PyTree


@dataclass(frozen=True)
class ModelConfig:
    """Typed, immutable model configuration."""

    @classmethod
    def from_config(cls, config: DictConfig) -> Self:
        values = {
            item.name: config[item.name] for item in fields(cls) if item.name in config
        }
        return cls(**values)


class CustomModel(nn.Module):
    """Small shared lifecycle for trainable Linen models."""

    config: ModelConfig
    config_type: ClassVar[type[ModelConfig]] = ModelConfig

    def configure_environment(self) -> None:
        """Set backend environment options before JAX is initialized."""

    def get_init_inputs(self, **kwargs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Return representative inputs used to initialize all parameters."""
        raise NotImplementedError

    def initialize_params(self, seed: int, **kwargs: Any) -> PyTree:
        """Initialize every parameter locally."""
        args, init_kwargs = self.get_init_inputs(**kwargs)
        return self.init(jax.random.key(seed), *args, **init_kwargs)["params"]

    def load_params(self, params: PyTree, checkpoint: ParameterCheckpoint) -> PyTree:
        """Load parameters from a resolved checkpoint."""
        return checkpoint.restore(params)
