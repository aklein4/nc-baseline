"""Compiled language-model training step.

References:
- docs/architecture.md#trainers-and-compiled-steps
- https://docs.jax.dev/en/latest/_autosummary/jax.jit.html
"""

from typing import ClassVar

import jax
import jax.numpy as jnp

from trainers.base_trainer import BaseTrainer
from utils.losses import lm_loss
from utils.typing_utils import PyTree


class LMTrainer(BaseTrainer):
    required_config_keys: ClassVar[list[str]] = BaseTrainer.required_config_keys + [
        "num_logit_iterations",
        "optimizer._target_",
        "optimizer.lr",
    ]

    def loss(
        self, params: PyTree, batch: dict[str, jax.Array]
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        input_ids = batch["input_ids"]
        pad = self.model.config.pad_token_id
        model_ids = jnp.where(input_ids == pad, 0, input_ids)
        states = self.model.apply(
            {"params": params},
            model_ids,
            shift_states=True,
            compute_logits=False,
        )
        return lm_loss(
            self.model,
            params,
            states,
            input_ids,
            pad,
            int(self.config.num_logit_iterations),
        )
