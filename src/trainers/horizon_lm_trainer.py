"""Stateless Llama baseline for episodic Horizons batches.

References:
- docs/architecture.md#trainers-and-compiled-steps
- https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html
"""

from typing import Any, ClassVar

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from trainers.base_trainer import BaseTrainer
from utils.losses import chunked_lm_loss
from utils.mesh_utils import get_batch_sharding
from utils.optimizer_utils import FROZEN
from utils.tree_utils import tree_add, tree_labels, tree_zeros
from utils.typing_utils import PyTree


def episodic_loss_metrics(
    losses: tuple[jax.Array, jax.Array],
    assistant_mask: jax.Array,
    valid_mask: jax.Array,
    aux_loss_weight: float,
) -> dict[str, jax.Array]:
    """Renormalize globally scaled token sums for episode and decade reports."""
    assistant_losses, auxiliary_losses = losses
    assistant_counts = assistant_mask[..., 1:].astype(jnp.float32).sum(axis=(0, 2))
    auxiliary_counts = valid_mask[..., 1:].sum(axis=(0, 2)) - assistant_counts
    counts = assistant_counts + aux_loss_weight * auxiliary_counts
    normalizer = jnp.where(counts.sum() > 0, counts.sum(), 1)
    assistant_sums = assistant_losses * normalizer
    auxiliary_sums = auxiliary_losses * normalizer

    def mean(sums: jax.Array, counts: jax.Array) -> jax.Array:
        return sums / jnp.where(counts > 0, counts, 1)

    metrics = {
        "loss": (assistant_losses + aux_loss_weight * auxiliary_losses).sum(),
        "total_loss": mean(assistant_sums.sum(), assistant_counts.sum()),
        "total_aux_loss": mean(auxiliary_sums.sum(), auxiliary_counts.sum()),
        "atom_count": valid_mask.sum(),
    }

    episode_count = assistant_losses.shape[0]
    for index in range(episode_count):
        metrics[f"lm_loss/episode_{index:02d}"] = mean(
            assistant_sums[index], assistant_counts[index]
        )
        metrics[f"aux_loss/episode_{index:02d}"] = mean(
            auxiliary_sums[index], auxiliary_counts[index]
        )

    for decade in range((episode_count - 1) // 10 + 1):
        start = max(1, decade * 10)
        stop = min(episode_count, (decade + 1) * 10)
        if start >= stop:
            continue
        metrics[f"grouped_lm_loss/decade_{decade:02d}"] = mean(
            assistant_sums[start:stop].sum(), assistant_counts[start:stop].sum()
        )
        metrics[f"grouped_aux_loss/decade_{decade:02d}"] = mean(
            auxiliary_sums[start:stop].sum(), auxiliary_counts[start:stop].sum()
        )
    return metrics


class HorizonLMTrainer(BaseTrainer):
    """Train independent episodes without a recurrent or fast-weight state."""

    required_config_keys: ClassVar[list[str]] = BaseTrainer.required_config_keys + [
        "num_logit_iterations",
        "aux_loss_weight",
        "groups.slow._target_",
        "groups.slow.lr",
    ]

    def __init__(self, model: Any, config: Any, params: PyTree, mesh: Mesh) -> None:
        labels = tree_labels(
            params,
            {FROZEN: ("embed_tokens", "lm_head")},
            "slow",
        )
        super().__init__(model, config, params, labels, mesh=mesh)

    def _episode_loss(
        self, params: PyTree, batch: dict[str, jax.Array]
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        states = self.model.apply(
            {"params": params},
            batch["input_ids"],
            shift_states=True,
            compute_logits=False,
        )
        assistant = batch["assistant_mask"].astype(jnp.float32)
        valid = batch["attention_mask"].astype(jnp.float32)
        losses, _ = chunked_lm_loss(
            self.model,
            params,
            states,
            batch["input_ids"][:, 1:],
            jnp.stack((assistant, valid - assistant), axis=-1)[:, 1:]
            / batch["loss_normalizer"],
            int(self.config.num_logit_iterations),
        )
        assistant_loss, auxiliary_loss = losses
        loss = assistant_loss + float(self.config.aux_loss_weight) * auxiliary_loss
        return loss, (assistant_loss, auxiliary_loss)

    def _local_grad_accum(
        self,
        params: PyTree,
        normalizer: jax.Array,
        input_ids: jax.Array,
        assistant_mask: jax.Array,
        attention_mask: jax.Array,
    ) -> Any:
        """Accumulate gradients and per-episode losses over one batch shard.

        Runs under jax.shard_map: every array argument here is this device's
        LOCAL shard only (params stay replicated). The single jax.lax.psum
        at the end is the only cross-device collective in the whole step,
        instead of the one-all-reduce-per-episode (64/step) that automatic
        SPMD inserted for the equivalent jax.lax.scan. Safe because
        chunked_lm_loss's reduction is a plain jnp.sum over pre-normalized
        weights, so local partial sums plus one final psum equal the global
        sum. See docs/gpu-hang-debugging-log.md Part 14.
        """
        episodes = tuple(
            jnp.swapaxes(value, 0, 1)
            for value in (input_ids, assistant_mask, attention_mask)
        )
        param_grads = tree_zeros(params)

        def body(carry: PyTree, values: Any) -> Any:
            local_ids, local_assistant, local_attention = values
            current = {
                "input_ids": local_ids,
                "assistant_mask": local_assistant,
                "attention_mask": local_attention,
                "loss_normalizer": normalizer,
            }
            (_, losses), grads = jax.value_and_grad(self._episode_loss, has_aux=True)(
                params, current
            )
            return tree_add(carry, grads), losses

        param_grads, losses = jax.lax.scan(body, param_grads, episodes)
        return jax.lax.psum((param_grads, losses), axis_name=("data", "fsdp"))

    def _train_step(
        self, params: PyTree, opt_state: PyTree, step: jax.Array, batch: Any
    ) -> Any:
        assistant = batch["assistant_mask"][..., 1:].astype(jnp.float32)
        auxiliary = batch["attention_mask"][..., 1:] - assistant
        normalizer = (assistant + float(self.config.aux_loss_weight) * auxiliary).sum()
        normalizer = jnp.where(normalizer > 0, normalizer, 1)

        batch_spec = get_batch_sharding(self.mesh).spec
        param_grads, losses = jax.shard_map(
            self._local_grad_accum,
            mesh=self.mesh,
            in_specs=(P(), P(), batch_spec, batch_spec, batch_spec),
            out_specs=(P(), P()),
            check_vma=False,
        )(
            params,
            normalizer,
            batch["input_ids"],
            batch["assistant_mask"],
            batch["attention_mask"],
        )
        metrics = episodic_loss_metrics(
            losses,
            batch["assistant_mask"],
            batch["attention_mask"],
            float(self.config.aux_loss_weight),
        )
        return self.apply_gradients(params, opt_state, step, param_grads, metrics)
