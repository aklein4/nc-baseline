"""Shared training loop, compiled update, and metrics.

References:
- docs/components.md#trainers-and-optimization
- https://optax.readthedocs.io/
"""

import logging
import shutil
from collections import deque
from collections.abc import Iterable, Mapping
from pathlib import Path
from time import monotonic
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import optax
from jax.experimental import multihost_utils
from omegaconf import DictConfig, OmegaConf

from models.base import CustomModel
from utils.checkpoints import (
    ParameterCheckpoint,
    format_step,
    save_checkpoint,
    upload_checkpoint,
)
from utils.optimizer_utils import gradient_norm, make_optimizer
from utils.typing_utils import PyTree

logger = logging.getLogger(__name__)


class BaseTrainer:
    """Universal training loop and ordinary differentiated Optax update."""

    required_config_keys: ClassVar[list[str]] = [
        "max_steps",
        "max_grad_norm",
        "run_dir",
        "debug",
        "checkpoint.interval",
        "checkpoint.save_optimizer",
    ]

    def __init__(
        self,
        model: CustomModel,
        config: DictConfig,
        params: PyTree,
        labels: PyTree | None = None,
    ) -> None:
        """Initialize trainable values and compile the concrete step function."""
        self.model = model
        self.config = config
        self.labels = labels
        self.validate_config()
        self.tx, self.schedules = make_optimizer(
            config, labels, model=model, params=params
        )
        self.params = params
        self.opt_state = self.tx.init(params)
        self.step = 0
        self.train_step = jax.jit(self._train_step, donate_argnums=(0, 1))

    def load_state(self, checkpoint: ParameterCheckpoint) -> None:
        """Restore the parameters, optimizer state, and step owned by this trainer."""
        if checkpoint.step is None:
            raise ValueError("training state restore requires a checkpoint step")
        templates = {"params": self.params, "optimizer": self.opt_state}
        restored = checkpoint.restore_state(templates)
        if "optimizer" not in restored:
            raise KeyError("checkpoint contains no optimizer state")
        self.params = restored["params"]
        self.opt_state = restored["optimizer"]
        self.step = int(checkpoint.step)

    def validate_config(self) -> None:
        """Fail early when the concrete trainer's declared config is incomplete."""
        missing = []
        sentinel = object()
        for key in self.required_config_keys:
            if OmegaConf.select(self.config, key, default=sentinel) is sentinel:
                missing.append(key)
        if missing:
            joined = ", ".join(missing)
            raise ValueError(
                f"{type(self).__name__} requires trainer config keys: {joined}"
            )

    def learning_rates(self, step: int | jax.Array) -> dict[str, Any]:
        """Evaluate configured learning-rate schedules for step metrics."""
        return {
            f"{name}_lr" if len(self.schedules) > 1 else "lr": (
                value(step) if callable(value) else value
            )
            for name, value in self.schedules.items()
        }

    def loss(self, params: PyTree, batch: dict[str, jax.Array]) -> Any:
        """Return ``(loss, metrics)`` for an ordinary differentiated trainer."""
        raise NotImplementedError

    def _train_step(
        self, params: PyTree, opt_state: PyTree, step: jax.Array, batch: Any
    ) -> tuple[PyTree, PyTree, dict[str, jax.Array]]:
        """Differentiate ``loss`` and apply one optimizer update."""
        (loss, metrics), grads = jax.value_and_grad(self.loss, has_aux=True)(
            params, batch
        )
        metrics = dict(metrics)
        metrics["loss"] = loss
        return self.apply_gradients(params, opt_state, step, grads, metrics)

    def apply_gradients(
        self,
        params: PyTree,
        opt_state: PyTree,
        step: jax.Array,
        grads: PyTree,
        metrics: dict[str, jax.Array],
    ) -> tuple[PyTree, PyTree, dict[str, jax.Array]]:
        """Apply gradients and attach optimizer metrics to the step result."""
        metrics["grad_norm"] = gradient_norm(grads, self.labels)
        updates, opt_state = self.tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        metrics.update(self.learning_rates(step))
        return params, opt_state, metrics

    def on_step(self, metrics: Mapping[str, float], wandb_run: Any = None) -> None:
        """Report one completed step to the console and optional W&B run."""
        if jax.process_index() != 0:
            return
        logger.info(
            "step: %d, loss: %.4f, grad_norm: %.3f, step_time_s: %.3f, steps/hr: %.1f",
            self.step,
            metrics["loss"],
            metrics["grad_norm"],
            metrics["step_time_s"],
            metrics["steps_per_hour"],
        )
        if wandb_run is not None:
            wandb_run.log(dict(metrics), step=self.step)

    def save(
        self,
        directory: str | Path,
        metadata: Mapping[str, object],
        *,
        repo_id: str | None = None,
        force: bool = False,
    ) -> None:
        """Save the current training state and optionally upload it."""
        payload = {"params": self.params}
        if self.config.checkpoint.save_optimizer:
            payload["optimizer"] = self.opt_state
        path = save_checkpoint(payload, directory, self.step, metadata, force)
        if jax.process_index() == 0:
            logger.info("Saved checkpoint to %s", path)
            if repo_id is not None:
                upload_checkpoint(path, repo_id, self.step)
                logger.info(
                    "Uploaded checkpoint to %s/%s", repo_id, format_step(self.step)
                )
                shutil.rmtree(path)
                logger.info("Removed temporary local checkpoint %s", path)
        multihost_utils.sync_global_devices(f"checkpoint_{self.step}")

    def train(
        self,
        batches: Iterable[Any],
        *,
        metadata: Mapping[str, object] | None = None,
        repo_id: str | None = None,
        wandb_run: Any = None,
    ) -> None:
        """Run training, including step reporting and periodic checkpoints."""
        checkpoint_interval = int(self.config.checkpoint.interval)
        if checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        checkpoint_dir = None
        if not self.config.debug:
            checkpoint_dir = (
                Path(self.config.run_dir).expanduser().resolve() / "checkpoints"
            )

        metadata = {} if metadata is None else metadata
        iterator = iter(batches)
        start = monotonic()
        last_step_time = start
        step_durations: deque[float] = deque(maxlen=10)
        atoms_seen = 0
        step_fn = None

        jax.block_until_ready((self.params, self.opt_state))
        logger.info("Starting training loop at step %d", self.step)

        while self.step < int(self.config.max_steps):
            try:
                batch = next(iterator)
            except StopIteration as error:
                raise RuntimeError(
                    "dataset exhausted before trainer.max_steps was reached"
                ) from error

            if step_fn is None:
                logger.info("Compiling first training step...")
                step_fn = self.train_step.lower(self.params, self.opt_state, jnp.asarray(self.step), batch).compile()
                logger.info("Compilation complete; executing first training step...")
                
            self.params, self.opt_state, metrics = step_fn(
                self.params, self.opt_state, jnp.asarray(self.step), batch
            )
            self.step += 1
            metrics = jax.device_get(metrics)
            atoms_seen += int(metrics.get("atom_count", 0))
            current_time = monotonic()
            elapsed = current_time - start
            step_time_s = current_time - last_step_time
            step_durations.append(step_time_s)
            last_step_time = current_time
            host_metrics = {key: float(value) for key, value in metrics.items()}
            host_metrics.update(
                step=self.step,
                atoms_seen=atoms_seen,
                step_time_s=step_time_s,
                steps_per_hour=len(step_durations)
                / max(sum(step_durations), 1e-6)
                * 3600,
                hours_elapsed=elapsed / 3600,
            )
            self.on_step(host_metrics, wandb_run)
            if checkpoint_dir is not None and self.step % checkpoint_interval == 0:
                self.save(
                    checkpoint_dir,
                    metadata,
                    repo_id=repo_id,
                )

        if checkpoint_dir is not None and self.step % checkpoint_interval:
            self.save(
                checkpoint_dir,
                metadata,
                repo_id=repo_id,
                force=True,
            )
        if wandb_run is not None:
            wandb_run.finish()
