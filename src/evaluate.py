"""Episodic ICL/persona evaluation; see docs/getting-started.md#evaluation.

Ports outer-loop/src/evaluate_{icl,persona}.py: independent fast state per row,
assistant-only teacher-forced scores, and an unweighted mean over benchmarks.
Hydra selects the benchmark and model; parameter loading follows src/train.py.
"""

import json
import logging
from pathlib import Path
from typing import Any

import datasets
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from models import make_model
from models.forte import ForteModel
from models.lora import LoRAModel
from utils import constants
from utils.checkpoints import format_step, open_checkpoint
from utils.git_utils import get_current_commit_hash
from utils.losses import chunked_lm_loss, frozen_head_lm_loss

logger = logging.getLogger(__name__)


def checkpoint_metadata(checkpoint) -> dict:
    info = {
        "resolved_path": str(checkpoint.path),
        "format": "safetensors" if checkpoint.readers else "orbax",
    }
    conversion = checkpoint.path / "conversion.json"
    if conversion.is_file():
        info["conversion"] = json.loads(conversion.read_text())
    return info


def validate_config(config: DictConfig) -> None:
    for key in ("batch_size", "num_eval", "num_logit_iterations"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if config.collator.max_length < 2:
        raise ValueError("collator.max_length must be at least 2")
    if not config.num_examples or any(n < 0 for n in config.num_examples):
        raise ValueError("num_examples must contain nonnegative adaptation counts")
    if config.max_rows is not None and config.max_rows <= 0:
        raise ValueError("max_rows must be positive or null")
    if config.eval_fn not in ("output_loss", "exact_match"):
        raise ValueError(f"Unknown eval_fn: {config.eval_fn}")
    if config.initialization_step is not None and not config.initialization:
        raise ValueError("initialization_step requires initialization")


def load_rows(config: DictConfig) -> list[dict[str, Any]]:
    """Keep source order; max_rows is a limit per subset, as in the originals."""
    dataset_args = OmegaConf.to_container(config.benchmark.dataset, resolve=True)
    subsets = config.subsets or datasets.get_dataset_config_names(
        dataset_args["path"],
        **{k: dataset_args[k] for k in ("revision", "token") if k in dataset_args},
    )
    rows = []
    maximum = max(config.num_examples)
    for subset in tqdm(subsets, desc="loading data"):
        count = 0
        for row in datasets.load_dataset(**dataset_args, name=subset):
            if row.get("num_examples", maximum) < maximum:
                continue
            train = row[config.benchmark.train_column]
            test = row[config.benchmark.test_column]
            if len(train) < maximum or len(test) < config.num_eval:
                continue
            rows.append({"subset": subset, "train_data": train, "test_data": test})
            count += 1
            if config.max_rows is not None and count >= config.max_rows:
                break
        if not count:
            raise ValueError(f"No eligible rows in subset {subset!r}")
    if not rows:
        raise ValueError("No eligible evaluation rows")
    return rows


def adaptation_lr_scale(config: DictConfig, index: int, steps: int) -> float:
    if not config.lr_scale_decay:
        return config.lr_scale
    fraction = index / (steps - 1) if steps > 1 else 0.0
    return config.lr_scale_start + fraction * (
        config.lr_scale_end - config.lr_scale_start
    )


def make_fns(model, config: DictConfig):
    """Build pure adaptation/scoring functions; only ephemeral state changes.

    Model-specific forward/update calls live here. New benchmarks only need a
    config if they expose lists of training and test chat conversations.
    """
    if isinstance(model, ForteModel):
        first = model.clone(mode="first")
        inference = model.clone(mode="inference")

        def hidden(params, state, buffer, batch, adapting):
            current = first if adapting else inference
            return current.apply(
                {"params": params},
                batch["input_ids"],
                valid_mask=batch["attention_mask"],
                forte_state=state,
                grad_buffer=buffer,
                shift_states=True,
                compute_logits=False,
            )

    elif isinstance(model, LoRAModel):

        def hidden(params, state, buffer, batch, adapting):
            return model.apply(
                {"params": params},
                batch["input_ids"],
                lora_state=state,
                grad_buffer=buffer,
                shift_states=True,
                compute_logits=False,
            )

    else:
        raise TypeError("Evaluation requires a ForteModel or LoRAModel")

    def objective(params, value, buffer, batch):
        states = hidden(params, value, buffer, batch, True)
        assistant = batch["assistant_mask"].astype(jnp.float32)
        valid = batch["attention_mask"].astype(jnp.float32)
        return frozen_head_lm_loss(
            model,
            params,
            states,
            batch["input_ids"],
            jnp.stack((assistant, valid - assistant), axis=-1),
            jnp.asarray([1.0, config.aux_weight], jnp.float32),
            config.num_logit_iterations,
        )[0]

    def adapt(params, state, batch, lr_scale):
        if isinstance(model, ForteModel):
            value_grad, buffer_grad = jax.grad(objective, argnums=(1, 2))(
                params, state.value, state.grad_buffer, batch
            )
            return model.update_state(
                state, value_grad * lr_scale, buffer_grad, mode="first"
            )
        buffer = jax.tree.map(jnp.zeros_like, state.value)
        gradient = jax.grad(objective, argnums=2)(params, state.value, buffer, batch)
        updated = model.update_state(state, gradient)
        return updated.replace(
            value=jax.tree.map(
                lambda old, new: old + lr_scale * (new - old),
                state.value,
                updated.value,
            )
        )

    def score(params, state, batch):
        states = hidden(params, state.value, None, batch, False)
        mask = batch["assistant_mask"][:, 1:].astype(jnp.float32)
        counts = mask.sum(axis=1)
        # One reduction per row keeps both cross entropy and exact match
        # memory-bounded without constructing full sequence vocabulary logits.
        weights = mask[..., None] * jnp.eye(mask.shape[0])[:, None, :]
        losses, correct = chunked_lm_loss(
            model,
            params,
            states,
            batch["input_ids"][:, 1:],
            weights,
            config.num_logit_iterations,
        )
        if config.eval_fn == "exact_match":
            return ((correct == counts) & (counts > 0)).astype(jnp.float32)
        return losses / jnp.maximum(counts, 1)

    if config.compile:
        adapt = jax.jit(adapt)
        score = jax.jit(score)
    return adapt, score


def evaluate_rows(model, params, adapt, score, collator, rows, config):
    counts = sorted(set(config.num_examples))
    totals = {n: {} for n in counts}
    row_counts = {}

    def encode(batch, column, index):
        return collator.collate_messages(
            [row[column][index] for row in batch],
            (len(batch), collator.max_length),
        )

    for start in tqdm(range(0, len(rows), config.batch_size), desc="batches"):
        batch = rows[start : start + config.batch_size]
        state = model.init_state(len(batch))
        for row in batch:
            subset = row["subset"]
            row_counts[subset] = row_counts.get(subset, 0) + 1
        for n in tqdm(range(counts[-1] + 1), desc="adapting", leave=False):
            if n:
                scale = jnp.asarray(
                    adaptation_lr_scale(config, n - 1, counts[-1]), jnp.float32
                )
                state = adapt(params, state, encode(batch, "train_data", n - 1), scale)
            if n not in totals:
                continue
            scores = np.zeros(len(batch), np.float64)
            for index in range(config.num_eval):
                scores += np.asarray(
                    score(params, state, encode(batch, "test_data", index)),
                    dtype=np.float64,
                )
            scores /= config.num_eval
            if not np.isfinite(scores).all():
                raise ValueError(f"Nonfinite scores after {n} adaptation examples")
            for row, value in zip(batch, scores, strict=True):
                subset = row["subset"]
                totals[n][subset] = totals[n].get(subset, 0.0) + float(value)

    results = []
    for n in counts:
        benchmarks = {k: v / row_counts[k] for k, v in totals[n].items()}
        results.append(
            {
                "num_examples": n,
                "benchmarks": benchmarks,
                "average": sum(benchmarks.values()) / len(benchmarks),
            }
        )
    return results


def save_results(config: DictConfig, results, checkpoint_info: dict) -> Path:
    """Keep legacy score rows; store run metadata once, on the first row."""
    if not results:
        raise ValueError("Cannot save an evaluation without scores")
    root = Path(config.output_dir).expanduser() / config.benchmark.results_dir
    if config.initialization_step is not None:
        root /= config.save_name or ""
        root /= str(config.initialization).replace("/", "--")
        label = format_step(config.initialization_step)
    else:
        root /= config.save_name or "fresh"
        root /= config.model_name
        label = f"base_lr_{config.model.base_lr:.0e}".replace("+", "")
    path = root / f"{config.result_label or label}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config": OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        "checkpoint": {
            "source": config.initialization,
            "step": config.initialization_step,
            **checkpoint_info,
        },
        "overrides": (
            list(HydraConfig.get().overrides.task) if HydraConfig.initialized() else []
        ),
        "git_commit": get_current_commit_hash(),
    }
    output = [{**results[0], "metadata": metadata}, *results[1:]]
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    logger.info("Wrote %s", path)
    return path


@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def main(config: DictConfig) -> None:
    validate_config(config)
    model = make_model(config.model)
    model.configure_environment()
    constants.JAX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(constants.JAX_CACHE_DIR))
    if jax.process_count() != 1:
        raise ValueError("Evaluation currently runs on one process/device")
    adapt, score = make_fns(model, config)
    rows = load_rows(config)
    logger.info("Loaded %d rows", len(rows))
    collator = instantiate(config.collator)
    logger.info("Initializing %s", config.model_name)
    params = model.initialize_params(config.seed)
    checkpoint_info = {}
    if config.initialization:
        logger.info(
            "Loading %s at step %s", config.initialization, config.initialization_step
        )
        with open_checkpoint(
            config.initialization,
            cache_dir=constants.CHECKPOINTS_PATH,
            step=config.initialization_step,
        ) as checkpoint:
            checkpoint_info = checkpoint_metadata(checkpoint)
            params = model.load_params(params, checkpoint)
    params = jax.device_put(params)
    results = evaluate_rows(model, params, adapt, score, collator, rows, config)
    save_results(config, results, checkpoint_info)


if __name__ == "__main__":
    for dependency in ("datasets", "httpx", "huggingface_hub"):
        logging.getLogger(dependency).setLevel(logging.WARNING)
    main()
