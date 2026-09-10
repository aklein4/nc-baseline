"""Configured batch iteration with step-based resume.

See docs/architecture.md#data-and-resume and the Datasets loading API:
https://huggingface.co/docs/datasets/package_reference/loading_methods
"""

from collections.abc import Iterator, Mapping
from typing import Any

import datasets
import jax
from hydra.utils import instantiate
from numpy.typing import NDArray
from omegaconf import DictConfig


def get_dataset(
    config: Mapping[str, Any],
) -> datasets.Dataset | datasets.IterableDataset:
    """Load a dataset and give each process a distinct shard."""
    dataset = datasets.load_dataset(**config)
    if jax.process_count() > 1:
        dataset = dataset.shard(jax.process_count(), jax.process_index())
    return dataset


def batches(
    config: DictConfig, batch_size: int, seed: int, start_batch: int = 0
) -> Iterator[dict[str, NDArray[Any]]]:
    """Yield configured batches, resuming from an absolute batch offset."""
    skip_batches = int(config.get("skip_batches", 0))
    if skip_batches < 0:
        raise ValueError("data.skip_batches must be non-negative")
    start_batch = max(start_batch, skip_batches)

    collator = instantiate(config.collator)
    if "dataset" not in config:
        yield from collator.batches(batch_size, seed + jax.process_index(), start_batch)
        return

    dataset = get_dataset(config.dataset)
    if start_batch:
        dataset = dataset.skip(start_batch * batch_size)
    rows = []
    for row in dataset:
        rows.append(row)
        if len(rows) == batch_size:
            yield collator(rows)
            rows = []
