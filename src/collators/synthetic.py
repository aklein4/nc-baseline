"""Synthetic token batches for smoke tests and local experiments."""

from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray


class SyntheticCollator:
    """Generate fixed-shape token batches from a NumPy random generator."""

    def __init__(
        self,
        sequence_length: int,
        vocab_size: int,
        episodes: int = 1,
        assistant_masks: bool | None = None,
    ) -> None:
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.episodes = episodes
        self.assistant_masks = (
            episodes > 1 if assistant_masks is None else assistant_masks
        )

    def __call__(
        self, rng: np.random.Generator, batch_size: int
    ) -> dict[str, NDArray[Any]]:
        """Generate one batch, including episodic masks when requested."""
        shape = (batch_size, self.sequence_length)
        if self.episodes > 1:
            shape = (batch_size, self.episodes, self.sequence_length)
        ids = rng.integers(0, self.vocab_size, shape, dtype=np.int32)
        batch = {"input_ids": ids}
        if self.assistant_masks:
            assistant = np.zeros(shape, bool)
            assistant[..., self.sequence_length // 2 :] = True
            batch.update(assistant_mask=assistant, attention_mask=np.ones(shape, bool))
        return batch

    def batches(
        self, batch_size: int, seed: int, start_batch: int = 0
    ) -> Iterator[dict[str, NDArray[Any]]]:
        """Yield deterministic batches, advancing the RNG when resuming."""
        rng = np.random.default_rng(seed)
        for _ in range(start_batch):
            self(rng, batch_size)
        while True:
            yield self(rng, batch_size)
