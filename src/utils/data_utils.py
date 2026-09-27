"""Configured batch iteration with step-based resume.

See docs/architecture.md#data-and-resume and the Datasets loading API:
https://huggingface.co/docs/datasets/package_reference/loading_methods
Streaming resume uses IterableDataset's own state:
https://huggingface.co/docs/datasets/stream#save-a-dataset-checkpoint-and-resume-iteration
"""

import logging
import queue
import threading
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import datasets
import jax
from hydra.utils import instantiate
from numpy.typing import NDArray
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# Crossing a streaming shard boundary costs real time: a reproducible ~+16s
# on an otherwise ~57s step, seen in two independent runs. The watchdog has
# to clear that by a wide margin or it fires on a healthy run.
BATCH_TIMEOUT_SECONDS = 600.0
BATCH_ATTEMPTS = 5
# Two batches in flight is enough to hide collation behind the device step
# without holding much host memory (~50MB per batch at the 1B config).
PREFETCH_DEPTH = 2

_DONE = object()


def get_dataset(
    config: Mapping[str, Any],
) -> datasets.Dataset | datasets.IterableDataset:
    """Load a dataset and give each process a distinct shard."""
    dataset = datasets.load_dataset(**config)
    if jax.process_count() > 1:
        dataset = dataset.shard(jax.process_count(), jax.process_index())
    return dataset


def _collate_batches(
    dataset: datasets.IterableDataset, collator: Any, batch_size: int
) -> Iterator[tuple[dict[str, NDArray[Any]], dict[str, Any] | None]]:
    """Yield each collated batch with the stream position that produced it.

    The position is captured after the batch's rows are consumed, so a
    restart from it replays nothing and drops nothing.
    """
    rows = []
    for row in dataset:
        rows.append(row)
        if len(rows) == batch_size:
            batch = collator(rows)
            rows = []
            try:
                state = dataset.state_dict()
            except (AttributeError, NotImplementedError):
                state = None
            yield batch, state


def _prefetch(
    source: Callable[[], Iterator[Any]], depth: int, timeout: float
) -> Iterator[Any]:
    """Drain ``source`` on a worker thread, raising if it stalls.

    Solves two problems at once. Collating a batch means tokenizing
    ``global_batch_size * cluster_length`` conversations, which used to run
    between steps while the GPUs sat idle; a worker thread overlaps it with
    the device step, since fast tokenizers release the GIL. And a streaming
    read that stalls -- Hub throttling, a connection dropped mid-shard --
    used to wedge training forever with no log line and no crash, because
    the read has no timeout of its own and the loop had no way to notice.
    Here the consumer gives up after ``timeout`` and the caller can resume.

    A stalled worker cannot be killed, only abandoned; it is a daemon so it
    never holds up process exit.
    """
    items: queue.Queue = queue.Queue(maxsize=depth)
    stop = threading.Event()

    def put(item: Any) -> bool:
        while not stop.is_set():
            try:
                items.put(item, timeout=1.0)
                return True
            except queue.Full:
                continue
        return False

    def worker() -> None:
        try:
            for item in source():
                if not put(item):
                    return
        except Exception as error:  # noqa: BLE001
            # Deliberately broad: this is a thread boundary, and anything that
            # escapes here would kill the worker silently, leaving the consumer
            # to report a misleading timeout instead of the real failure. The
            # exception is re-raised on the consumer thread, not swallowed.
            # (KeyboardInterrupt/SystemExit go to the main thread, never here.)
            put(error)
        else:
            put(_DONE)

    thread = threading.Thread(target=worker, name="batch-prefetch", daemon=True)
    thread.start()
    try:
        while True:
            try:
                item = items.get(timeout=timeout)
            except queue.Empty:
                raise TimeoutError(
                    f"no batch produced in {timeout:.0f}s; the streaming read "
                    "is stalled"
                ) from None
            if item is _DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()


def _open_stream(
    config: Mapping[str, Any], skip_rows: int, state: dict[str, Any] | None
) -> datasets.IterableDataset:
    """Reopen the stream at a saved position, or at a row offset."""
    dataset = get_dataset(config)
    if skip_rows:
        dataset = dataset.skip(skip_rows)
    if state is not None:
        # Restore only onto an identically shaped pipeline -- the state
        # describes this arrangement of wrappers, not just a row count.
        dataset.load_state_dict(state)
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
        # Generated locally, so it can neither stall nor need resuming.
        yield from collator.batches(batch_size, seed + jax.process_index(), start_batch)
        return

    skip_rows = start_batch * batch_size
    state: dict[str, Any] | None = None
    delivered = 0

    for attempt in range(BATCH_ATTEMPTS):
        if state is None and delivered:
            # No resumable state, so replay from the top and discard. Correct
            # but slow; only reachable on a datasets build without state_dict.
            dataset = _open_stream(
                config.dataset, skip_rows + delivered * batch_size, None
            )
        else:
            dataset = _open_stream(config.dataset, skip_rows, state)

        try:
            for batch, position in _prefetch(
                lambda stream=dataset: _collate_batches(stream, collator, batch_size),
                PREFETCH_DEPTH,
                BATCH_TIMEOUT_SECONDS,
            ):
                yield batch
                # Only advance after the consumer has taken the batch, so a
                # restart resumes from what training actually saw rather than
                # from how far the prefetch thread had run ahead.
                delivered += 1
                state = position
            return
        except (TimeoutError, OSError) as error:
            remaining = BATCH_ATTEMPTS - attempt - 1
            logger.exception(
                "Batch stream failed after %d batches (%s); %d attempt(s) left",
                delivered,
                type(error).__name__,
                remaining,
            )
            if not remaining:
                raise

    raise RuntimeError("unreachable: batch stream retry loop exited")
