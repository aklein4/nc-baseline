"""Provider safetensors and portable local/Hugging Face checkpoints.

See docs/components.md#checkpoint-utilities and
docs/architecture.md#initialization-and-checkpoints.
Upstream APIs:
- https://orbax.readthedocs.io/en/latest/guides/checkpoint/
- https://huggingface.co/docs/huggingface_hub/guides/repository
- https://huggingface.co/docs/safetensors/
"""

import json
import mmap
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any, Self

import jax
import ml_dtypes
import numpy as np
import orbax.checkpoint as ocp
from huggingface_hub import HfApi, snapshot_download

from utils.typing_utils import PyTree

DTYPES = {
    "BF16": (np.uint16, ml_dtypes.bfloat16),
    "F16": (np.float16, None),
    "F32": (np.float32, None),
    "F64": (np.float64, None),
    "I32": (np.int32, None),
    "I64": (np.int64, None),
}


def format_step(step: int) -> str:
    """Format a training step as its canonical 12-digit checkpoint name."""
    return f"{step:012d}"


class SafeTensorFile:
    """Tiny NumPy safetensors reader, including BF16 without a PyTorch dependency."""

    def __init__(self, path: str | PathLike[str]) -> None:
        self.file = open(  # noqa: SIM115 - remains open while the mmap is alive
            path, "rb"
        )
        header_size = int.from_bytes(self.file.read(8), "little")
        self.header = json.loads(self.file.read(header_size))
        self.offset = 8 + header_size
        self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

    def get(self, name: str) -> np.ndarray:
        """Return one tensor as a zero-copy NumPy view when possible."""
        info = self.header[name]
        storage_dtype, view_dtype = DTYPES[info["dtype"]]
        start, end = info["data_offsets"]
        array = np.frombuffer(
            self.mapping,
            dtype=storage_dtype,
            count=(end - start) // np.dtype(storage_dtype).itemsize,
            offset=self.offset + start,
        ).reshape(info["shape"])
        return array.view(view_dtype) if view_dtype is not None else array

    def close(self) -> None:
        """Close the memory map and its backing file."""
        self.mapping.close()
        self.file.close()


def _checkpoint_path(
    source: str | PathLike[str],
    cache_dir: str | PathLike[str] | None = None,
    step: int | None = None,
) -> Path:
    """Resolve a local checkpoint or download one from Hugging Face."""
    path = Path(source).expanduser()
    if path.exists():
        return path.resolve()
    kwargs = {}
    if cache_dir is not None:
        local_dir = Path(cache_dir).expanduser().resolve() / str(source).replace(
            "/", "--"
        )
        local_dir.mkdir(parents=True, exist_ok=True)
        kwargs["local_dir"] = local_dir
    patterns = (
        (f"{format_step(step)}/state/**",)
        if step is not None
        else ("config.json", "*.safetensors", "*.safetensors.index.json")
    )
    return Path(
        snapshot_download(repo_id=str(source), allow_patterns=patterns, **kwargs)
    )


def _weight_files(
    path: Path,
) -> tuple[dict[str, str], dict[str, SafeTensorFile]]:
    index = path / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        return weight_map, {
            name: SafeTensorFile(path / name) for name in set(weight_map.values())
        }
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors found in {path}")
    readers = {file.name: SafeTensorFile(file) for file in files}
    weight_map = {
        key: file.name
        for file in files
        for key in readers[file.name].header
        if key != "__metadata__"
    }
    return weight_map, readers


class ParameterCheckpoint:
    """One resolved parameter source, either safetensors or an Orbax tree."""

    def __init__(self, path: Path, step: int | None = None) -> None:
        self.step = step
        if step is not None:
            checkpoint_path = path / format_step(step)
            if not checkpoint_path.is_dir():
                raise FileNotFoundError(f"checkpoint step {step} not found in {path}")
            path = checkpoint_path
        self.path = path
        self.weight_map: dict[str, str] = {}
        self.readers: dict[str, SafeTensorFile] = {}
        if (self.path / "model.safetensors.index.json").exists() or any(
            self.path.glob("*.safetensors")
        ):
            self.weight_map, self.readers = _weight_files(self.path)

    def __contains__(self, name: str) -> bool:
        return name in self.weight_map

    def read(self, name: str, transpose: bool = False) -> np.ndarray:
        if name not in self.weight_map:
            raise KeyError(f"checkpoint is missing {name}")
        array = self.readers[self.weight_map[name]].get(name)
        if transpose:
            array = array.T
        # Detach returned FP32 arrays from the mmap as well as converted dtypes;
        # model trees must remain usable after the checkpoint context closes.
        return np.array(array, dtype=np.float32, copy=True)

    def restore(self, params: PyTree) -> PyTree:
        if (self.path / "state").is_dir():
            return _restore_tree({"params": params}, self.path / "state", partial=True)[
                "params"
            ]
        return restore_params(params, self.path)

    def restore_state(self, templates: Mapping[str, PyTree]) -> dict[str, PyTree]:
        """Restore a training payload using the supplied templates."""
        templates = dict(templates)
        saved = _restore_tree(templates, self.path / "state")
        return _place_like(saved, templates)

    def close(self) -> None:
        for reader in self.readers.values():
            reader.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def open_checkpoint(
    source: str | PathLike[str],
    cache_dir: str | PathLike[str] | None = None,
    step: int | None = None,
) -> ParameterCheckpoint:
    """Open one local or Hugging Face parameter/training checkpoint."""
    return ParameterCheckpoint(_checkpoint_path(source, cache_dir, step), step)


def _save_tree(
    tree: PyTree, path: Path, force: bool = False, process_local: bool = False
) -> None:
    kwargs = {}
    if process_local:
        process = jax.process_index()
        kwargs["multiprocessing_options"] = ocp.options.MultiprocessingOptions(
            primary_host=process,
            active_processes={process},
        )
    checkpointer = ocp.StandardCheckpointer(**kwargs)
    checkpointer.save(path, tree, force=force)
    checkpointer.wait_until_finished()
    checkpointer.close()


def save_params(
    params: PyTree,
    path: str | PathLike[str],
    force: bool = False,
    process_local: bool = False,
) -> None:
    """Save a standalone parameter tree for activation initialization."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Standalone initialization artifacts contain values, not the topology of
    # the accelerator that produced them.
    _save_tree(jax.tree.map(np.asarray, params), path, force, process_local)


def restore_params(params: PyTree, path: str | PathLike[str]) -> PyTree:
    """Restore a portable standalone parameter tree into host arrays."""
    path = Path(path).expanduser().resolve()
    # Device placement belongs to the caller's current mesh, not the device
    # topology recorded by the machine that produced this initialization tree.
    return _restore_tree(params, path)


def _host_tree(tree: PyTree) -> PyTree:
    """Gather globally sharded arrays and return a portable host pytree."""
    from jax.experimental import multihost_utils

    def host(value: Any) -> np.ndarray:
        if isinstance(value, jax.Array) and not value.is_fully_addressable:
            value = multihost_utils.process_allgather(value, tiled=True)
        return np.asarray(value)

    return jax.tree.map(host, tree)


def _restore_tree(template: PyTree, path: Path, partial: bool = False) -> PyTree:
    """Restore a portable host pytree with the structure of ``template``."""
    device = jax.sharding.SingleDeviceSharding(jax.local_devices()[0])

    def target(value: Any) -> jax.ShapeDtypeStruct:
        value = np.asarray(value)
        return jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=device)

    target = jax.tree.map(target, template)
    checkpointer = ocp.PyTreeCheckpointer()
    output = checkpointer.restore(
        path,
        args=ocp.args.PyTreeRestore(item=target, partial_restore=partial),
    )
    checkpointer.close()
    return jax.tree.map(np.asarray, output)


def _place_like(tree: PyTree, template: PyTree) -> PyTree:
    """Place restored host values with the template arrays' shardings."""
    return jax.tree.map(
        lambda value, expected: (
            jax.device_put(value, expected.sharding)
            if isinstance(expected, jax.Array)
            else value
        ),
        tree,
        template,
    )


def save_checkpoint(
    payload: Mapping[str, PyTree],
    directory: str | PathLike[str],
    step: int,
    metadata: Mapping[str, object],
    force: bool = False,
) -> Path:
    """Gather and save one portable training checkpoint on process zero."""
    path = Path(directory).expanduser().resolve() / format_step(step)
    payload = dict(payload)
    payload = _host_tree(payload)
    if jax.process_index() != 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_tree(payload, path / "state", force, process_local=True)
    metadata = dict(metadata)
    metadata.update(step=step, checkpoint_items=sorted(payload))
    (path / "metadata.json").write_text(json.dumps(metadata, indent=4) + "\n")
    return path


def create_hf_repo(repo_id: str, private: bool = False) -> None:
    """Create the model repository used by source training checkpoints."""
    # Official Hub client: https://huggingface.co/docs/huggingface_hub/guides/repository
    HfApi().create_repo(
        repo_id=repo_id, repo_type="model", private=private, exist_ok=True
    )


def upload_checkpoint(path: str | PathLike[str], repo_id: str, step: int) -> None:
    """Upload a locally saved checkpoint directory to Hugging Face."""
    HfApi().upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=Path(path),
        path_in_repo=format_step(step),
        commit_message=f"Upload checkpoint at step {step}",
    )
