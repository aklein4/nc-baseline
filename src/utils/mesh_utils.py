"""JAX multi-process initialization, meshes, and batch sharding.

See docs/architecture.md#mesh-and-sharding and JAX multi-process execution:
https://docs.jax.dev/en/latest/multi_process.html
"""

from typing import Any

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def initialize_distributed(config: Any) -> dict[str, Any]:
    """Initialize JAX across TPU VMs or CUDA/ROCm hosts.

    JAX detects Cloud TPU, Slurm, and Open MPI environments. Explicit fields
    remain available for launchers that only provide rank/coordinator values.
    https://docs.jax.dev/en/latest/multi_process.html
    """
    initialize = config["initialize"]
    if jax.distributed.is_initialized() or initialize is False:
        return {}

    explicit = config.get("coordinator_address") is not None
    kwargs = {
        key: config.get(key)
        for key in (
            "coordinator_address",
            "num_processes",
            "process_id",
            "local_device_ids",
            "cluster_detection_method",
        )
        if config.get(key) is not None
    }
    try:
        # Delegate cluster detection to JAX so new TPU and launcher
        # environments do not require updates to a local variable list.
        jax.distributed.initialize(**kwargs)
    except ValueError as error:
        if (
            initialize == "auto"
            and not explicit
            and str(error) == "coordinator_address should be defined."
        ):
            return {}
        raise
    return kwargs


def create_mesh(fsdp_axis_size: int) -> Mesh:
    """Create a global data x host-local-FSDP device mesh."""
    local_count = jax.local_device_count()

    fsdp_axis_size = local_count if int(fsdp_axis_size) == -1 else int(fsdp_axis_size)
    if fsdp_axis_size < 1 or local_count % fsdp_axis_size:
        raise ValueError(
            f"fsdp_axis_size={fsdp_axis_size} must divide local_device_count={local_count}"
        )

    # reshaping keeps FSDP host-local
    devices = np.asarray(
        sorted(jax.devices(), key=lambda device: (device.process_index, device.id))
    )

    return Mesh(devices.reshape(-1, fsdp_axis_size), ("data", "fsdp"))


def get_batch_sharding(mesh: Mesh) -> NamedSharding:
    """Shard an array's leading batch dimension across the whole mesh."""
    return NamedSharding(mesh, P(("data", "fsdp")))
