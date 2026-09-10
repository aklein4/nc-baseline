"""Small extensible multi-host JAX training entry point.

Repository guides:
- docs/architecture.md
- docs/configuration.md

Upstream APIs:
- https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/
- https://docs.jax.dev/en/latest/multi_process.html
- https://orbax.readthedocs.io/en/latest/guides/checkpoint/
"""

import logging
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path

import hydra
import jax
import numpy as np
import wandb
from hydra.utils import get_class
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from omegaconf import DictConfig, OmegaConf

from models import make_model
from utils import constants
from utils.checkpoints import (
    create_hf_repo,
    open_checkpoint,
)
from utils.data_utils import batches
from utils.git_utils import get_current_commit_hash
from utils.mesh_utils import create_mesh, get_batch_sharding, initialize_distributed
from utils.sharding_utils import place_parameters

logger = logging.getLogger(__name__)


def global_batch(
    local_batch: Mapping[str, np.ndarray], sharding: NamedSharding
) -> dict[str, jax.Array]:
    """Place one process-local NumPy batch into globally sharded JAX arrays."""
    return jax.tree.map(
        lambda value: jax.make_array_from_process_local_data(
            sharding, np.asarray(value)
        ),
        local_batch,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    logger.info("Starting training...")
    trainer_config = config.trainer
    resume = trainer_config.checkpoint.load_state
    if resume:
        if not config.initialization:
            raise ValueError(
                "trainer.checkpoint.load_state requires an initialization checkpoint"
            )
        if config.initialization_step is None:
            raise ValueError(
                "trainer.checkpoint.load_state requires initialization_step"
            )

    # load the model
    model = make_model(config.model)
    logger.info("Model loaded.")

    # configure the environment
    model.configure_environment()
    dist_kwargs = initialize_distributed(config.distributed)
    logger.info("Environment configured: %s", dist_kwargs)

    # jax cache location
    constants.JAX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(constants.JAX_CACHE_DIR))

    # set up the mesh
    mesh = create_mesh(config.mesh.fsdp_axis_size)
    batch_sharding = get_batch_sharding(mesh)
    if trainer_config.global_batch_size % jax.device_count():
        raise ValueError(
            "trainer.global_batch_size must be divisible by the global device count"
        )
    local_batch_size = (
        trainer_config.global_batch_size
        * jax.local_device_count()
        // jax.device_count()
    )
    logger.info("Created mesh: %s", str(mesh))
    logger.info("Local batch size: %s", local_batch_size)

    if jax.process_index() == 0:
        print("\n ===== Configuration ===== \n")
        print(OmegaConf.to_yaml(config, resolve=True))
        print(f"devices: {jax.device_count()} ({jax.process_count()} processes)")
        print(" ========================= \n", flush=True)

    run_dir = Path(trainer_config.run_dir).expanduser().resolve()
    git_commit = get_current_commit_hash()

    repo_id = None
    if not trainer_config.debug and trainer_config.huggingface.enabled:
        if not constants.HF_ID:
            raise ValueError("Set HF_ID or run tpu_startup.sh")
        repo_id = f"{constants.HF_ID}/{trainer_config.wandb.project}_{config.name}"
        if jax.process_index() == 0:
            create_hf_repo(repo_id, trainer_config.huggingface.private)
        multihost_utils.sync_global_devices("huggingface_repo_ready")

    # Resolve once; resume restores the full state instead of loading params twice.
    checkpoint_context = (
        open_checkpoint(config.initialization, step=config.initialization_step)
        if config.initialization
        else nullcontext()
    )
    with checkpoint_context as checkpoint:
        params = model.initialize_params(config.seed)
        if checkpoint is not None and not resume:
            params = model.load_params(params, checkpoint)
        params = place_parameters(params, mesh, config.model.sharding)

        with jax.set_mesh(mesh):
            trainer = get_class(trainer_config._target_)(
                model=model,
                config=trainer_config,
                params=params,
            )
            if resume:
                trainer.load_state(checkpoint)
                logger.info("Restored step %d", trainer.step)

    wandb_run = None
    if jax.process_index() == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        parameter_count = sum(value.size for value in jax.tree.leaves(trainer.params))
        logger.info("Initialized %s parameters", f"{parameter_count:,}")
        if trainer_config.wandb.enabled:
            notes = f"GIT HASH: {git_commit}"
            if config.get("notes"):
                notes += f"\n\n{config.notes}"
            wandb_run = wandb.init(
                project=trainer_config.wandb.project,
                name=config.name,
                notes=notes,
                config=OmegaConf.to_container(config, resolve=True),
            )

    local_batches = batches(config.data, local_batch_size, config.seed, trainer.step)
    training_batches = (
        global_batch(local_batch, batch_sharding) for local_batch in local_batches
    )

    with jax.set_mesh(mesh):
        trainer.train(
            training_batches,
            metadata={
                "git_commit": git_commit,
                "wandb_run_id": wandb_run.id if wandb_run is not None else None,
                "config": OmegaConf.to_container(config, resolve=True),
            },
            repo_id=repo_id,
            wandb_run=wandb_run,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    for dependency in ("datasets", "httpx", "huggingface_hub"):
        logging.getLogger(dependency).setLevel(logging.WARNING)
    main()
