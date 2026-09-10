"""Repository paths and environment values.

See docs/configuration.md and docs/components.md#shared-utilities.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_PATH = Path(__file__).resolve().parents[2]

LOCAL_DATA_PATH = REPO_PATH / "local_data"
CHECKPOINTS_PATH = LOCAL_DATA_PATH / "checkpoints"
JAX_CACHE_DIR = LOCAL_DATA_PATH / "jax_cache"


load_dotenv(REPO_PATH / ".env")

HF_ID = os.getenv("HF_ID")
