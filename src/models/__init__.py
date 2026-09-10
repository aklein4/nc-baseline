"""Typed model construction selected by Hydra configuration.

References:
- docs/components.md#model-abstraction
- https://hydra.cc/docs/advanced/instantiate_objects/overview/
"""

from hydra.utils import get_class
from omegaconf import DictConfig

from models.base import CustomModel


def make_model(config: DictConfig) -> CustomModel:
    Model = get_class(config._target_)
    return Model(Model.config_type.from_config(config))
