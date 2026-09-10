"""Small reusable pytree operations.

See docs/components.md#tree-and-array-utilities.
"""

from collections.abc import Mapping, Sequence

import jax
from flax import traverse_util

from utils.typing_utils import PyTree


def tree_add(left: PyTree, right: PyTree) -> PyTree:
    """Add matching leaves from two pytrees."""
    return jax.tree.map(lambda x, y: x + y, left, right)


def tree_zeros(tree: PyTree) -> PyTree:
    """Create a zero-filled pytree with the same structure and leaf types."""
    return jax.tree.map(jax.numpy.zeros_like, tree)


def tree_labels(
    tree: PyTree, groups: Mapping[str, Sequence[str]], default: str
) -> PyTree:
    """Label leaves by the first group whose substring occurs in their path."""
    labels = {}
    for path in traverse_util.flatten_dict(tree):
        name = "/".join(map(str, path))
        labels[path] = next(
            (
                label
                for label, patterns in groups.items()
                if any(pattern in name for pattern in patterns)
            ),
            default,
        )
    return traverse_util.unflatten_dict(labels)
