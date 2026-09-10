"""Small shared typing aliases.

See docs/components.md#shared-utilities and JAX pytrees:
https://docs.jax.dev/en/latest/pytrees.html
"""

from typing import Any, TypeAlias

PyTree: TypeAlias = Any
"""
A type alias for a PyTree, which is a nested structure of lists, tuples, and dictionaries
containing JAX arrays or other PyTrees.
"""
