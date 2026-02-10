"""spatial_index – tile-based spatial bucketing (5 km tiles).

The indexing logic lives on Highway. This module re-exports Highway as the
entry point until the index is extracted into its own class.
"""

from .highway import Highway

__all__ = ["Highway"]
