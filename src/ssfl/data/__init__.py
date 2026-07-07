"""Mini-N-BaIoT data core: one-time cache builder, scenario partitioners, loaders.

Build the cache explicitly with ``uv run python -m ssfl.data.build``; then use
the loader functions re-exported here — they are the only sanctioned data
access path for the rest of the project.
"""

from ssfl.data.loader import (
    CacheMissingError,
    load_arrays,
    load_client,
    load_meta,
    load_open,
    load_partition,
    load_splits,
    load_test,
    num_clients,
)

__all__ = [
    "CacheMissingError",
    "load_arrays",
    "load_client",
    "load_meta",
    "load_open",
    "load_partition",
    "load_splits",
    "load_test",
    "num_clients",
]
