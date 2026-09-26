"""Strata: vector search engine (C++ core, Python bindings).

Quick reference (full docs in each class's docstring and in strata/_core.pyi):

    import numpy as np
    import strata

    index = strata.BruteForceIndex(dim=128, metric="l2")
    index.add(base)                                   # (n, 128) float32: used without copying
    ids, dists = index.search_batch(queries, k=10)    # (nq, 10) each; GIL released

Conventions:
- float32 C-contiguous inputs are not copied; other dtypes/layouts are converted with one copy.
- Results are (ids, distances); missing results are id -1, distance +inf (FAISS convention).
- Distances are lower-is-closer for every metric: squared L2, negated dot, 1 - cosine.
- Searches release the GIL and share the index's lock; add()/remove() take it exclusively, so
  inserts from Python threads are serialized.
"""

from __future__ import annotations

from typing import Any

from strata._core import (
    Analyzer,
    AttributeTable,
    Bm25Index,
    BruteForceIndex,
    CompiledFilter,
    Filter,
    PqIndex,
    ProductQuantizer,
    build_info,
    cosine_distance,
    distances,
    fuse_rrf,
    fuse_weighted,
    has_hnsw,
    inner_product,
    l2_squared,
    porter_stem,
)

if has_hnsw:
    from strata._core import HnswIndex
else:

    class HnswIndex:  # type: ignore[no-redef]
        """HNSW index (not built: src/index/hnsw.cpp does not exist yet).

        The bindings switch to the real implementation automatically once the file exists and
        the package is rebuilt (`pip install -e .`).
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError(
                "HNSW core not built: src/index/hnsw.cpp does not exist yet. Add it, then "
                "rebuild with `pip install -e .` (CMake picks it up automatically)."
            )


from strata.hybrid import HybridIndex

__all__ = [
    "Analyzer",
    "AttributeTable",
    "Bm25Index",
    "BruteForceIndex",
    "CompiledFilter",
    "Filter",
    "HnswIndex",
    "HybridIndex",
    "PqIndex",
    "ProductQuantizer",
    "build_info",
    "cosine_distance",
    "distances",
    "fuse_rrf",
    "fuse_weighted",
    "has_hnsw",
    "inner_product",
    "l2_squared",
    "porter_stem",
]
