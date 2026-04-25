"""ChronoQuant: Delta-encoded KV cache compression.

Video-compression-inspired I-frame/P-frame architecture for KV cache.
Zero static metadata overhead. Constant compression ratio across all context lengths.
"""

from chronoquant.codec import (
    ChronoQuantConfig,
    ChronoQuantCodec,
    compress_kv,
    decompress_kv,
)
from chronoquant.analysis import (
    compute_delta_statistics,
    compute_cosine_similarity_distribution,
)

__all__ = [
    "ChronoQuantConfig",
    "ChronoQuantCodec",
    "compress_kv",
    "decompress_kv",
    "compute_delta_statistics",
    "compute_cosine_similarity_distribution",
]
