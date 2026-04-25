"""ChronoQuant analysis: measure delta statistics on real KV traces."""

import torch
import math
from typing import Dict, List
from dataclasses import dataclass
from chronoquant.codec import ChronoQuantCodec, ChronoQuantConfig


@dataclass
class DeltaStats:
    stride: int
    mean_cosine_similarity: float
    median_cosine_similarity: float
    min_cosine_similarity: float
    p10_cosine_similarity: float
    mean_delta_relative_magnitude: float
    median_delta_relative_magnitude: float
    mean_reconstruction_error: float
    compression_ratio: float
    n_samples: int


def compute_cosine_similarity_distribution(
    kv_sequence: torch.Tensor, max_lag: int = 64,
) -> Dict[int, List[float]]:
    """Cosine similarity between tokens at various lags."""
    seq_len = kv_sequence.shape[0]
    result = {}
    for lag in range(1, min(max_lag + 1, seq_len)):
        sims = []
        for t in range(lag, seq_len):
            a = kv_sequence[t].float()
            b = kv_sequence[t - lag].float()
            cos = torch.nn.functional.cosine_similarity(
                a.unsqueeze(0), b.unsqueeze(0)
            ).item()
            sims.append(cos)
        result[lag] = sims
    return result


def compute_delta_statistics(
    kv_sequence: torch.Tensor,
    strides: List[int] = [4, 8, 16, 32, 64, 128],
    delta_bits: int = 4,
) -> List[DeltaStats]:
    """Compute delta statistics for different strides."""
    seq_len, head_dim = kv_sequence.shape
    results = []
    for stride in strides:
        if stride >= seq_len:
            continue
        config = ChronoQuantConfig(stride=stride, delta_bits=delta_bits)
        codec = ChronoQuantCodec(config)
        cosine_sims, delta_rel_mags, recon_errors = [], [], []
        for t in range(seq_len):
            if codec._is_keyframe(t):
                continue
            anchor_pos = (t // stride) * stride
            if anchor_pos >= seq_len:
                continue
            anchor = kv_sequence[anchor_pos].float()
            current = kv_sequence[t].float()
            cos = torch.nn.functional.cosine_similarity(
                current.unsqueeze(0), anchor.unsqueeze(0)
            ).item()
            cosine_sims.append(cos)
            delta = current - anchor
            an = anchor.norm().item()
            if an > 1e-10:
                delta_rel_mags.append(delta.norm().item() / an)
            codes, scale = codec._quantize_delta_symmetric(delta)
            delta_hat = codec._dequantize_delta_symmetric(codes, scale)
            recon = anchor + delta_hat
            cn = current.norm().item()
            if cn > 1e-10:
                recon_errors.append((current - recon).norm().item() / cn)
        if not cosine_sims:
            continue
        ct = torch.tensor(cosine_sims)
        dt = torch.tensor(delta_rel_mags) if delta_rel_mags else torch.tensor([0.0])
        rt = torch.tensor(recon_errors) if recon_errors else torch.tensor([0.0])
        compressed = codec.compress_sequence(kv_sequence)
        mem = codec.memory_bytes(compressed, head_dim)
        results.append(DeltaStats(
            stride=stride,
            mean_cosine_similarity=ct.mean().item(),
            median_cosine_similarity=ct.median().item(),
            min_cosine_similarity=ct.min().item(),
            p10_cosine_similarity=ct.quantile(0.1).item(),
            mean_delta_relative_magnitude=dt.mean().item(),
            median_delta_relative_magnitude=dt.median().item(),
            mean_reconstruction_error=rt.mean().item(),
            compression_ratio=mem["compression_ratio"],
            n_samples=len(cosine_sims),
        ))
    return results


def compare_with_isoquant_overhead(
    seq_len: int, head_dim: int = 256, n_layers: int = 32,
    n_kv_heads: int = 8, svd_rank: int = 32, stride: int = 64,
    delta_bits: int = 4,
) -> Dict:
    """Compare ChronoQuant vs IsoQuant memory at a given context length."""
    n_total = n_layers * n_kv_heads
    iso_meta_ph = (head_dim*svd_rank*2 + svd_rank*svd_rank*2 +
                   svd_rank*16*2 + head_dim*4 + head_dim*2)
    iso_meta = 2 * n_total * iso_meta_ph
    iso_kv = 2 * n_total * seq_len * (svd_rank * 0.5 + 2)
    iso_total = iso_meta + iso_kv
    nkf = math.ceil(seq_len / stride)
    npf = seq_len - nkf
    ckf = 2 * n_total * nkf * head_dim * 2
    cpf = 2 * n_total * npf * (head_dim * (delta_bits/8) + 4)
    c_total = ckf + cpf
    baseline = 2 * n_total * seq_len * head_dim * 2
    return {
        "baseline_fp16": {"total_mb": baseline/1e6},
        "isoquant": {
            "metadata_mb": iso_meta/1e6, "compressed_kv_mb": iso_kv/1e6,
            "total_mb": iso_total/1e6,
            "compression_ratio": baseline/iso_total,
            "metadata_fraction": iso_meta/iso_total,
        },
        "chronoquant": {
            "metadata_mb": 0.0, "keyframe_mb": ckf/1e6, "pframe_mb": cpf/1e6,
            "total_mb": c_total/1e6,
            "compression_ratio": baseline/max(1, c_total),
        },
        "context_length": seq_len,
    }
