"""ChronoQuant evaluation: attention-output fidelity and end-to-end PPL."""

import torch
from typing import Dict, Optional
from chronoquant.codec import ChronoQuantCodec, ChronoQuantConfig


def evaluate_attention_fidelity(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    config: ChronoQuantConfig,
) -> Dict[str, float]:
    """Measure attention-output error with ChronoQuant compression.
    
    Args:
        Q: (n_heads, seq_len, head_dim)
        K: (n_heads, seq_len, head_dim)  
        V: (n_heads, seq_len, head_dim)
        config: ChronoQuantConfig
        
    Returns:
        Dict with error metrics
    """
    n_heads, seq_len, head_dim = Q.shape
    codec = ChronoQuantCodec(config)
    scale = head_dim ** -0.5
    
    # Baseline attention
    A_base = torch.softmax(Q @ K.transpose(-2, -1) * scale, dim=-1)
    Y_base = A_base @ V
    
    # Compress and decompress K and V
    K_hat_list, V_hat_list = [], []
    total_mem = {"compressed": 0, "baseline": 0}
    
    for h in range(n_heads):
        ck = codec.compress_sequence(K[h])
        cv = codec.compress_sequence(V[h])
        K_hat_list.append(codec.decompress_sequence(ck, head_dim))
        V_hat_list.append(codec.decompress_sequence(cv, head_dim))
        
        mk = codec.memory_bytes(ck, head_dim)
        mv = codec.memory_bytes(cv, head_dim)
        total_mem["compressed"] += mk["total_compressed"] + mv["total_compressed"]
        total_mem["baseline"] += mk["baseline_fp16"] + mv["baseline_fp16"]
    
    K_hat = torch.stack(K_hat_list)
    V_hat = torch.stack(V_hat_list)
    
    # Compressed attention
    A_hat = torch.softmax(Q @ K_hat.transpose(-2, -1) * scale, dim=-1)
    Y_hat = A_hat @ V_hat
    
    # Metrics
    output_frob = (Y_base - Y_hat).norm().item()
    output_rel = output_frob / max(1e-10, Y_base.norm().item())
    
    k_frob = (K.float() - K_hat).norm().item()
    k_rel = k_frob / max(1e-10, K.float().norm().item())
    v_frob = (V.float() - V_hat).norm().item()
    v_rel = v_frob / max(1e-10, V.float().norm().item())
    
    # Attention distribution error
    attn_kl = torch.nn.functional.kl_div(
        A_hat.clamp(min=1e-10).log(), A_base.clamp(min=1e-10),
        reduction='batchmean', log_target=False,
    ).item()
    
    return {
        "output_rel_error": output_rel,
        "output_frob_error": output_frob,
        "k_rel_error": k_rel,
        "v_rel_error": v_rel,
        "attention_kl": attn_kl,
        "compression_ratio": total_mem["baseline"] / max(1, total_mem["compressed"]),
        "compressed_bytes": total_mem["compressed"],
        "baseline_bytes": total_mem["baseline"],
    }
