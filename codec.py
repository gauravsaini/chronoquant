"""ChronoQuant Codec: I-frame/P-frame delta encoding for KV cache.

Key design choices:
- I-frames stored at FP16 (full precision keyframes)
- P-frames stored as INT4 deltas from nearest I-frame with per-tensor scale
- Anchors are always I-frames (no error propagation chains)
- Keys stored pre-RoPE to keep deltas purely semantic
- Zero static metadata overhead
"""

import torch
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict


@dataclass
class ChronoQuantConfig:
    """Configuration for ChronoQuant compression."""
    stride: int = 64          # Default keyframe stride (used if stride_k/v not set)
    stride_k: int = 0         # Key-specific stride (0 = use stride)
    stride_v: int = 0         # Value-specific stride (0 = use stride)
    delta_bits: int = 4       # Bits for delta quantization
    store_pre_rope: bool = True   # Store keys before RoPE application
    symmetric_quant: bool = True  # Symmetric quantization for deltas

    @property
    def effective_stride_k(self) -> int:
        return self.stride_k if self.stride_k > 0 else self.stride

    @property
    def effective_stride_v(self) -> int:
        return self.stride_v if self.stride_v > 0 else self.stride


@dataclass
class CompressedToken:
    """Storage for a single compressed token's K or V."""
    is_keyframe: bool
    # Keyframe data (only if is_keyframe=True)
    fp16_data: Optional[torch.Tensor] = None  # (head_dim,) FP16
    # P-frame data (only if is_keyframe=False)
    anchor_idx: Optional[int] = None           # Index of the keyframe
    delta_codes: Optional[torch.Tensor] = None # (head_dim,) INT8 codes (4-bit packed)
    delta_scale: Optional[float] = None        # Per-tensor FP16 scale


@dataclass
class CompressedSequence:
    """Full compressed KV sequence for one head."""
    config: ChronoQuantConfig
    num_tokens: int = 0
    keyframe_indices: List[int] = field(default_factory=list)
    keyframe_data: List[torch.Tensor] = field(default_factory=list)  # List of (head_dim,) FP16
    pframe_anchor_idx: List[int] = field(default_factory=list)       # anchor keyframe index for each P-frame
    pframe_delta_codes: List[torch.Tensor] = field(default_factory=list)  # INT8 codes
    pframe_delta_scales: List[float] = field(default_factory=list)
    pframe_positions: List[int] = field(default_factory=list)        # token positions of P-frames
    

class ChronoQuantCodec:
    """Encoder/decoder for ChronoQuant delta-coded KV cache."""
    
    def __init__(self, config: ChronoQuantConfig):
        self.config = config
        self.n_levels = 2 ** config.delta_bits  # 16 for 4-bit
    
    def _is_keyframe(self, token_idx: int) -> bool:
        """Determine if a token position should be a keyframe."""
        return token_idx % self.config.stride == 0
    
    def _find_anchor(self, token_idx: int, keyframe_indices: List[int]) -> int:
        """Find the nearest keyframe anchor for a P-frame token."""
        # Binary search for nearest keyframe <= token_idx
        best = keyframe_indices[0]
        for kf in keyframe_indices:
            if kf <= token_idx:
                best = kf
            else:
                break
        return best
    
    def _quantize_delta_symmetric(self, delta: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Symmetric INT4 quantization of a delta vector.
        
        Args:
            delta: (head_dim,) float tensor
            
        Returns:
            codes: (head_dim,) int8 tensor with values in [-8, 7] for 4-bit
            scale: float scalar
        """
        half_levels = self.n_levels // 2  # 8 for 4-bit
        
        # Per-tensor scale: map max absolute value to half_levels - 1
        amax = delta.abs().max().item()
        if amax < 1e-10:
            return torch.zeros_like(delta, dtype=torch.int8), 0.0
        
        scale = amax / (half_levels - 1)
        
        # Quantize: round to nearest integer level
        codes = torch.clamp(
            torch.round(delta / scale),
            -half_levels, half_levels - 1
        ).to(torch.int8)
        
        return codes, scale
    
    def _dequantize_delta_symmetric(self, codes: torch.Tensor, scale: float) -> torch.Tensor:
        """Dequantize INT4 codes back to float delta.
        
        Args:
            codes: (head_dim,) int8 tensor
            scale: float scalar
            
        Returns:
            delta: (head_dim,) float tensor
        """
        return codes.float() * scale
    
    def compress_sequence(
        self, 
        kv_sequence: torch.Tensor,
    ) -> CompressedSequence:
        """Compress a full KV sequence for one head.
        
        Args:
            kv_sequence: (seq_len, head_dim) tensor of K or V vectors
            
        Returns:
            CompressedSequence with I-frames and P-frames
        """
        seq_len, head_dim = kv_sequence.shape
        compressed = CompressedSequence(
            config=self.config,
            num_tokens=seq_len,
        )
        
        for t in range(seq_len):
            if self._is_keyframe(t):
                # I-frame: store full FP16
                compressed.keyframe_indices.append(t)
                compressed.keyframe_data.append(kv_sequence[t].to(torch.float16))
            else:
                # P-frame: store delta from nearest keyframe
                anchor_pos = self._find_anchor(t, compressed.keyframe_indices)
                anchor_kf_idx = compressed.keyframe_indices.index(anchor_pos)
                anchor_data = compressed.keyframe_data[anchor_kf_idx].float()
                
                delta = kv_sequence[t].float() - anchor_data
                codes, scale = self._quantize_delta_symmetric(delta)
                
                compressed.pframe_anchor_idx.append(anchor_kf_idx)
                compressed.pframe_delta_codes.append(codes)
                compressed.pframe_delta_scales.append(scale)
                compressed.pframe_positions.append(t)
        
        return compressed
    
    def decompress_sequence(self, compressed: CompressedSequence, head_dim: int) -> torch.Tensor:
        """Decompress a CompressedSequence back to full KV tensor.
        
        Args:
            compressed: CompressedSequence
            head_dim: dimension of each KV vector
            
        Returns:
            (seq_len, head_dim) tensor
        """
        output = torch.zeros(compressed.num_tokens, head_dim, dtype=torch.float32)
        
        # Place keyframes
        for kf_idx, kf_pos in enumerate(compressed.keyframe_indices):
            output[kf_pos] = compressed.keyframe_data[kf_idx].float()
        
        # Reconstruct P-frames
        for i, pf_pos in enumerate(compressed.pframe_positions):
            anchor_kf_idx = compressed.pframe_anchor_idx[i]
            anchor = compressed.keyframe_data[anchor_kf_idx].float()
            delta = self._dequantize_delta_symmetric(
                compressed.pframe_delta_codes[i],
                compressed.pframe_delta_scales[i]
            )
            output[pf_pos] = anchor + delta
        
        return output
    
    def memory_bytes(self, compressed: CompressedSequence, head_dim: int) -> Dict[str, int]:
        """Calculate exact memory usage of a compressed sequence."""
        n_keyframes = len(compressed.keyframe_indices)
        n_pframes = len(compressed.pframe_positions)
        
        # Keyframe: head_dim * 2 bytes (FP16) + 2 bytes index
        keyframe_bytes = n_keyframes * (head_dim * 2 + 2)
        
        # P-frame: head_dim * 0.5 bytes (INT4) + 2 bytes scale + 2 bytes anchor_idx
        pframe_bytes = n_pframes * (head_dim // 2 + 2 + 2)
        
        # Static metadata: 0 bytes!
        metadata_bytes = 0
        
        # Baseline comparison
        baseline_bytes = compressed.num_tokens * head_dim * 2  # FP16
        
        return {
            "keyframe_bytes": keyframe_bytes,
            "pframe_bytes": pframe_bytes,
            "metadata_bytes": metadata_bytes,
            "total_compressed": keyframe_bytes + pframe_bytes + metadata_bytes,
            "baseline_fp16": baseline_bytes,
            "compression_ratio": baseline_bytes / max(1, keyframe_bytes + pframe_bytes),
            "n_keyframes": n_keyframes,
            "n_pframes": n_pframes,
        }


def compress_kv(
    keys: torch.Tensor,
    values: torch.Tensor,
    config: ChronoQuantConfig,
) -> Tuple[List[CompressedSequence], List[CompressedSequence]]:
    """Compress batched KV tensors.
    
    Args:
        keys: (batch, n_heads, seq_len, head_dim) 
        values: (batch, n_heads, seq_len, head_dim)
        config: ChronoQuantConfig
        
    Returns:
        Tuple of (compressed_keys, compressed_values), each a list over heads
    """
    codec = ChronoQuantCodec(config)
    B, H, S, D = keys.shape
    
    compressed_k = []
    compressed_v = []
    
    for h in range(H):
        # Use first batch element for now (extend for multi-batch later)
        k_seq = keys[0, h]  # (S, D)
        v_seq = values[0, h]  # (S, D)
        
        compressed_k.append(codec.compress_sequence(k_seq))
        compressed_v.append(codec.compress_sequence(v_seq))
    
    return compressed_k, compressed_v


def decompress_kv(
    compressed_k: List[CompressedSequence],
    compressed_v: List[CompressedSequence],
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decompress KV sequences back to tensors.
    
    Args:
        compressed_k: list of CompressedSequence for keys
        compressed_v: list of CompressedSequence for values
        head_dim: dimension per head
        
    Returns:
        keys: (1, n_heads, seq_len, head_dim)
        values: (1, n_heads, seq_len, head_dim)
    """
    codec = ChronoQuantCodec(compressed_k[0].config)
    
    k_list = []
    v_list = []
    for ck, cv in zip(compressed_k, compressed_v):
        k_list.append(codec.decompress_sequence(ck, head_dim))
        v_list.append(codec.decompress_sequence(cv, head_dim))
    
    keys = torch.stack(k_list, dim=0).unsqueeze(0)    # (1, H, S, D)
    values = torch.stack(v_list, dim=0).unsqueeze(0)  # (1, H, S, D)
    
    return keys, values
