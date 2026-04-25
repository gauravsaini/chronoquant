"""ChronoQuant HuggingFace cache integration for end-to-end PPL evaluation.

Integrates ChronoQuant delta encoding into HF's DynamicCache so we can
measure real perplexity impact via teacher-forced evaluation.
"""

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer
from chronoquant.codec import ChronoQuantConfig, ChronoQuantCodec


class ChronoQuantCacheLayer(DynamicLayer):
    """A DynamicLayer that applies ChronoQuant compression on-the-fly.
    
    On each update(), incoming K/V states are compressed via delta encoding
    and immediately decompressed. This "quantize-dequantize" loop simulates
    the reconstruction error that would be seen in a real compressed cache,
    allowing us to measure PPL impact.
    """
    
    def __init__(self, config: ChronoQuantConfig, component: str = "both"):
        """
        Args:
            config: ChronoQuantConfig 
            component: "both", "k", or "v" — which component to compress
        """
        super().__init__()
        self.chrono_config = config
        self.codec_k = ChronoQuantCodec(
            ChronoQuantConfig(
                stride=config.effective_stride_k,
                delta_bits=config.delta_bits,
                symmetric_quant=config.symmetric_quant,
            )
        )
        self.codec_v = ChronoQuantCodec(
            ChronoQuantConfig(
                stride=config.effective_stride_v,
                delta_bits=config.delta_bits,
                symmetric_quant=config.symmetric_quant,
            )
        )
        self.component = component
    
    def _compress_decompress(self, states: torch.Tensor, codec: ChronoQuantCodec) -> torch.Tensor:
        """Apply quantize-dequantize loop to simulate compression error.
        
        Args:
            states: (batch, n_heads, seq_len, head_dim)
            codec: ChronoQuantCodec instance
            
        Returns:
            Reconstructed states with same shape
        """
        orig_dtype = states.dtype
        device = states.device
        B, H, S, D = states.shape
        
        out = torch.zeros_like(states, dtype=torch.float32)
        
        for b in range(B):
            for h in range(H):
                seq = states[b, h].float().cpu()  # (S, D)
                compressed = codec.compress_sequence(seq)
                reconstructed = codec.decompress_sequence(compressed, D)
                out[b, h] = reconstructed.to(device)
        
        return out.to(orig_dtype)
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Intercept KV update, apply ChronoQuant compression."""
        if self.component in ("both", "k"):
            key_states = self._compress_decompress(key_states, self.codec_k)
        if self.component in ("both", "v"):
            value_states = self._compress_decompress(value_states, self.codec_v)
        
        return super().update(key_states, value_states, *args, **kwargs)


class ChronoQuantCache(DynamicCache):
    """DynamicCache with ChronoQuant compression on selected layers."""
    
    def __init__(
        self,
        config=None,
        num_layers: int = None,
        chrono_config: ChronoQuantConfig = None,
        compressed_layers: set[int] = None,
        **kwargs,
    ):
        """
        Args:
            config: HF model config (optional)
            num_layers: total number of layers
            chrono_config: ChronoQuant configuration
            compressed_layers: set of layer indices to compress (None = all)
        """
        if config is not None:
            kwargs["config"] = config
        super().__init__(**kwargs)
        
        chrono_config = chrono_config or ChronoQuantConfig()
        
        if len(self.layers) == 0:
            if num_layers is None:
                raise ValueError("Must provide either config or num_layers")
            for _ in range(num_layers):
                self.layers.append(DynamicLayer())
        
        # Replace selected layers with ChronoQuant layers
        for i in range(len(self.layers)):
            if compressed_layers is None or i in compressed_layers:
                sliding = getattr(self.layers[i], "sliding_window", None)
                new_layer = ChronoQuantCacheLayer(chrono_config)
                if sliding is not None:
                    new_layer.sliding_window = sliding
                self.layers[i] = new_layer
    
    def has_previous_state(self, layer_idx: int = None) -> bool:
        return self.get_seq_length() > 0
