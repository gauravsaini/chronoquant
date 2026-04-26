# ChronoQuant PyTorch Trace Pipeline

This directory contains the PyTorch implementation of the **ChronoQuant** compression framework.

## Overview
While `chronoquant_mlx` handles end-to-end inference and native hardware acceleration on Apple Silicon, this directory is dedicated to **trace-level evaluation and conceptual validation**. It acts as the algorithmic testbed.

Before deploying custom Metal/CUDA kernels, the ChronoQuant architecture (including residual drift validation and baseline precision analysis) was evaluated here to isolate mathematical distortion from framework overhead.

## Key Usage
- Extracting specific Q/K/V states from single layers of a model using PyTorch hooks.
- Simulating quantizers mathematically to measure Inner-product distortion ($D_{prod}$) and Mean Squared Error ($D_{mse}$).
- Comparing exact layer-wise distortion against standard baselines (e.g., TurboQuant).

*Note: For actual model evaluation (Perplexity, Needle-in-a-Haystack, Tokens/s), use the `scripts/` directory targeting the MLX runtime.*
