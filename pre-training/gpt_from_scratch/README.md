# GPT From Scratch

This folder builds a GPT-like model incrementally. It starts with attention math, adds a data pipeline, and culminates in a full training loop with evaluation.

## What Is Implemented
- Scaled dot-product attention with masking and shape checks.
- Data preprocessing and batching for next-token prediction.
- Decoder block assembly with layer norm, MLP, and residuals.
- Training loop, loss tracking, and evaluation hooks.

## Notebooks
- `attention.ipynb`: derives attention equations and implements the tensor operations.
- `data_processing.ipynb`: tokenizes and batches the Edu-FineWeb10B shards for efficient training.
- `gpt.ipynb`: combines blocks into a full GPT model with optimizer and evaluation.
- `gpt-2.ipynb`: scales configuration to a GPT-2 class size to study depth and width effects.
