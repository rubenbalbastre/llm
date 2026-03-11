# Pre-Training Experiments

This folder contains notebooks that build a GPT-style model from first principles and explore scaling behaviors. The goal is to cover data processing, architecture assembly, training loops, and analysis of scaling laws.

## Subfolders
- `gpt_from_scratch/`: attention math, data pipeline, and full GPT model assembly.
- `huggingface/`: a self-supervised training loop using Transformers utilities.
- `scaling_laws/`: experiments and plots that study compute, parameter, and token scaling.

## Highlights
- Step-by-step implementation of a decoder-only Transformer.
- End-to-end training with custom batching and evaluation.
- Scaling law analysis using cached binaries for reproducibility.
