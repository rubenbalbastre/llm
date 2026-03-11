# Hugging Face Post-Training

This folder contains alignment experiments implemented with Transformers and related tooling.

## What Is Implemented
- Supervised fine-tuning (SFT) for instruction following.
- Direct Preference Optimization (DPO) with preference pairs.
- RLVR-style experiments on top of SFT baselines.

## Notebooks
- `sft.ipynb`: SFT on instruction-style datasets with standard Trainer utilities.
- `DPO.ipynb`: DPO experiments without explicit reward models.
- `RLVR.ipynb`: RLVR variants using preference or verifiable rewards.
