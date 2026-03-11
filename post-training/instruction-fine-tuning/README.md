# Instruction Fine-Tuning

This folder focuses on instruction tuning and chat-style formatting.

## What Is Implemented
- Dataset loading and formatting into chat prompts.
- Tokenizer extension with chat-specific tokens.
- Supervised fine-tuning of a GPT-2 class model.
- Lightweight evaluation with MT-Bench and HellaSwag for sanity checks.

## Notebooks
- `instruction_fine_tuning.ipynb`: instruction-tunes GPT-2 (124M) on the Nemotron Instruction-Following dataset and logs quick evaluations.
