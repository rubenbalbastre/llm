# Post-Training and Alignment

This folder collects alignment techniques after pre-training, including instruction tuning, preference optimization, and RLVR methods.

## Subfolders
- `instruction-fine-tuning/`: instruction tuning workflows and chat formatting.
- `text-classification/`: downstream fine-tuning for classification tasks.
- `huggingface/`: SFT, DPO, and RLVR notebooks using Transformers tooling.
- `reinforcement_learning_verifiable_rewards/`: GRPO-based RLVR from scratch.

## Highlights
- Instruction tuning with chat templates and task-specific tokens.
- Preference-based optimization without explicit reward models.
- RLVR experiments that use verifiable rewards for math tasks.
